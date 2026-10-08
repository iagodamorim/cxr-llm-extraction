"""
cost_estimation.py
===================

Estimate per-vendor USD cost of the LLM inference runs based on:
  - token counts estimated from the saved prompt and responses
  - vendor list-price (per 1M tokens, as of May 2026)

Token estimation:
  - OpenAI: tiktoken cl100k_base (gpt-4 family)
  - Gemini / Anthropic: same tokenizer as a reasonable proxy
    (true counts differ by ~10% but pricing assumptions matter more
     than fractional token differences for an order-of-magnitude figure)

Pricing assumed (USD per 1M tokens, list price, May 2026):

  Vendor               input    output
  ----------------------------------
  GPT-4.1              2.00     8.00
  GPT-4.1-mini         0.40     1.60
  Gemini 2.5 Pro       1.25     10.00
  Gemini 2.5 Flash     0.30     2.50
  Claude Sonnet 4.6    3.00     15.00
  Claude Haiku 4.5     1.00     5.00

NOTES:
  - Anthropic ephemeral prompt caching was enabled in the inference pipeline;
    Anthropic charges 0.10x for cached read input tokens. For long shared
    system prompts (~6000 tokens) this can drop effective input cost ~10x
    once warm. We report BOTH gross (no-cache) and cached-effective USD.
  - The system prompt (cxr_prompt_v4.yaml) is a constant per call; the
    report text and the JSON response vary per call.
  - Output is also reported per 1000 reports, which is the unit used
    in the manuscript.

Inputs:
  --prompt-file        prompts/cxr_prompt_v4.yaml
  --responses-dir      responses/
  --reports-csv        data/outputs/reports_master.csv  (for input report text)
  --output             data/outputs/cost_estimates.csv

Author: IPDA et al. PAP v2.1 Phase C+
"""

import argparse
import csv
from glob import glob
from pathlib import Path

import pandas as pd
import tiktoken


# ---- pricing table (USD per 1M tokens, May 2026 list price) ----
PRICING = {
    # vendor key (matches *_test.csv basename) : (input_per_M, output_per_M)
    "gpt-4-1":          (2.00,  8.00),
    "gpt-4-1-mini":     (0.40,  1.60),
    "gemini-2-5-pro":   (1.25,  10.00),
    "gemini-2-5-flash": (0.30,  2.50),
    "sonnet-4-6":       (3.00,  15.00),
    "haiku-4-5":        (1.00,  5.00),
}

# Anthropic ephemeral cache pricing multiplier for INPUT only
# Cache read = 0.10x base; cache write (first call) = 1.25x base.
# For 6 models * (295 dev + 780 test) = 6450 calls with shared system prompt:
# 1 write per model = 6 writes total + 1069 reads per model.
# Effective input cost per call after cache warm = 0.10x base.
ANTHROPIC_CACHED_INPUT_MULTIPLIER = 0.10


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt-file", default="prompts/cxr_prompt_v4.yaml")
    ap.add_argument("--responses-dir", default="responses")
    ap.add_argument("--reports-csv", default="data/outputs/reports_master.csv")
    ap.add_argument("--output", default="data/outputs/cost_estimates.csv")
    args = ap.parse_args()

    enc = tiktoken.get_encoding("cl100k_base")

    # ---- prompt size ----
    prompt_text = Path(args.prompt_file).read_text()
    prompt_tokens = len(enc.encode(prompt_text))
    print(f"System prompt ({args.prompt_file}): {prompt_tokens:,} tokens, "
          f"{len(prompt_text):,} chars")

    # ---- reports (input variable) ----
    reports = pd.read_csv(args.reports_csv)
    reports = reports.set_index("accession_number")
    reports["report"] = reports["report"].fillna("").astype(str)
    print(f"Reports indexed: {len(reports)}")

    # Build report-text token table once
    rep_tokens = {acc: len(enc.encode(txt)) for acc, txt in reports["report"].items()}
    rep_tok_series = pd.Series(rep_tokens, name="report_tokens")
    print(f"Report-text tokens: median={rep_tok_series.median():.0f}, "
          f"mean={rep_tok_series.mean():.0f}, "
          f"p95={rep_tok_series.quantile(0.95):.0f}, "
          f"max={rep_tok_series.max():.0f}")

    rows = []
    for split in ("dev", "test"):
        print(f"\n=== Split: {split} ===")
        for fpath in sorted(glob(str(Path(args.responses_dir) / f"*_{split}.csv"))):
            name = Path(fpath).name
            if "smoke" in name.lower():
                continue
            vendor = name.replace(f"_{split}.csv", "")
            if vendor not in PRICING:
                print(f"  WARN: no pricing for {vendor}, skipping")
                continue
            df = pd.read_csv(fpath)
            df = df[df["split"] == split]
            df["raw_response"] = df["raw_response"].fillna("").astype(str)
            n_calls = len(df)

            # Output tokens (response)
            df["out_tok"] = df["raw_response"].apply(lambda s: len(enc.encode(s)))
            out_total = int(df["out_tok"].sum())
            out_mean = float(df["out_tok"].mean())

            # Input tokens = prompt + per-report report text
            # join report token counts
            df = df.merge(rep_tok_series, left_on="accession_number",
                          right_index=True, how="left")
            df["report_tokens"] = df["report_tokens"].fillna(0).astype(int)
            df["in_tok"] = prompt_tokens + df["report_tokens"]
            in_total = int(df["in_tok"].sum())
            in_mean = float(df["in_tok"].mean())

            in_price, out_price = PRICING[vendor]
            in_cost_gross = in_total / 1_000_000 * in_price
            out_cost = out_total / 1_000_000 * out_price
            total_gross = in_cost_gross + out_cost

            # Anthropic cached price for the constant system prompt fraction
            is_anthropic = vendor.startswith(("sonnet", "haiku"))
            if is_anthropic:
                # input = prompt + report; only the prompt fraction caches
                cacheable = prompt_tokens * n_calls
                non_cacheable = in_total - cacheable
                in_cost_cached = (
                    cacheable / 1_000_000 * in_price * ANTHROPIC_CACHED_INPUT_MULTIPLIER +
                    non_cacheable / 1_000_000 * in_price
                )
                total_cached = in_cost_cached + out_cost
            else:
                in_cost_cached = in_cost_gross
                total_cached = total_gross

            usd_per_1000_gross = total_gross / n_calls * 1000
            usd_per_1000_cached = total_cached / n_calls * 1000

            rows.append({
                "vendor": vendor,
                "split": split,
                "n_calls": n_calls,
                "prompt_tokens_const": prompt_tokens,
                "input_tokens_total": in_total,
                "input_tokens_mean_per_call": round(in_mean, 1),
                "output_tokens_total": out_total,
                "output_tokens_mean_per_call": round(out_mean, 1),
                "price_input_per_M_USD": in_price,
                "price_output_per_M_USD": out_price,
                "input_cost_USD_gross": round(in_cost_gross, 4),
                "input_cost_USD_cached": round(in_cost_cached, 4),
                "output_cost_USD": round(out_cost, 4),
                "total_cost_USD_gross": round(total_gross, 4),
                "total_cost_USD_cached_anthropic_only": round(total_cached, 4),
                "USD_per_1000_reports_gross": round(usd_per_1000_gross, 3),
                "USD_per_1000_reports_cached": round(usd_per_1000_cached, 3),
            })
            print(f"  {vendor:18s}  N={n_calls:4d}  "
                  f"in={in_total:>9,}  out={out_total:>7,}  "
                  f"total $ = {total_gross:6.3f} (gross), {total_cached:6.3f} (cached)  "
                  f"=> {usd_per_1000_gross:5.2f}/1000 gross, "
                  f"{usd_per_1000_cached:5.2f}/1000 cached")

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nWrote: {args.output}  ({len(out)} rows)")

    # ---- Test-only consolidated summary (the manuscript figure) ----
    print(f"\n=== Test-split summary (USD per 1000 reports) ===")
    test = out[out["split"] == "test"].copy()
    test = test.sort_values("USD_per_1000_reports_cached")
    print(f"{'vendor':18s}  {'gross USD/1k':>15s}  {'cached USD/1k':>15s}")
    for _, r in test.iterrows():
        print(f"  {r['vendor']:18s}  {r['USD_per_1000_reports_gross']:>13.2f}  "
              f"{r['USD_per_1000_reports_cached']:>13.2f}")

    # Total run cost across all 6 models, both splits
    total_run_gross = out["total_cost_USD_gross"].sum()
    total_run_cached = out["total_cost_USD_cached_anthropic_only"].sum()
    print(f"\n=== Total inference run cost (6 models × 1075 reports = 6450 calls) ===")
    print(f"  Gross (no caching):                   USD {total_run_gross:7.2f}")
    print(f"  With Anthropic ephemeral caching:     USD {total_run_cached:7.2f}")


if __name__ == "__main__":
    main()
