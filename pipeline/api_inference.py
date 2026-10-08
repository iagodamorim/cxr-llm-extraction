"""
api_inference.py
=================

Runs prompt v4 (cxr_prompt_v4.yaml) against one of three frontier LLMs
on every report in the master CSV. Saves all responses (parsed + raw)
to a per-vendor CSV with incremental progress so you can safely resume
on failure.

Usage:
    python api_inference.py --vendor gpt4o   --input reports_master.csv
    python api_inference.py --vendor gemini  --input reports_master.csv
    python api_inference.py --vendor claude  --input reports_master.csv

Optional flags:
    --prompt prompts/cxr_prompt_v4.yaml   (default)
    --output {vendor}_responses.csv  (default; auto-named per vendor)
    --limit N                      (test on N rows; default = all)
    --only-split dev               (run only dev or test rows)
    --model-id MODEL               (override default model identifier)
    --resume                       (skip rows already in output csv)

Required environment variables:
    OPENAI_API_KEY      (for vendor=gpt4o)
    GOOGLE_API_KEY      (for vendor=gemini)
    ANTHROPIC_API_KEY   (for vendor=claude)

Required packages:
    pip install openai google-genai anthropic pyyaml pandas tqdm

Author: I.P. D'Amorim et al. PAP v2.1 (locked 2026-05-18).
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

DEFAULT_MODEL_IDS = {
    "gpt4o":  "gpt-4.1",
    "gemini": "gemini-2.5-pro",
    "claude": "claude-sonnet-4-6",
}

V4_CLASSES = ["opacity", "pneumonia", "cardiomegaly", "effusion",
              "pneumothorax", "mass", "yesfinding"]


def load_prompt(yaml_path: Path) -> str:
    """Concatenate the 6 components of prompt v3 into a single system prompt."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tpl = cfg["templates"]
    parts = []
    for key in ("cxr.intro", "cxr.rules", "cxr.classes", "cxr.output", "cxr.examples"):
        if key in tpl:
            parts.append(tpl[key]["template"].strip())
    return "\n\n".join(parts)


def build_user_message(report_text: str, content_tpl: str) -> str:
    return content_tpl.replace("{report}", report_text)


# ----- Vendor adapters -------------------------------------------------------

def call_gpt4o(system_prompt: str, user_msg: str, model_id: str, max_retries: int = 5):
    from openai import OpenAI
    client = OpenAI()
    last_err = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model_id,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            )
            latency = time.time() - t0
            text = resp.choices[0].message.content
            return text, latency, None
        except Exception as e:
            last_err = e
            sleep = 2 ** attempt
            time.sleep(sleep)
    return None, None, str(last_err)


def call_gemini(system_prompt: str, user_msg: str, model_id: str, max_retries: int = 5):
    from google import genai
    from google.genai import types
    client = genai.Client()
    last_err = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = client.models.generate_content(
                model=model_id,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            latency = time.time() - t0
            text = resp.text
            return text, latency, None
        except Exception as e:
            last_err = e
            sleep = 2 ** attempt
            time.sleep(sleep)
    return None, None, str(last_err)


def call_claude(system_prompt: str, user_msg: str, model_id: str, max_retries: int = 5):
    import anthropic
    client = anthropic.Anthropic()
    last_err = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = client.messages.create(
                model=model_id,
                max_tokens=1024,
                temperature=0,
                system=[{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_msg}],
            )
            latency = time.time() - t0
            text = resp.content[0].text
            return text, latency, None
        except Exception as e:
            last_err = e
            sleep = 2 ** attempt
            time.sleep(sleep)
    return None, None, str(last_err)


VENDOR_CALL = {"gpt4o": call_gpt4o, "gemini": call_gemini, "claude": call_claude}


# ----- Parser ----------------------------------------------------------------

def parse_response(raw_text):
    """Parse the JSON response. Return (parsed_dict, error_msg)."""
    if raw_text is None:
        return None, "empty response"
    s = raw_text.strip()
    # Strip ``` fences if present
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # Try to extract the first {...} block if model wrapped it in prose
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        s = m.group(0)
    try:
        d = json.loads(s)
    except Exception as e:
        return None, f"json_parse_error: {e}"
    return d, None


def extract_class_predictions(parsed: dict) -> dict:
    """Map parsed JSON into per-class binary predictions + abnormalities_found.

    Per the v4 prompt, when pneumonia=1, opacity MUST also be 1 (hierarchical
    entailment). We do NOT silently fix violations — instead we track them in
    `entailment_violation` for transparent reporting (MI-CLEAR-LLM Items 3/5).
    Downstream analyses can choose to report raw F1, violation rate, or
    enforced F1 separately.
    """
    out = {c: None for c in V4_CLASSES}
    out["abnormalities_found"] = ""
    if not isinstance(parsed, dict):
        out["entailment_violation"] = False
        return out
    findings = parsed.get("findings", [])
    for f in findings:
        if not isinstance(f, dict): continue
        cname = str(f.get("class_name", "")).strip().lower()
        if cname in V4_CLASSES:
            try:
                out[cname] = int(str(f.get("presence", "0")).strip())
            except (ValueError, TypeError):
                out[cname] = None
            if cname == "yesfinding":
                out["abnormalities_found"] = f.get("abnormalities_found", "") or ""
    out["entailment_violation"] = (out.get("pneumonia") == 1 and out.get("opacity") != 1)
    return out


# ----- Main loop -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vendor", required=True, choices=["gpt4o", "gemini", "claude"])
    ap.add_argument("--input", required=True, help="reports_master.csv")
    ap.add_argument("--prompt", default="prompts/cxr_prompt_v4.yaml")
    ap.add_argument("--output", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-split", choices=["dev", "test"], default=None)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.output or f"{args.vendor}_responses.csv")
    model_id = args.model_id or DEFAULT_MODEL_IDS[args.vendor]

    df = pd.read_csv(args.input)
    if args.only_split:
        df = df[df["split"] == args.only_split].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)

    # Resume support
    done = set()
    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path)
        done = set(prev["accession_number"].astype(str).tolist())
        print(f"Resume: {len(done)} reports already in {out_path}; skipping those.")
    else:
        # Write header
        hdr = ["accession_number", "split", "vendor", "model_id", "timestamp",
               "raw_response", "parse_ok", "parse_error", "latency_s",
               *V4_CLASSES, "abnormalities_found", "entailment_violation"]
        pd.DataFrame(columns=hdr).to_csv(out_path, index=False)

    print(f"Vendor: {args.vendor} | Model: {model_id}")
    print(f"Prompt: {args.prompt}")
    print(f"Input:  {args.input}  ({len(df)} reports, after filters)")
    print(f"Output: {out_path}  (resume mode: {args.resume})")

    sys_prompt = load_prompt(Path(args.prompt))
    with open(args.prompt, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    content_tpl = cfg["templates"]["cxr.content"]["template"]

    caller = VENDOR_CALL[args.vendor]

    todo = df[~df["accession_number"].astype(str).isin(done)].reset_index(drop=True)
    print(f"To process: {len(todo)} reports\n")

    for _, row in tqdm(todo.iterrows(), total=len(todo), desc=args.vendor):
        acc = str(row["accession_number"])
        split = row["split"]
        report_text = str(row["report"])
        user_msg = build_user_message(report_text, content_tpl)

        raw, lat, err = caller(sys_prompt, user_msg, model_id)
        timestamp = datetime.now(timezone.utc).isoformat()

        parsed, parse_err = (None, "no_response") if raw is None else parse_response(raw)
        preds = extract_class_predictions(parsed or {})

        out_row = {
            "accession_number": acc,
            "split": split,
            "vendor": args.vendor,
            "model_id": model_id,
            "timestamp": timestamp,
            "raw_response": raw if raw else f"ERROR: {err}",
            "parse_ok": parsed is not None,
            "parse_error": parse_err or "",
            "latency_s": round(lat, 3) if lat else None,
            **preds,
        }
        pd.DataFrame([out_row]).to_csv(out_path, mode="a", header=False, index=False)

    print(f"\nDone. Wrote: {out_path}")
    final = pd.read_csv(out_path)
    print(f"Total rows in output: {len(final)}")
    print(f"Parse-ok rate: {100 * final['parse_ok'].mean():.1f}%")


if __name__ == "__main__":
    main()
