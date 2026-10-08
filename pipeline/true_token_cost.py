"""
true_token_cost.py  (resumable)
==============================
TRUE per-vendor token cost, replacing the cl100k_base estimate.

Reconstructs the EXACT payload api_inference.py sent:
  system = assembled components (intro+rules+classes+output+examples)
  user   = cxr.content template with {report} substituted
INPUT tokens (dominant): exact per-vendor tokenizer
  OpenAI    -> tiktoken o200k_base (offline) [+7 chat envelope]
  Anthropic -> messages.count_tokens(system,[user]) (exact billed input)
  Google    -> models.count_tokens(contents=user, system_instruction=system)
OUTPUT tokens (minor term): counted offline with o200k from saved raw_response.
Pricing + Anthropic prompt-cache (0.10x on the constant system fraction).

Resumable: caches Anthropic/Google input counts to JSON; safe to re-run.
Each invocation works for ~WALL seconds then flushes and exits 2 if unfinished,
0 when the final CSV is written.
"""
import os, sys, time, json, threading
from glob import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd, yaml, tiktoken

ROOT = Path(__file__).resolve().parent.parent
PROMPT = ROOT/"prompts/cxr_prompt_v4.yaml"
REPORTS = ROOT/"data/outputs/reports_master.csv"
RESP_DIR = ROOT/"responses"
OUT = ROOT/"data/outputs/true_token_cost.csv"
CACHE_A = ROOT/"data/outputs/_ttc_anth_input.json"
CACHE_G = ROOT/"data/outputs/_ttc_goog_input.json"
LOG = ROOT/"data/outputs/_ttc_progress.log"
WALL = 38  # seconds budget per invocation

PRICING = {"gpt-4-1":(2.00,8.00),"gpt-4-1-mini":(0.40,1.60),
           "gemini-2-5-pro":(1.25,10.00),"gemini-2-5-flash":(0.30,2.50),
           "sonnet-4-6":(3.00,15.00),"haiku-4-5":(1.00,5.00)}
VENDOR_OF={"gpt-4-1":"openai","gpt-4-1-mini":"openai","gemini-2-5-pro":"google",
           "gemini-2-5-flash":"google","sonnet-4-6":"anthropic","haiku-4-5":"anthropic"}
ANTH_CACHE_MULT=0.10
o200=tiktoken.get_encoding("o200k_base")

def log(m):
    with open(LOG,"a") as f: f.write(m+"\n")
    print(m, flush=True)

def load_system():
    c=yaml.safe_load(open(PROMPT,encoding="utf-8"))["templates"]
    parts=[c[k]["template"].strip() for k in
           ("cxr.intro","cxr.rules","cxr.classes","cxr.output","cxr.examples") if k in c]
    return "\n\n".join(parts), c["cxr.content"]["template"]

def retry(fn,tries=6):
    for i in range(tries):
        try: return fn()
        except Exception as e:
            if i==tries-1: raise
            time.sleep(min(2**i,15))

def jload(p): return json.loads(p.read_text()) if p.exists() else {}
def jsave(p,d): p.write_text(json.dumps(d))

def fill_cache(cache_path, accs, user_msg, counter, label):
    cache=jload(cache_path)
    todo=[a for a in accs if a not in cache]
    if not todo:
        log(f"{label}: already complete ({len(cache)})"); return cache, True
    log(f"{label}: {len(cache)} cached, {len(todo)} to do")
    t0=time.time(); flush_at=time.time()
    lock=threading.Lock(); done=0
    def work(a):
        return a, counter(user_msg(a))
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs=ex.map(work, todo)
        for a,v in futs:
            with lock:
                cache[a]=v; done+=1
            if time.time()-flush_at>5:
                jsave(cache_path,cache); flush_at=time.time()
            if time.time()-t0>WALL:
                jsave(cache_path,cache)
                log(f"{label}: budget hit, {len(cache)}/{len(accs)} cached")
                return cache, len(cache)>=len(accs)
    jsave(cache_path,cache)
    log(f"{label}: DONE {len(cache)} in {time.time()-t0:.0f}s")
    return cache, True

def main():
    SYS,CONTENT=load_system()
    sys_o200=len(o200.encode(SYS))
    reps=pd.read_csv(REPORTS).set_index("accession_number")["report"].fillna("").astype(str)
    call_accs={"dev":set(),"test":set()}
    for f in glob(str(RESP_DIR/"*_*.csv")):
        if "smoke" in f.lower(): continue
        split="test" if f.endswith("_test.csv") else "dev"
        df=pd.read_csv(f,usecols=["accession_number","split"])
        call_accs[split]|=set(df[df.split==split]["accession_number"].astype(str))
    uniq=sorted(call_accs["dev"]|call_accs["test"])
    def user_msg(a): return CONTENT.replace("{report}", reps.get(a,""))
    log(f"system o200k={sys_o200} chars={len(SYS)} | uniq reports={len(uniq)}")

    import anthropic; from google import genai; from google.genai import types
    ac=anthropic.Anthropic(); gc=genai.Client()
    def anth_c(u): return retry(lambda: ac.messages.count_tokens(
        model="claude-haiku-4-5", system=SYS,
        messages=[{"role":"user","content":u}])).input_tokens
    def goog_c(u): return retry(lambda: gc.models.count_tokens(
        model="gemini-2.5-flash", contents=u,
        config=types.CountTokensConfig(system_instruction=SYS))).total_tokens

    in_anth, okA = fill_cache(CACHE_A, uniq, user_msg, anth_c, "anthropic-input")
    if not okA: sys.exit(2)
    in_goog, okG = fill_cache(CACHE_G, uniq, user_msg, goog_c, "google-input")
    if not okG: sys.exit(2)

    in_openai={a: sys_o200 + len(o200.encode(user_msg(a))) + 7 for a in uniq}
    INPUT={"openai":in_openai,"anthropic":in_anth,"google":in_goog}
    # exact billed system-token count per vendor (for cache fraction)
    sys_tok_anth=anth_c(" ")  # ~ system + empty
    rows=[]
    for split in ("dev","test"):
        for f in sorted(glob(str(RESP_DIR/f"*_{split}.csv"))):
            vk=Path(f).name.replace(f"_{split}.csv","")
            if vk not in PRICING or "smoke" in f.lower(): continue
            vend=VENDOR_OF[vk]
            df=pd.read_csv(f); df=df[df.split==split].copy()
            df["raw_response"]=df["raw_response"].fillna("").astype(str)
            accs=df["accession_number"].astype(str).tolist()
            in_total=int(sum(INPUT[vend].get(a,0) for a in accs))
            out_total=int(df["raw_response"].apply(lambda s: len(o200.encode(s))).sum())
            n=len(df); inp,outp=PRICING[vk]
            in_gross=in_total/1e6*inp; out_cost=out_total/1e6*outp
            if vend=="anthropic":
                cache_tok=sys_tok_anth*n; noncache=in_total-cache_tok
                in_cached=cache_tok/1e6*inp*ANTH_CACHE_MULT + noncache/1e6*inp
            else:
                in_cached=in_gross
            rows.append(dict(vendor=vk,split=split,n_calls=n,
                input_tokens_total=in_total,input_tokens_mean=round(in_total/n,1),
                output_tokens_total=out_total,output_tokens_mean=round(out_total/n,1),
                price_in=inp,price_out=outp,
                input_USD_gross=round(in_gross,4),input_USD_cached=round(in_cached,4),
                output_USD=round(out_cost,4),
                total_USD_gross=round(in_gross+out_cost,4),
                total_USD_cached=round(in_cached+out_cost,4),
                USD_per_1000_gross=round((in_gross+out_cost)/n*1000,3),
                USD_per_1000_cached=round((in_cached+out_cost)/n*1000,3)))
    out=pd.DataFrame(rows); out.to_csv(OUT,index=False)
    log(f"WROTE {OUT}")
    t=out[out.split=='test'].sort_values('USD_per_1000_cached')
    log("=== TEST split USD/1000 reports (TRUE tokens) ===")
    for _,r in t.iterrows():
        log(f"  {r['vendor']:16s} gross {r['USD_per_1000_gross']:6.2f} cached {r['USD_per_1000_cached']:6.2f}  in/call {r['input_tokens_mean']:.0f} out/call {r['output_tokens_mean']:.0f}")
    log(f"TOTAL run gross USD {out.total_USD_gross.sum():.2f} | cached USD {out.total_USD_cached.sum():.2f}")
    log("STATUS: COMPLETE")

if __name__=="__main__":
    main()
