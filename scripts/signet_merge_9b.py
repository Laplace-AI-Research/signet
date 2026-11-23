"""Build the Signet-9B merged checkpoint and check it against the adapter path (2026-11-23).

Run on a GPU pod (loads a 9B model twice; not for the Mac):

    python scripts/signet_merge_9b.py --adapters release/signet/hf/Signet-9B/adapters --out merged_9b

1. Loads Qwen/Qwen3.5-9B text-only at the pinned revision, merges stage1_sft then stage2_grpo, saves bf16 safetensors
   shards plus the base tokenizer to --out.
2. Parity check: reloads --out and compares greedy generations and last-token logits with the in-memory merged model
   on a few fixed prompts (val split, so no test items). Exits non-zero if any generation differs.
Writes --out/merge_report.json with the shard hashes and the check results.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE, REV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def prompts(tok, n):
    from conviction.prompts import build_messages  # episode -> chat messages with SYSTEM_BASE
    from conviction.training.data import load_jsonl
    eps = load_jsonl("data/conviction_bench/v0.1/val.jsonl")[:: 97][:n]
    out = []
    for e in eps:
        msgs = build_messages(e["messages"], "base")
        out.append(tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False))
    return out


def run(model, tok, texts):
    gens, logits = [], []
    for t in texts:
        enc = tok(t, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            logits.append(model(**enc).logits[0, -1].float().cpu())
            g = model.generate(**enc, max_new_tokens=200, do_sample=False, temperature=None, top_p=None, top_k=None)
        gens.append(tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return gens, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_check", type=int, default=8)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(BASE, revision=REV, dtype=torch.bfloat16, device_map="cuda")
    for stage in ("stage1_sft", "stage2_grpo"):
        model = PeftModel.from_pretrained(model, f"{a.adapters}/{stage}").merge_and_unload()
    model.eval()
    out = Path(a.out)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(out)
    texts = prompts(tok, a.n_check)
    g_ref, l_ref = run(model, tok, texts)
    del model
    torch.cuda.empty_cache()
    re = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16, device_map="cuda").eval()
    g_new, l_new = run(re, tok, texts)
    same = [x == y for x, y in zip(g_ref, g_new)]
    maxdiff = max(float((x - y).abs().max()) for x, y in zip(l_ref, l_new))
    shards = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.safetensors"))}
    rep = {"base": BASE, "revision": REV, "adapters_in_order": ["stage1_sft", "stage2_grpo"],
           "generations_identical": f"{sum(same)}/{len(same)}", "max_last_token_logit_diff": maxdiff,
           "shards_sha256": shards, "sample_generations": g_new[:2]}
    json.dump(rep, open(out / "merge_report.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))
    raise SystemExit(0 if all(same) else 1)


if __name__ == "__main__":
    main()
