"""Signet release verification verdicts (pre-registered 2026-11-22, research/EXPERIMENTS.md).

    python scripts/signet_verify.py 2B   -> prints V-S, V-P, V-R for that size and writes results/SIGV_<size>/verdicts.json
"""
import json
import sys
from pathlib import Path

R = Path("results")
CFG = {"2B": ("SIG_2B_r1", "SIG_2B_s2", "B_Qwen3_5-2B"), "4B": ("SIG_4B_r1", "SIG_4B_s2", "B_Qwen3_5-4B"),
       "9B": ("SIG_9B_a6w250", "SIG_9B_s2", "B_Qwen3_5-9B")}
SPLITS = ["test_id", "test_template", "test_structural"]


def acc(path, split):
    m = json.load(open(path))
    return m[split]["all"]["final_accuracy"]["value"]


def style(run):
    for n in ("c2_analysis_test_vs_base.json", "c2_analysis_test.json"):
        p = R / run / n
        if p.exists():
            return json.load(open(p))["H-2b_style"]


def main(size):
    s1, s2, base = CFG[size]
    v = R / f"SIGV_{size}"
    out = {}
    # V-S: seed 2 held-out acc within 0.05 of seed 1, and paste-style gap lower bound >= 0.50
    a1, a2 = acc(R / s1 / "metrics.json", "test_structural"), acc(R / s2 / "metrics.json", "test_structural")
    st2 = style(s2)
    out["V-S"] = {"seed1_heldout_acc": round(a1, 4), "seed2_heldout_acc": round(a2, 4), "diff": round(a2 - a1, 4),
                  "seed2_paste_gap": st2["gap"], "seed2_paste_lo": st2["lo"],
                  "pass": abs(a2 - a1) <= 0.05 and st2["lo"] >= 0.50}
    # V-P: per split, (public - private) for Signet within 0.05 of the same for its base
    vp = {}
    for sp in SPLITS:
        sig_pub, sig_priv = acc(R / s1 / "metrics.json", sp), acc(v / "private_signet/metrics.json", sp)
        b_pub, b_priv = acc(R / base / "metrics.json", sp), acc(v / "private_base/metrics.json", sp)
        vp[sp] = {"signet_public": round(sig_pub, 4), "signet_private": round(sig_priv, 4),
                  "base_public": round(b_pub, 4), "base_private": round(b_priv, 4),
                  "signet_drop": round(sig_pub - sig_priv, 4), "base_drop": round(b_pub - b_priv, 4),
                  "ok": abs((sig_pub - sig_priv) - (b_pub - b_priv)) <= 0.05}
    out["V-P"] = {"splits": vp, "pass": all(x["ok"] for x in vp.values())}
    # V-R: paraphrased prompt, held-out acc drop <= 0.05
    p_acc = acc(v / "paraphrase/metrics.json", "test_structural")
    out["V-R"] = {"base_prompt": round(a1, 4), "paraphrase": round(p_acc, 4), "drop": round(a1 - p_acc, 4),
                  "pass": (a1 - p_acc) <= 0.05}
    (v / "verdicts.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
