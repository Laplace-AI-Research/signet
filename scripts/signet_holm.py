"""Holm correction across the confirmatory tests for the released Signet models (2026-11-23).

    python scripts/signet_holm.py      # writes results/comparisons/signet_holm.json

Family: for each released model (Signet-2B, -4B, -9B), five tests on the test splits:
  * pure-channel gap above the pre-registered bar of +0.30 (one-sided)
  * paste-style gap above the pre-registered bar of +0.20 (one-sided)
  * held-out final accuracy vs the off-the-shelf base (two-sided)
  * held-out pressure-flip rate vs the base (two-sided)
  * held-out forged-record resistance vs the base (two-sided)
p-values come from 10,000 problem-group bootstrap resamples, with the (k + 1) / (n + 1) convention so that no p-value
is reported as zero; Holm's step-down procedure is then applied to all 15 tests.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("a", ROOT / "scripts/analysis_c2.py")
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
from conviction.evaluation.statistics import METRICS, paired_bootstrap  # noqa: E402

R = ROOT / "results"
N_BOOT = 10_000
MODELS = {"Signet-2B": ("SIG_2B_r1", "B_Qwen3_5-2B", "SIG_2B_a6"), "Signet-4B": ("SIG_4B_r1", "B_Qwen3_5-4B", "SIG_4B_a6w250"),
          "Signet-9B": ("SIG_9B_a6w250", "B_Qwen3_5-9B", "SIG_9B_a6w250")}


def gap_draws(pos, neg):
    clusters = sorted({c for c, _ in pos}); ci = {c: i for i, c in enumerate(clusters)}
    a, b, n = (np.zeros(len(clusters)) for _ in range(3))
    for (c, x), (_, y) in zip(pos, neg):
        a[ci[c]] += x; b[ci[c]] += y; n[ci[c]] += 1
    draws = np.random.default_rng(0).integers(0, len(clusters), size=(N_BOOT, len(clusters)))
    return float((a.sum() - b.sum()) / n.sum()), (a[draws].sum(1) - b[draws].sum(1)) / n[draws].sum(1)


def pairs_pure(orig, ctrl):
    pos, neg = [], []
    for eid, t in orig.items():
        if t["slot"] not in ("c_trusted", "w_trusted"):
            continue
        ch = ctrl.get(eid.replace("_trusted", "_channel"))
        if ch is None or t["expected_final_answer"] is None:
            continue
        v, cl = A._norm(t["expected_final_answer"]), f"{t['split']}:{t['group_id']}"
        pos.append((cl, A._norm(t["pred_answer"]) == v)); neg.append((cl, A._norm(ch["pred_answer"]) == v))
    return pos, neg


def pairs_style(c2):
    pos, neg = [], []
    for eid, t in c2.items():
        if not eid.endswith("_tool") or t["slot"][:2] not in ("c_", "w_"):
            continue
        p = c2.get(eid[: -len("_tool")] + "_paste")
        if p is None or t["expected_final_answer"] is None:
            continue
        v, cl = A._norm(t["expected_final_answer"]), f"{t['split']}:{t['group_id']}"
        pos.append((cl, A._norm(t["pred_answer"]) == v)); neg.append((cl, A._norm(p["pred_answer"]) == v))
    return pos, neg


def one_sided(d, bar):
    return (int((d <= bar).sum()) + 1) / (len(d) + 1)


def main():
    tests = []
    for name, (run, base, _) in MODELS.items():
        orig, ctrl = A.load_preds(R / run, A.TEST3), A.load_preds(R / run / "controls", A.TEST3)
        g, d = gap_draws(*pairs_pure(orig, ctrl))
        tests.append({"model": name, "test": "pure-channel gap > +0.30", "estimate": g, "p": one_sided(d, 0.30)})
        g, d = gap_draws(*pairs_style(A.load_preds(R / run / "controls2", A.TEST3)))
        tests.append({"model": name, "test": "paste-style gap > +0.20", "estimate": g, "p": one_sided(d, 0.20)})
        a = A.load_preds(R / base, ["test_structural"]); b = A.load_preds(R / run, ["test_structural"])
        ms = [m for m in METRICS if m.name in ("final_accuracy", "pressure_flip_rate", "fabricated_evidence_resistance")]
        pb = paired_bootstrap(list(a.values()), list(b.values()), metrics=ms, n_boot=N_BOOT)
        for m, label in (("final_accuracy", "held-out accuracy vs base"), ("pressure_flip_rate", "held-out flips vs base"),
                         ("fabricated_evidence_resistance", "held-out forged rejection vs base")):
            r = pb[m]
            # two-sided percentile p with the (k + 1) / (n + 1) floor
            p = max(r["p_boot"], 2 / (N_BOOT + 1))
            tests.append({"model": name, "test": label, "estimate": r["diff"], "p": min(1.0, p)})
    # Holm step-down
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p"])
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (len(tests) - rank) * tests[i]["p"])
        running = max(running, adj)
        tests[i]["p_holm"] = running
    out = {"n_tests": len(tests), "n_boot": N_BOOT, "alpha": 0.05, "tests": tests,
           "all_reject": all(t["p_holm"] < 0.05 for t in tests), "max_p_holm": max(t["p_holm"] for t in tests)}
    (R / "comparisons" / "signet_holm.json").write_text(json.dumps(out, indent=1) + "\n")
    for t in tests:
        print(f"{t['model']:10s} {t['test']:36s} est={t['estimate']:+.3f} p={t['p']:.5f} holm={t['p_holm']:.5f}")
    print("all reject at 0.05:", out["all_reject"], "| max Holm p:", out["max_p_holm"])


if __name__ == "__main__":
    main()
