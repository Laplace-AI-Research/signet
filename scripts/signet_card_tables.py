"""Per-size evaluation tables for the Signet model cards (2026-11-23).

    python scripts/signet_card_tables.py > release/signet/card_tables.md

Test split, greedy, non-thinking, SYSTEM_BASE. Values with 95% problem-group bootstrap intervals from metrics.json;
channel gaps from c2_analysis_test*.json (Signet) or the base runs (via signet_family_table.gaps).
"""
import importlib.util
import json
from pathlib import Path

R = Path("results")
spec = importlib.util.spec_from_file_location("fam", Path(__file__).parent / "signet_family_table.py")
fam = importlib.util.module_from_spec(spec); spec.loader.exec_module(fam)

SIZES = {"2B": ("B_Qwen3_5-2B", "SIG_2B_r1"), "4B": ("B_Qwen3_5-4B", "SIG_4B_r1"), "9B": ("B_Qwen3_5-9B", "SIG_9B_a6w250")}
ROWS = [
    ("`test_id` final accuracy", "test_id", "final_accuracy"),
    ("`test_template` final accuracy", "test_template", "final_accuracy"),
    ("`test_structural` final accuracy (held-out types)", "test_structural", "final_accuracy"),
    ("`test_id` pressure flip rate ↓", "test_id", "pressure_flip_rate"),
    ("`test_structural` pressure flip rate ↓", "test_structural", "pressure_flip_rate"),
    ("`test_structural` legitimate correction", "test_structural", "legitimate_correction_rate"),
    ("`test_id` forged-record resistance", "test_id", "fabricated_evidence_resistance"),
    ("`test_structural` forged-record resistance", "test_structural", "fabricated_evidence_resistance"),
    ("`test_id` insufficient-evidence accuracy", "test_id", "insufficient_accuracy"),
]


def cell(m, s, k):
    v = m.get(s, {}).get("all", {}).get(k)
    if not v or v.get("value") is None:
        return "—"
    return f"{v['value']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"


for size, (base, sig) in SIZES.items():
    mb, ms = (json.load(open(R / r / "metrics.json")) for r in (base, sig))
    print(f"## Signet-{size}\n\n| Metric | Qwen3.5-{size} | Signet-{size} |\n|---|---|---|")
    for label, s, k in ROWS:
        print(f"| {label} | {cell(mb, s, k)} | {cell(ms, s, k)} |")
    bp, bs = fam.gaps(base); sp, ss = fam.gaps(sig)
    print(f"| Channel gap, pure (tool − user-turn adoption) [lo] | {bp} | {sp} |")
    print(f"| Channel gap, never-trained paste styles [lo] | {bs} | {ss} |\n")
