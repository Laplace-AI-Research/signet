"""Conviction Signet family: one table, built from results/ (2026-11-22).

    python scripts/signet_family_table.py > results/comparisons/signet_family.md

Test-split numbers only (held-out problem types = test_structural). Channel gaps come from c2_analysis_test*.json:
pure = control C construction, paste = never-trained paste styles (v0.1_controls2). Missing runs are skipped.
"""
import json
from pathlib import Path

R = Path("results")
ROWS = [  # (label, run dir, kind)
    ("Anchor 1.7B (released)", "E2_sft", "released"),
    ("Meridian 1.7B (released)", "V2", "released"),
    ("Vector 1.7B (released)", "E3b_grpo_base", "released"),
    ("A6 / Signet recipe @ Qwen3-1.7B", "C2_A6", "a6"),
    ("R1 RL-only @ Qwen3-1.7B", "C2_R1", "r1"),
    ("R1 RL-only @ Qwen3-1.7B, seed 2", "SIG_1.7B_r1s2", "r1"),
    ("R2a RL-only 800 steps @ Qwen3-1.7B", "SIG_1.7B_r2a", "r1"),
    ("Qwen3.5-2B (off the shelf)", "B_Qwen3_5-2B", "base"),
    ("A6 recipe @ Qwen3.5-2B", "SIG_2B_a6", "a6"),
    ("RL-only @ Qwen3.5-2B", "SIG_2B_r1", "r1"),
    ("Qwen3.5-4B (off the shelf)", "B_Qwen3_5-4B", "base"),
    ("A6 recipe @ Qwen3.5-4B (250-ep warm-start)", "SIG_4B_a6w250", "a6"),
    ("A6 warm-start only @ Qwen3.5-4B", "SIG_4B_a6_sftonly", "a6"),
    ("RL-only @ Qwen3.5-4B", "SIG_4B_r1", "r1"),
    ("Qwen3.5-9B (off the shelf)", "B_Qwen3_5-9B", "base"),
    ("A6 recipe @ Qwen3.5-9B (250-ep warm-start)", "SIG_9B_a6w250", "a6"),
    ("A6 warm-start only @ Qwen3.5-9B", "SIG_9B_a6_sftonly", "a6"),
    ("RL-only @ Qwen3.5-9B", "SIG_9B_r1", "r1"),
]
COLS = [("held-out acc", "test_structural", "final_accuracy"), ("held-out flips ↓", "test_structural", "pressure_flip_rate"),
        ("held-out forged res", "test_structural", "fabricated_evidence_resistance"),
        ("held-out legit corr", "test_structural", "legitimate_correction_rate"),
        ("test_id forged res", "test_id", "fabricated_evidence_resistance"), ("strict fmt", "test_structural", "strict_format_rate")]


def metric(m, s, k):
    v = m.get(s, {}).get("all", {}).get(k)
    return "—" if not v or v.get("value") is None else f"{v['value']:.3f}"


CTRL_C = json.load(open(R / "comparisons/robustness_C_P_S.json"))["C"]["models"] if (R / "comparisons/robustness_C_P_S.json").exists() else {}
BASE_KEY = {"E2_sft": "E2_sft", "V2": "V2", "E3b_grpo_base": "E3b_grpo_base", "B_Qwen3_5-2B": "B_Qwen3_5-2B",
            "B_Qwen3_5-4B": "B_Qwen3_5-4B", "B_Qwen3_5-9B": "B_Qwen3_5-9B"}
BASE_CTRL2 = {"B_Qwen3_5-2B": ["SIG_2B_a6", "SIG_2B_r1"], "B_Qwen3_5-4B": ["SIG_4B_a6w250", "SIG_4B_r1"],
              "B_Qwen3_5-9B": ["SIG_9B_a6w250", "SIG_9B_r1"]}


def base_gaps(run):
    """Released adapters / off-the-shelf bases: pure gap from control C; paste gap from a ladder run's base_controls2."""
    import importlib.util
    pure = CTRL_C.get(BASE_KEY.get(run, ""), {}).get("channel_gap")
    pure_s = "—" if not pure else f"{pure['diff']:+.3f} [{pure['lo']:+.2f}]"
    paste_s = "—"
    for sig in BASE_CTRL2.get(run, []):
        d = R / sig / "base_controls2"
        if (d / "predictions_test_id.jsonl").exists():
            spec = importlib.util.spec_from_file_location("a", Path(__file__).parent / "analysis_c2.py")
            a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
            g = a.style_gap(a.load_preds(d, a.TEST3))
            paste_s = f"{g['gap']:+.3f} [{g['lo']:+.2f}]"
            break
    return pure_s, paste_s


def gaps(run):
    if run in BASE_KEY:
        return base_gaps(run)
    for name in ("c2_analysis_test_vs_base.json", "c2_analysis_test.json"):
        p = R / run / name
        if p.exists():
            a = json.load(open(p))
            f = lambda x: "—" if not x else f"{x['gap']:+.3f} [{x['lo']:+.2f}]"  # noqa: E731
            return f(a.get("H-2a_pure_channel")), f(a.get("H-2b_style"))
    return "—", "—"


def main():
    head = ["model"] + [c[0] for c in COLS] + ["pure-channel gap [lo]", "paste-style gap [lo]"]
    print("| " + " | ".join(head) + " |"); print("|" + "---|" * len(head))
    for label, run, _ in ROWS:
        p = R / run / "metrics.json"
        if not p.exists():
            continue
        m = json.load(open(p))
        print("| " + " | ".join([label] + [metric(m, s, k) for _, s, k in COLS] + list(gaps(run))) + " |")


if __name__ == "__main__":
    main()
