"""Pre-registered selection between A4 and A6 (research/EXPERIMENTS.md, 2026-11-22), from val analyses only.

    python scripts/c2_select.py results/C2_A4/c2_analysis_val.json results/C2_A6/c2_analysis_val.json   -> prints a4|a6

A6 is chosen iff its held-out paste-style gap lower bound (H-2b construction) is higher than A4's AND its val final
accuracy is at least A4's minus 0.03. Otherwise A4.
"""
import json
import sys


def choose(a4: dict, a6: dict) -> str:
    better_channel = a6["H-2b_style"]["lo"] > a4["H-2b_style"]["lo"]
    acc_ok = a6["final_accuracy"] >= a4["final_accuracy"] - 0.03
    return "a6" if (better_channel and acc_ok) else "a4"


if __name__ == "__main__":
    a4, a6 = (json.load(open(p)) for p in sys.argv[1:3])
    print(choose(a4, a6))
