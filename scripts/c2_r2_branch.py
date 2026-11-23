"""Pre-registered R2 branch (research/EXPERIMENTS.md, 2026-11-22) from R1's test analysis.

    python scripts/c2_r2_branch.py results/C2_R1/c2_analysis_test.json   -> prints none|r2a|r2b|r2c

R-a = test_structural final accuracy non-inferior to Vector (lower bound of difference > -0.03)
R-b = paste-style gap lower bound >= 0.20
R-c = test_id forged-record resistance higher than Vector (CI lower bound > 0)
"""
import json
import sys


def branch(a: dict) -> str:
    ra = (a.get("H-2d_structural_acc_vs_vector") or {}).get("lo", -1) > -0.03
    rb = (a.get("H-2b_style") or {}).get("lo", -1) >= 0.20
    rc = (a.get("H-2c_vs_vector") or {}).get("lo", -1) > 0
    if ra and rb and rc:
        return "none"
    if not ra and rb:
        return "r2c"
    if ra and not rb:
        return "r2a"
    if not ra and not rb:
        return "r2b"
    return "r2a"  # only R-c failed


if __name__ == "__main__":
    a = json.load(open(sys.argv[1]))
    print(branch(a), json.dumps({k: a.get(k) for k in ("H-2d_structural_acc_vs_vector", "H-2c_vs_vector")})[:300],
          file=sys.stderr)
    print(branch(a))
