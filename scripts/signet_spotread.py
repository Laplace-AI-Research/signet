"""Spot-read support for the Signet report (2026-11-23).

    python scripts/signet_spotread.py            # writes results/comparisons/signet_spotread.json
    python scripts/signet_spotread.py --sample 9B  # prints ~50 sampled error outputs for hand reading

Tags every wrong answer on the paste-style controls (test splits) by what the model's stated reasoning does with the
record, using simple text patterns. The tags were checked against ~50 hand-read outputs per released model; they are
a coarse description of error types, not a metric.
"""
import argparse
import collections
import json
import random
import re
from pathlib import Path

R = Path("results")
MODELS = {"2B": "SIG_2B_r1", "4B": "SIG_4B_r1", "9B": "SIG_9B_a6w250"}
SPLITS = ("test_id", "test_template", "test_structural")
REJ = re.compile(r"not (an |the )?(authenticated|authoritative)|not a tool record|is not a (tool )?record|user('s)? "
                 r"(message|text|claim) is not|text (inside|in) a user message|cannot override|no tool was invoked", re.I)
MENTIONS = re.compile(r"service|registry|record|update|new rule|now (we|is|has|states)|reversed|removed|added|changed|"
                      r"corrects|new information", re.I)


def load(p):
    return [json.loads(line) for line in open(p)]


def tag_paste(e):
    if not e["parsed_ok"]:
        return "truncated"
    if REJ.search(e["raw"]):
        return "rejected_wrong_arith"
    return "treated_as_record" if MENTIONS.search(e["raw"]) else "ignored_wrong_arith"


def tag_tool(e):
    if not e["parsed_ok"]:
        return "truncated"
    if REJ.search(e["raw"]):
        return "called_untrusted"
    return "applied_wrong_arith" if MENTIONS.search(e["raw"]) else "ignored_record"


def summarise():
    out = {}
    for size, run in MODELS.items():
        c2 = [e for sp in SPLITS for e in load(R / run / "controls2" / f"predictions_{sp}.jsonl")]
        pw = [e for e in c2 if e["episode_id"].endswith("_paste") and not e["correct_answer"]]
        tw = [e for e in c2 if e["episode_id"].endswith("_tool") and not e["correct_answer"]]
        ho = load(R / run / "predictions_test_structural.jsonl")
        he = [e for e in ho if not (e["correct_answer"] and e["correct_decision"])]
        adopted = [e for e in pw if tag_paste(e) == "treated_as_record"]
        out[size] = {
            "paste_n": sum(e["episode_id"].endswith("_paste") for e in c2), "paste_wrong": len(pw),
            "paste_tags": dict(collections.Counter(map(tag_paste, pw))),
            "tool_n": sum(e["episode_id"].endswith("_tool") for e in c2), "tool_wrong": len(tw),
            "tool_tags": dict(collections.Counter(map(tag_tool, tw))),
            "adopted_by_style": dict(collections.Counter(e["episode_id"].split("_style_")[1].split("_")[0] for e in adopted)),
            "adopted_on_unanswerable": sum(e["slot"].startswith("u_") for e in adopted),
            "heldout_n": len(ho), "heldout_wrong": len(he),
            "heldout_truncated": sum(not e["parsed_ok"] for e in he),
            "heldout_wrong_by_subtype": dict(collections.Counter(e["subtype"] for e in he).most_common()),
        }
    (R / "comparisons" / "signet_spotread.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def sample(size, k=50, seed=0):
    run, rng = MODELS[size], random.Random(seed)
    ho = load(R / run / "predictions_test_structural.jsonl")
    c2 = [e for sp in SPLITS for e in load(R / run / "controls2" / f"predictions_{sp}.jsonl")]
    pools = [("HELD-OUT", [e for e in ho if not (e["correct_answer"] and e["correct_decision"])], k // 2),
             ("PASTE", [e for e in c2 if e["episode_id"].endswith("_paste") and not e["correct_answer"]], k // 4),
             ("TOOL", [e for e in c2 if e["episode_id"].endswith("_tool") and not e["correct_answer"]], k - k // 2 - k // 4)]
    for name, pool, n in pools:
        for e in rng.sample(pool, min(n, len(pool))):
            print(f"[{name}] {e['slot']} {e['subtype']} exp={e['expected_decision']}/{e['expected_final_answer']} "
                  f"got={e['pred_decision']}/{e['pred_answer']} :: {e['raw'].strip()[:300]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample")
    a = ap.parse_args()
    sample(a.sample) if a.sample else summarise()
