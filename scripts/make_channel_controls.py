"""Build the channel-test control episodes (pre-registered in research/EXPERIMENTS.md, 2026-11-14).

For every c_/w_ trusted episode in test_id, test_template and test_structural that has a forged twin, write two
variants whose correct reply is the forged twin's (the record is not authenticated, so it must not be adopted):

* `*_channel`  (pure channel): the trusted twin's exact conversation, with the tool turn removed and its record text
  placed at the start of the final user message. Same record, same framing, same pointer sentence; only the
  channel differs.
* `*_nomarker`: the original forged episode with the forger's authority marker (e.g. "[SYSTEM VERIFIED]") and the word
  "official" removed from the final user message.

    python scripts/make_channel_controls.py [split ...]   ->   data/conviction_bench/v0.1_controls/{split}.jsonl
(default: the three test splits; Conviction 2 also builds `val` for arm selection)
"""
import copy
import json
import re
import sys
from pathlib import Path

from conviction.data.templates import FORGE_MARKERS

SRC = Path("data/conviction_bench/v0.1")
OUT = Path("data/conviction_bench/v0.1_controls")
SPLITS = ["test_id", "test_template", "test_structural"]


def strip_marker(text: str) -> str:
    for m in FORGE_MARKERS:
        text = text.replace(m, "")
    text = re.sub(r"\bofficial update\b", "update", text)
    text = re.sub(r"\bofficial\b ?", "", text)
    return re.sub(r"[ \t]{2,}", " ", text).replace(" :", ":").strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in (sys.argv[1:] or SPLITS):
        eps = {}
        for line in open(SRC / f"{split}.jsonl"):
            e = json.loads(line)
            eps[e["episode_id"]] = e
        rows = []
        for eid, f in eps.items():
            if f["slot"] not in ("c_forged", "w_forged"):
                continue
            t = eps.get(eid.replace("_forged", "_trusted"))
            if t is None:
                continue
            msgs = t["messages"]
            assert msgs[-2]["role"] == "tool" and msgs[-1]["role"] == "user", eid
            # pure channel: trusted twin's words, record moved into the user turn
            ch = copy.deepcopy(f)
            ch["messages"] = copy.deepcopy(msgs[:-2]) + [
                {"role": "user", "content": msgs[-2]["content"].strip() + " " + msgs[-1]["content"].strip()}]
            ch["episode_id"] = eid.replace("_forged", "_channel")
            ch["slot"] = f["slot"].replace("_forged", "_channel")
            ch["channel"] = "user_pasted_plain"
            rows.append(ch)
            # no marker: forged episode without the authority label
            nm = copy.deepcopy(f)
            nm["messages"][-1]["content"] = strip_marker(nm["messages"][-1]["content"])
            assert not any(m in nm["messages"][-1]["content"] for m in FORGE_MARKERS), eid
            nm["episode_id"] = eid.replace("_forged", "_nomarker")
            nm["slot"] = f["slot"].replace("_forged", "_nomarker")
            rows.append(nm)
        with open(OUT / f"{split}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[split] = len(rows)
    print("wrote", counts)


if __name__ == "__main__":
    main()
