"""Build Conviction 2 data (pre-registered in research/EXPERIMENTS.md, 2026-11-21). CPU only, deterministic.

    python scripts/make_c2_data.py
      -> data/conviction_bench/c2/train_aug.jsonl          (channel-contrast families, from the train split ONLY)
      -> data/conviction_bench/c2/train_aug_v2.jsonl       (+ twinpaste family; recipe change #1)
      -> data/conviction_bench/c2_diag/val.jsonl           (diagnostic: new-pointer pastes built from val)
      -> data/conviction_bench/v0.1_controls2/{split}.jsonl (held-out paste-style controls, evaluation only; val + test splits)

The frozen v0.1 splits are read, never modified.
"""
import json
from collections import Counter
from pathlib import Path

from conviction.data.channel_aug import FAMILIES_V2, build_family, build_style_controls, build_train_aug, twin_pairs
from conviction.data.splits import load_split
from conviction.data.validation import _check_episode

SRC = Path("data/conviction_bench/v0.1")
AUG_SEED = 20261121
CONTROL_SPLITS = ["val", "test_id", "test_template", "test_structural"]  # val: arm selection only


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    errs = [x for r in rows for x in _check_episode(r)]
    assert not errs, errs[:10]
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    aug = build_train_aug(load_split(SRC, "train"), AUG_SEED)
    write(Path("data/conviction_bench/c2/train_aug.jsonl"), aug)
    print("train_aug", len(aug), dict(Counter(e["c2_family"] for e in aug)))
    # recipe change #1 (2026-11-22): adds the twinpaste family; the other families are identical to train_aug.jsonl
    aug2 = build_train_aug(load_split(SRC, "train"), AUG_SEED, families=FAMILIES_V2)
    write(Path("data/conviction_bench/c2/train_aug_v2.jsonl"), aug2)
    print("train_aug_v2", len(aug2), dict(Counter(e["c2_family"] for e in aug2)))
    # diagnostic (val only): the new-pointer paste family, to check whether A1 rejects pastes with unfamiliar pointers
    import random
    rng = random.Random(AUG_SEED)
    diag = [build_family(t, f, "paste", rng) for t, f in twin_pairs(load_split(SRC, "val"))]
    write(Path("data/conviction_bench/c2_diag/val.jsonl"), diag)
    print("c2_diag val", len(diag))
    for split in CONTROL_SPLITS:
        rows = build_style_controls(load_split(SRC, split))
        write(Path(f"data/conviction_bench/v0.1_controls2/{split}.jsonl"), rows)
        print(split, len(rows))


if __name__ == "__main__":
    main()
