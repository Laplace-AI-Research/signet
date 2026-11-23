"""Training-set selection. SFT and GRPO draw from *disjoint groups* of the
train split (hash buckets), so GRPO optimises on problems SFT never saw."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict

BUCKETS = {"sft": (0, 50), "grpo": (50, 100)}  # percent ranges of hash(group_id)


def bucket_of(group_id: str) -> int:
    return int(hashlib.sha1(group_id.encode()).hexdigest(), 16) % 100


def select(episodes: list[dict], bucket: str, n: int | None, seed: int,
           one_per_group: bool = False, include_clean: bool = True) -> list[dict]:
    """Select up to n episodes from a bucket, stratified by slot so every
    behaviour is represented. ``one_per_group`` (E5 ablation) drops counterfactual
    siblings: at most one episode per underlying problem."""
    lo, hi = BUCKETS[bucket]
    pool = [e for e in episodes if lo <= bucket_of(e["group_id"]) < hi
            and (include_clean or e["slot"] != "clean")]
    rng = random.Random(seed)
    if one_per_group:
        by_g = defaultdict(list)
        for e in pool:
            by_g[e["group_id"]].append(e)
        pool = [rng.choice(v) for _, v in sorted(by_g.items())]
    by_slot = defaultdict(list)
    for e in pool:
        by_slot[e["slot"]].append(e)
    for v in by_slot.values():
        rng.shuffle(v)
    if n is None or n >= len(pool):
        out = list(pool)
    else:  # round-robin over slots
        out, slots = [], sorted(by_slot)
        while len(out) < n:
            progressed = False
            for s in slots:
                if by_slot[s] and len(out) < n:
                    out.append(by_slot[s].pop()); progressed = True
            if not progressed:
                break
    rng.shuffle(out)
    return out


def select_mixed(episodes: list[dict], aug: list[dict] | None, bucket: str, n: int, seed: int,
                 aug_fraction: float = 0.0, pair_trusted: bool = False, **kw) -> list[dict]:
    """Conviction 2: ``n`` episodes, ``round(n * aug_fraction)`` of them from the channel-contrast families
    (``aug``, built from the train split by scripts/make_c2_data.py), the rest from v0.1 train. Both parts come from
    the same hash bucket, so SFT and GRPO stay on disjoint problem groups. ``aug_fraction=0`` is exactly ``select``."""
    n_aug = round(n * aug_fraction) if aug else 0
    out = select(episodes, bucket, n - n_aug, seed, **kw)
    if n_aug and pair_trusted:
        # Recipe change #1: every selected v0.1 trusted (tool) episode brings its twinpaste (identical words, record
        # pasted into the user turn), so v0.1 pointer wording is 50/50 adopt/reject. The rest of the aug quota comes
        # from the other families.
        from conviction.data.channel_aug import twin_key
        twins = {twin_key(e): e for e in aug if e.get("c2_family") == "twinpaste"}
        paired = [twins[twin_key(e)] for e in out
                  if e["slot"].endswith("_trusted") and twin_key(e) in twins]
        rest = [e for e in aug if e.get("c2_family") != "twinpaste"]
        out += paired + select(rest, bucket, max(0, n_aug - len(paired)), seed, **kw)
        random.Random(seed).shuffle(out)
    elif n_aug:
        out += select(aug, bucket, n_aug, seed, **kw)
        random.Random(seed).shuffle(out)
    return out


def load_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]
