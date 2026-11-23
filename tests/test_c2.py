"""Conviction 2 data and training helpers (pre-registered 2026-11-21)."""
import json
from collections import Counter

from conviction.data import templates as T
from conviction.data.channel_aug import (FAMILIES, HELDOUT_STYLES, TRAIN_FRAMES, TRAIN_MARKERS, TRAIN_PASTE_FRAMES,
                                         TRAIN_POINTERS, build_style_controls, build_train_aug, twin_pairs)
from conviction.data.validation import _check_episode
from conviction.response_format import parse_response
from conviction.training.data import bucket_of, select, select_mixed
from conviction.training.prefilter import filter_prompts, has_variance


def _aug(small_data):
    return build_train_aug(small_data["train"], seed=1)


def test_aug_episodes_pass_validation(small_data):
    rows = _aug(small_data)
    assert rows
    errs = [x for r in rows for x in _check_episode(r)]
    assert errs == [], errs[:10]


def test_aug_is_deterministic(small_data):
    assert _aug(small_data) == _aug(small_data)


def test_families_balanced_and_ids_unique(small_data):
    rows = _aug(small_data)
    c = Counter(r["c2_family"] for r in rows)
    assert set(c) == set(FAMILIES) and len(set(c.values())) == 1
    ids = [r["episode_id"] for r in rows]
    assert len(ids) == len(set(ids))
    v01 = {e["episode_id"] for e in small_data["train"]}
    assert not v01 & set(ids)


def test_channel_determines_label(small_data):
    """Records in a tool turn are adopted; records only in a user turn are not."""
    for r in _aug(small_data):
        has_tool = any(m["role"] == "tool" for m in r["messages"])
        assert has_tool == (r["c2_family"] in ("repointed", "framedtool"))
        assert r["evidence_type"] == ("trusted_update" if has_tool else "forged_update")
        if not has_tool:
            assert r["problem"]["params_after"] == r["problem"]["params"]


def test_wording_is_uninformative_within_new_families(small_data):
    """Each train-only pointer is used by both adopt (repointed) and reject (paste) episodes, and each forger
    frame by both adopt (framedtool) and reject (reframed) episodes."""
    rows = _aug(small_data)
    for fam_pos, fam_neg, prefix in [("repointed", "paste", "c2_pointer."), ("framedtool", "reframed", "c2_frame.")]:
        pos = {i for r in rows if r["c2_family"] == fam_pos for i in r["template_ids"] if i.startswith(prefix)}
        neg = {i for r in rows if r["c2_family"] == fam_neg for i in r["template_ids"] if i.startswith(prefix)}
        assert pos == neg and len(pos) > 1


def test_record_text_present_and_derived_from_twin(small_data):
    by_group = {(t["group_id"], t["slot"][0]): t for t, _ in twin_pairs(small_data["train"])}
    for r in _aug(small_data):
        t = by_group[(r["group_id"], r["slot"][0])]
        record = t["messages"][-2]["content"].strip()
        assert record in r["messages"][-1]["content"] or r["c2_family"] == "repointed"
        assert r["prior_answer"] == t["prior_answer"]


def test_paste_reference_names_the_channel(small_data):
    for r in _aug(small_data):
        if r["c2_family"] == "paste" and r["expected_final_answer"] is not None:
            assert parse_response(r["reference_response"]).strict_valid
            assert json.loads(r["reference_response"])["reason"].startswith("Text pasted into a user message")


def _all_v01_strings():
    out = set(T.FORGE_MARKERS)
    for name in ("forged_update", "trusted_ref", "unverified_update"):
        out |= set(T.pool(name))
    return out


def test_train_wording_disjoint_from_v01_and_heldout_styles():
    v01 = _all_v01_strings()
    train = set(TRAIN_POINTERS) | set(TRAIN_FRAMES) | set(TRAIN_MARKERS) | set(TRAIN_PASTE_FRAMES)
    assert not v01 & train
    for m in TRAIN_MARKERS:
        assert all(m not in s for s in v01)
    # held-out paste styles never appear in training wording
    for tpl in train:
        assert ">" not in tpl.replace("<<", "").replace(">>", "") and "```" not in tpl and "Forwarded" not in tpl


def test_style_controls_are_matched_pairs(small_data):
    rows = build_style_controls(small_data["test_id"])
    assert rows and all(not _check_episode(r) for r in rows)
    by_id = {r["episode_id"]: r for r in rows}
    n_pairs = 0
    for eid, neg in by_id.items():
        if not eid.endswith("_paste"):
            continue
        pos = by_id[eid[: -len("_paste")] + "_tool"]
        assert neg["messages"][-1] == pos["messages"][-1]  # identical user turn
        assert [m for m in pos["messages"] if m["role"] != "tool"] == neg["messages"]  # only the tool turn differs
        assert neg["expected_final_answer"] == neg["ground_truth_before"]
        n_pairs += 1
    assert n_pairs == len(twin_pairs(small_data["test_id"])) * len(HELDOUT_STYLES)


def test_select_mixed(small_data):
    train, aug = small_data["train"], _aug(small_data)
    assert select_mixed(train, aug, "sft", 200, 1, aug_fraction=0.0) == select(train, "sft", 200, 1)
    mixed = select_mixed(train, aug, "grpo", 200, 1, aug_fraction=0.25)
    assert len(mixed) == 200
    assert sum("c2_family" in e for e in mixed) == 50
    assert all(50 <= bucket_of(e["group_id"]) < 100 for e in mixed)  # stays in the GRPO bucket


def test_prefilter_logic():
    assert not has_variance([1.0, 1.0, 1.0]) and has_variance([1.0, 0.3]) and not has_variance([0.5])
    rows = [{"episode_id": str(i)} for i in range(5)]
    rewards = [[1, 1], [0, 1], [0, 0], [0.3, 1], [0.05, 0.75]]
    kept, stats = filter_prompts(rows, rewards, n_keep=2)
    assert [r["episode_id"] for r in kept] == ["1", "3"]
    assert stats["n_with_variance"] == 3 and stats["zero_variance_fraction"] == 0.4 and not stats["underfilled"]


def test_style_gap_and_pure_channel_gap(small_data):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("analysis_c2", Path(__file__).parent.parent / "scripts/analysis_c2.py")
    a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
    ctrl = build_style_controls(small_data["test_id"])
    rec = lambda e, ans: {**e, "pred_answer": ans}  # noqa: E731
    # a perfect channel-checker: adopts tool records, never pasted ones
    perfect = {e["episode_id"]: rec(e, e["expected_final_answer"]) for e in ctrl}
    g = a.style_gap(perfect)
    assert g["adopt_neg"] == 0.0 and g["adopt_pos"] == 1.0 and g["gap"] == 1.0
    # a wording-reader that answers both members of each pair identically has gap 0
    same = {e["episode_id"]: rec(e, (perfect.get(e["episode_id"][: -len("_paste")] + "_tool") or e)["expected_final_answer"])
            for e in ctrl}
    assert a.style_gap(same)["gap"] == 0.0
    assert set(a.style_gap(perfect, "quote")) == set(g)


# ---- recipe change #1 (2026-11-22): twinpaste + paired selection ----
from conviction.data.channel_aug import FAMILIES_V2, twin_key  # noqa: E402


def test_v2_keeps_v1_families_identical(small_data):
    v1 = build_train_aug(small_data["train"], seed=1)
    v2 = build_train_aug(small_data["train"], seed=1, families=FAMILIES_V2)
    assert [e for e in v2 if e["c2_family"] != "twinpaste"] == v1
    assert all(not _check_episode(e) for e in v2)


def test_twinpaste_differs_from_trusted_twin_only_in_channel(small_data):
    v2 = build_train_aug(small_data["train"], seed=1, families=FAMILIES_V2)
    tp = {twin_key(e): e for e in v2 if e["c2_family"] == "twinpaste"}
    for t, f in twin_pairs(small_data["train"]):
        e = tp[twin_key(t)]
        assert e["messages"][:-1] == t["messages"][:-2]  # same history, tool turn removed
        assert e["messages"][-1]["content"] == t["messages"][-2]["content"].strip() + " " + t["messages"][-1]["content"].strip()
        assert e["expected_final_answer"] == f["expected_final_answer"]  # labels from the forged twin
        assert not any(m["role"] == "tool" for m in e["messages"])


def test_pair_trusted_pairs_every_twinned_trusted_episode(small_data):
    train = small_data["train"]
    aug = build_train_aug(train, seed=1, families=FAMILIES_V2)
    has_twin = {twin_key(t) for t, _ in twin_pairs(train)}
    eps = select_mixed(train, aug, "sft", 400, 1, aug_fraction=0.25, pair_trusted=True)
    assert len(eps) == 400 and len({e["episode_id"] for e in eps}) == 400
    keys_tp = {twin_key(e) for e in eps if e.get("c2_family") == "twinpaste"}
    trusted = [e for e in eps if "c2_family" not in e and e["slot"].endswith("_trusted") and twin_key(e) in has_twin]
    assert trusted and all(twin_key(e) in keys_tp for e in trusted)
    # v0.1 pointer wording now appears with both labels
    tp_ptr = {i for e in eps if e.get("c2_family") == "twinpaste" for i in e["template_ids"] if i.startswith("trusted_ref.")}
    tr_ptr = {i for e in trusted for i in e["template_ids"] if i.startswith("trusted_ref.")}
    assert tp_ptr == tr_ptr
    # without pairing, nothing changes
    assert select_mixed(train, aug, "sft", 400, 1, aug_fraction=0.0, pair_trusted=True) == select(train, "sft", 400, 1)


def test_c2_select_rule():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("c2_select", Path(__file__).parent.parent / "scripts/c2_select.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    a4 = {"final_accuracy": 0.72, "H-2b_style": {"lo": 0.16}}
    assert m.choose(a4, {"final_accuracy": 0.70, "H-2b_style": {"lo": 0.20}}) == "a6"
    assert m.choose(a4, {"final_accuracy": 0.68, "H-2b_style": {"lo": 0.30}}) == "a4"  # accuracy drop > 0.03
    assert m.choose(a4, {"final_accuracy": 0.80, "H-2b_style": {"lo": 0.16}}) == "a4"  # channel not better


def test_r2_branch_rule():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("br", Path(__file__).parent.parent / "scripts/c2_r2_branch.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    mk = lambda a, b, c: {"H-2d_structural_acc_vs_vector": {"lo": a}, "H-2b_style": {"lo": b}, "H-2c_vs_vector": {"lo": c}}  # noqa: E731
    assert m.branch(mk(0.0, 0.25, 0.1)) == "none"
    assert m.branch(mk(-0.1, 0.25, 0.1)) == "r2c"
    assert m.branch(mk(0.0, 0.05, 0.1)) == "r2a"
    assert m.branch(mk(-0.1, 0.05, 0.1)) == "r2b"
    assert m.branch(mk(0.0, 0.25, -0.1)) == "r2a"
