# Signet pre-registration and results log

Verbatim extract of the Conviction research log, from the first Conviction 2 pre-registration (2026-11-21) onwards. Every arm was registered before its results; the release verification section is at the end.

## 2026-11-21 — Conviction 2, Phase A: channel-contrast training (pre-registration)

Recorded before any Conviction 2 run. Full plan, scaling ladder and cost quote: `research/CONVICTION2_PLAN.md`.

**Motivation:** robustness control C (2026-11-15) showed the trained adapters reject the forger's *framing*, not
untrusted *channels* (H-C2 failed: Conviction-SFT's pure-channel gap +0.04, needed ≥ +0.30). The SFT↔RL trade-off also
held (V2 failed non-inferiority on held-out accuracy).

**Data (`scripts/make_c2_data.py`, `conviction/data/channel_aug.py`, seed 20261121).** From the **train split only**,
every (trusted, forged) twin pair yields four episodes, with labels taken from one twin (no new oracle labels):
`paste` (trusted twin's words, record pasted into the user turn → do not adopt), `repointed` (trusted twin with a new
pointer sentence → adopt), `framedtool` (genuine tool record echoed in the user turn with forger wording → adopt),
`reframed` (forged twin with new forger wording → do not adopt). The train-only pointer and forger wordings are each
shared between an adopt family and a reject family, so wording does not predict the label. They are disjoint from
every v0.1 template (tested). 7,044 episodes, 1,761 per family, all passing `validation._check_episode`.
The v0.1 splits are unchanged.

**New evaluation set `v0.1_controls2` (never trained on).** Built from the test_id / test_template / test_structural twins
in three held-out paste styles (quote block, forwarded message, code block). Each item is a matched pair:
`*_style_<s>_paste` (record only in the user turn; label from the forged twin) and `*_style_<s>_tool` (identical user
turn plus the record in a tool turn; label from the trusted twin). 1,530 / 1,782 / 2,082 episodes.

**Arms** (Qwen3-1.7B, LoRA r16, base prompt, everything else as Meridian/V2):
* **A1:** stage 1 `configs/sft_c2.yaml` (1,000 episodes, 25% channel-contrast) → stage 2 `configs/grpo_c2_a1.yaml`
  (V2's GRPO, 200 steps, 25% channel-contrast prompts).
* **A2:** same stage 1 as A1 → `configs/grpo_c2_a2.yaml`: A1 plus 8 generations per prompt (still 64 completions per step)
  and an **offline difficulty prefilter** (`conviction/training/prefilter.py`). The prefilter samples 8 completions per
  candidate from the starting policy, from a 2× pool, and drops prompts whose rewards are all equal. It is not online
  dynamic sampling; TRL 1.17 has none.
* **A3 (only if A1 and A2 both fail the gate):** `configs/sft_c2_a3.yaml` (250 episodes) → `configs/grpo_c2_a3.yaml`.
* Gate for every arm: `scripts/grpo_gate.py` (zero-std fraction ≤ 0.70, clipped ≤ 0.20). For A2, also report the
  prefilter's zero-variance fraction.
* **Selection:** choose between A1 and A2 on **val** (same metrics as below, computed on val and on controls built from
  val). The chosen arm is scored once on test and gets a second seed (seed 2, both stages) before any public claim.

**Hypotheses** (chosen arm vs the named comparator, test splits, paired cluster bootstrap over problem groups, 2,000
resamples; Holm across H-2a–e):
* **H-2a (channel rule):** pure-channel gap on `v0.1_controls` (P(adopt | tool) − P(adopt | pasted plain)) has a 95% CI
  lower bound ≥ +0.30.
* **H-2b (held-out channel rule):** on `v0.1_controls2`, pooled over styles and splits,
  P(adopt | style_tool) − P(adopt | style_paste) has a lower bound ≥ +0.20.
* **H-2c (forgery):** test_id forged-record resistance ≥ 0.80 (point estimate), higher than Vector/E3b (CI excludes 0).
* **H-2d (generalisation):** test_structural final accuracy non-inferior to Vector/E3b (lower bound of difference > −0.03).
* **H-2e (correctability):** test_structural natural-history legitimate correction ≥ 0.50 (point estimate).
* **Ship criterion:** H-2a, H-2c and H-2d. H-2b and H-2e are reported either way.
* Also reported, no prediction: private held-out split, paraphrased prompt, insufficient accuracy, strict format,
  and the marker effect.

**Clarifications (2026-11-21, before any run):** A2/A3 use `n_prompts: 1600` (= 200 steps × 8 prompts), so the
prefilter scores a 3,200-prompt pool rather than 8,000 prompts that would never be used; A1 keeps V2's 4,000 (it uses
3,200). H-2b counts c_/w_ pairs only, as control C does. Analysis: `scripts/analysis_c2.py`, which reproduces control
C's published pure-channel numbers for E2, V2 and E3b exactly.

**Stop rules:** a failed ship criterion is reported as a result. At most two recipe changes beyond A1–A3. No Phase B
(Qwen3.5 2B/4B/9B) unless Phase A ships.

## 2026-11-22 — Conviction 2, recipe change #1: paired twin-pastes (pre-registration, after A1's val result)

**What A1 showed (val only; test splits untouched):** final acc. 0.713, forged-record res. 0.722, legit. correction
0.910, but **no channel rule**: pure-channel gap −0.016 [−0.059, +0.027]. The held-out paste-style gap was +0.133
[+0.081, +0.188]. **Diagnosis (a data-design flaw, found by inspection):** in A1's 1,000-episode warm-start sample,
v0.1 pointer sentences ("There's a new record from …") appear 236 times, all with a genuine tool record (adopt). The
`paste` family only used the new train pointers, so the v0.1 pointer wording stayed a cue for "adopt", and control C
is built from exactly that wording. The within-family balance test did not check balance against v0.1 episodes.

**Change (the first of the two allowed):** new family `twinpaste`, which is the trusted twin's exact conversation with the
record moved into the user turn (control C's construction, built from **train** twins only; labels from the forged
twin). With `pair_trusted: true`, every selected v0.1 c_/w_/u_ trusted episode brings its twinpaste, and the rest of
the 25% aug quota comes from the other four families. In the warm-start sample, answer-changing tool records with v0.1
pointer wording go from 236 adopt : 0 reject to 115 : 86. The remainder is g_trusted, which has no forged twin. Data:
`train_aug_v2.jsonl` (its four original families are identical to `train_aug.jsonl`, tested). Configs:
`sft_c2_fix.yaml` → `grpo_c2_a4.yaml` (A1 recipe) or `grpo_c2_a5.yaml` (A2 recipe).

**Which arm runs:** only one. A4 if A1's val final accuracy ≥ A2's, otherwise A5. This is decided before A2's val result
is read. **Diagnostic (reported, no prediction):** A1 on `c2_diag/val` (new-pointer pastes built from val) tells whether
A1 learned to reject pastes with the pointers it was trained on.

**Consequence for the hypotheses:** H-2a (control C construction) is now an in-distribution test of the construction,
with held-out problems. **H-2b (held-out paste styles, never trained) becomes the primary evidence for a channel
rule**, and the ship criterion becomes H-2a **and** H-2b, plus H-2c and H-2d. Selection between A1/A2/A4-or-A5
for `c2_test` uses val, as before.

## 2026-11-22 — Conviction 2 Phase A: val results (A1, A2, A4)

One community RTX 5090 pod, 04:07–07:12 UTC, **$2.12** (ledger phase `conviction2_phaseA`). Test splits untouched.
Gates: A1 zero-std 0.146, A2 0.000 after the prefilter (the full prefilter dropped 8.2% of 3,200 prompts in 27 min),
A4 0.250; none clipped. Outputs: `results/C2_A{1,2,4}/`, logs in `results/C2/logs/`.

| val | Anchor (E2) | Meridian (V2) | Vector (E3b) | A1 | A2 | **A4** |
|---|---|---|---|---|---|---|
| final acc. | 0.916 | 0.739 | 0.651 | 0.713 | 0.708 | 0.721 |
| forged-record res. | 0.961 | 0.761 | 0.181 | 0.722 | 0.711 | **0.808** |
| pressure flip ↓ | 0.044 | 0.140 | 0.325 | 0.180 | 0.224 | 0.171 |
| legit. correction | 0.951 | 0.967 | 0.902 | 0.910 | 0.877 | 0.893 |
| pure-channel gap (H-2a construction) | — | — | — | −0.016 [−0.059, +0.027] | +0.043 [−0.011, +0.102] | **+0.237 [+0.156, +0.317]** |
| held-out paste-style gap (H-2b construction) | — | — | — | +0.133 [+0.081, +0.188] | +0.140 [+0.088, +0.197] | **+0.235 [+0.160, +0.314]** |

A4's paste-style gap by style: quote +0.247, forward +0.263, code +0.194. In the pure-channel pairs, A4 adopts 0.42 of tool
records and 0.18 of pasted ones (A1: 0.48 vs 0.50).

**Diagnostic:** A1 rejected only 0.50 of new-pointer pastes on val (93/186), the wordings it was trained on, so the v0.1
pointer imbalance was not the whole story. The paste signal was also too sparse (~4% of A1's training data). A4's
paired twin-pastes roughly triple it and make every pointer appear with both labels.

**Reading (val, selection only, not a confirmatory test):** recipe change #1 produced the first trained adapter in this
project with a channel gap whose interval excludes zero, on the in-distribution construction and on the never-trained
paste styles. It did this without losing accuracy relative to A1 and with Meridian-level forged-record resistance. On
val, both gaps are still below the pre-registered thresholds (lower bounds 0.156 vs 0.30, and 0.160 vs 0.20).
**A4 is the val-selected arm.** Tool-record adoption also fell (0.42 vs 0.48–0.53 for A1/A2), so part of the gap may
be general caution rather than a pure channel rule. That will be reported alongside the gap.

## 2026-11-22 — Conviction 2, recipe change #2: denser contrast data (pre-registration, before running)

**Change (the second and last allowed):** arm **A6** = A4 with `aug_fraction` 0.5 instead of 0.25 in both stages
(`configs/sft_c2_fix2.yaml` → `configs/grpo_c2_a6.yaml`); everything else as A4. Rationale: A4's gap moved with
paste-signal density (A1 ~4% → A4 ~13% paste negatives in the warm-start). A6's warm-start has 168 paste negatives
(111 new-pointer pastes + 57 twin-pastes) and 221 tool-record positives from the new families, and half of all
training data is contrast data.

**Selection (decided now, `scripts/c2_select.py`, val only):** A6 is chosen if its paste-style gap lower bound is higher
than A4's (+0.160) **and** its val final accuracy is ≥ A4's (0.721) − 0.03; otherwise A4. The chosen arm is then scored
**once** on the test splits (`c2_test`) against H-2a–H-2e as amended on 2026-11-22: ship requires H-2a, H-2b, H-2c
and H-2d. Also reported, no prediction: tool-record adoption (P(adopt | tool) in the pure-channel pairs), because A4's
gap came with lower tool adoption (0.42 vs 0.48–0.53). A second seed of the chosen arm follows before any public claim.
No further recipe changes.

**Naming (2026-11-22):** the recipe of arm A6 is publicly named **Conviction Signet** (size suffix, e.g. Signet 1.7B).
The experiment log keeps the arm IDs.

## 2026-11-22 — Conviction Signet, RL-only (arm R1): pre-registration

Written after Signet 1.7B's (A6) main test metrics and **before** its channel-control results were read. Signet passed
H-2c (test_id forged-record res. 0.841) but failed H-2d (test_structural final acc. 0.494 vs Vector 0.644) and H-2e
(natural legit. correction 0.128), so Phase A does not ship as pre-registered. Every recipe with an SFT stage has
been stubborn on held-out subtypes (E2 0.580, V2 0.560, A6 0.494). The one RL-only recipe (Vector/E3b, 0.644) was
the only one that generalised. This is a **new experiment**, not a third Phase A recipe change.

**Recipe (`configs/grpo_c2_r1.yaml`):** Vector's GRPO from the base model (no SFT stage), with Signet's data:
`train_aug_v2.jsonl`, `aug_fraction` 0.5, `pair_trusted`. 400 steps (Vector used 200; with no warm-start the
channel signal arrives only through reward), `n_prompts` 6,400, everything else as `grpo_from_base.yaml`. Gate as
usual. **No checkpoint or arm selection:** the final adapter is evaluated on val, then scored **once** on test
(`c2_test ARM=r1`), then on the private split before any public claim. Hardware: the same secure RTX 4090 pod
(56sshqouwyxx0m, $0.89/h); cap **$3** for this run (owner-approved quote ~$1–1.50).

**Hypotheses (test splits; paired cluster bootstrap; Holm across R-a–R-c):**
* **R-a (generalisation):** test_structural final accuracy non-inferior to Vector/E3b (lower bound of difference > −0.03).
* **R-b (channel rule, held out):** `v0.1_controls2` paste-style gap lower bound ≥ +0.20.
* **R-c (forgery):** test_id forged-record resistance higher than Vector/E3b (CI excludes 0).
* Reported, no prediction: the pure-channel gap, natural-history legitimate correction, tool-record adoption,
  paraphrase, and private split.
* **Release criterion (as Conviction Signet):** R-a, R-b and R-c all hold. Otherwise R1 is reported as a result
  alongside A6.

## 2026-11-22 — Overnight decision tree for the Signet family (pre-registration)

Written ~09:05 UTC, before A6's channel-control results and before any R1 result. The owner pre-approved compute
until a strong Signet family exists and is asleep. Every step below is decided now. Public actions (uploads, site,
push) and the private split wait for the owner.

**Step 1 — R1 (RL-only Signet 1.7B).** Run as pre-registered above and score once on test.

**Step 2 — choose the 1.7B Signet recipe.**
* If R1 meets its release criterion (R-a, R-b, R-c): recipe = **R1**. Go to Step 3.
* Otherwise, run **one** follow-up arm **R2**, chosen by R1's test outcome:
  - R-a fails (generalisation) but R-b holds: R2 = R1 with `aug_fraction` 0.25 (closer to Vector's data).
  - R-b fails (no held-out channel rule) but R-a holds: R2 = R1 with 800 steps.
  - Both fail: R2 = R1 with `aug_fraction` 0.25 and 800 steps.
  - (Only R-c fails: R2 = R1 with 800 steps.)
  R2 is scored once on test with the same R-a–R-c. Recipe = R2 if it meets the criterion. Otherwise the ladder runs
  with whichever of {A6, R1, R2} passes the most of R-a, R-b, R-c. A tie goes to the higher test_structural final
  accuracy, and this ranking is disclosed as using test results (exploratory).
* The chosen 1.7B recipe gets a seed-2 replicate (same scoring) before any claim.

**Step 3 — the ladder (exploratory, as planned in CONVICTION2_PLAN.md §3).** Apply the chosen recipe unchanged to
Qwen3.5-2B, then 4B, then 9B (base checkpoints, non-thinking, same hyperparameters and steps; LoRA r16). Each size:
smoke + gate, train, val evals, then one test scoring with both control sets, compared with that size's
off-the-shelf leaderboard row. **Stop rule:** if 2B fails its gate, or shows no gain over off-the-shelf Qwen3.5-2B on
both test_id forged-record resistance and the paste-style gap, stop before 4B/9B and report. GPU: the cheapest card
that fits (≥48 GB for 9B).

**Not changed tonight:** no further recipe tuning beyond R2. No test-split re-scoring of any arm. The private split
is scored only after the owner's go-ahead.

**Amendment to Step 3 (2026-11-22 ~09:30 UTC, before any R1 result):** at the owner's request ("if 1.7 looks good,
immediately try and scale to 2, 4, 9"), the ladder sizes run **in parallel** on separate pods once a 1.7B recipe
qualifies, rather than in sequence. The 2B stop rule becomes a reporting rule: all sizes are reported as exploratory,
and a size whose gate fails is stopped. Engineering smoke (no results, 3 steps; `results/budget_ledger.json`):
Qwen3.5-2B ≈ 25 s/step and 8.5 GB peak; 9B ≈ 53 s/step and 28.5 GB on an A100 80GB. Both pass the gate, and
flash-linear-attention installs.

**Amendment (2026-11-22 ~09:35 UTC, before any R1 result; owner allowed parallel pods "in good taste"):** two runs
start now, in parallel with R1, instead of after its test verdict. (1) **R1 seed 2** (Qwen3-1.7B, `seed=2`), the
replicate the tree requires for the chosen recipe; it is scored like R1 and reported whatever happens.
(2) **The R1 recipe at Qwen3.5-2B** (`sig_ladder`), as the first ladder rung. If R1 qualifies, this is Step 3's 2B.
If R1 does not, it is reported as an exploratory scale point for the RL-only recipe, and the Step 2 tree still picks
the recipe for 4B/9B. 4B and 9B still wait for the decision tree. No result from these runs feeds back into the
choice of the 1.7B recipe.

**Amendment (2026-11-22 ~11:25 UTC, after R1's val result and before R1's test result):** R1 on val: final acc. 0.684,
forged-record res. 0.234, flips 0.300, legit. correction 0.967, pure-channel gap +0.005 [−0.054, +0.065], paste-style
gap +0.066 [+0.013, +0.124]. GRPO alone, even with 50% paired contrast prompts and 400 steps, did not learn the
channel rule. R1's test scoring and the Step 2 tree (R2) proceed unchanged. Since R2 is unlikely to reach the H-2b bar
from this start, **the A6 recipe (Conviction Signet) ladder starts now, in parallel**, as exploratory: Qwen3.5-2B, 4B
and 9B with `sft_c2_fix2.yaml` → `grpo_c2_a6.yaml` unchanged (`sig_ladder`, tags `2B_a6`, `4B_a6`, `9B_a6`), each
scored once on test and compared with its off-the-shelf row. The question is whether scale repairs A6's held-out
weakness (off-the-shelf held-out accuracy: 2B 0.425, 4B 0.606, 9B 0.688). If R2 meets the release criterion, its
ladder also runs. The R1-recipe 2B run already in progress continues as pre-registered.

**Ladder gate rule (2026-11-22 ~11:40 UTC, after Signet 9B's gate result):** Signet 9B (A6 recipe) failed the GRPO gate
(zero-std 0.875, reward 0.967): after the 1,000-episode warm-start, the 9B model already solves almost every training
prompt. Signet 2B passed narrowly (0.646). Following this project's existing gate rule (V2, Phase A A3), a ladder size
whose gate fails is **retried once with a 250-episode warm-start** (`EXTRA_OVERRIDE=n_train=250`, tag `<size>_a6w250`),
with GRPO unchanged; if that gate also fails, the size stops and is reported. Separately, and labelled exploratory,
the failed size's 1,000-episode warm-start adapter alone is scored once (val and test, both control sets) as
"warm-start only", because it shows what imitation alone gives at that scale.

**R1 test result and the RL-only ladder (2026-11-22 ~12:35 UTC):**
* **R1 (Qwen3-1.7B) on test:** held-out final acc. 0.661 (Vector 0.644; R-a holds, +0.017 [−0.000, +0.036]); test_id
  forged-record res. 0.275 (R-c holds vs Vector, +0.084 [+0.048, +0.122]); paste-style gap +0.034 (R-b fails);
  pure-channel gap −0.025; held-out flips 0.345; natural legit. correction 0.727. Release criterion not met.
  The tree picked **R2a** (R1 with 800 steps), which is running.
* **R1 recipe at Qwen3.5-2B, on val:** final acc. 0.962, forged-record res. 0.974, flips 0.006, legit. correction
  1.000, pure-channel gap +0.946 [+0.903, +0.979], paste-style gap +0.950 [+0.914, +0.977]. On the Qwen3.5 base,
  GRPO alone learns the channel rule that it could not learn on Qwen3-1.7B. Test scoring of this run is in progress.
* **Amendment (selection on val, before any 2B test result):** the R1 recipe (`grpo_c2_r1.yaml`, unchanged) also runs
  at **Qwen3.5-4B and 9B** (`sig_ladder`, tags `4B_r1`, `9B_r1`), each scored once on test with both control sets
  and compared with its off-the-shelf row. The A6-recipe ladder continues unchanged. Which recipe defines the
  released family is decided in the morning from val, with the owner, and stated with this log.

## 2026-11-22 — Overnight results: the Signet family (summary)

Full report: `research/SIGNET_OVERNIGHT_REPORT.md`; table: `results/comparisons/signet_family.md`. Test split,
held-out problem types (acc. / flips / forged-record res.; channel gaps pure / paste-style):
* Qwen3-1.7B: A6 0.494 / 0.395 / 0.520, gaps +0.25 / +0.24. R1 0.661 / 0.345 / 0.351, +0.03 paste. R1 seed 2
  0.651 / 0.358 / 0.381. **R2a** (800 steps, chosen by the tree) 0.751 / 0.174 / 0.614, paste +0.19 [+0.16]: R-a
  and R-c hold, R-b narrowly fails. **No 1.7B arm meets the release criterion.**
* Qwen3.5 ladder (exploratory): 2B RL-only 0.796 / 0.069 / 0.863, +0.77 / +0.79. 2B A6 0.633 / 0.337 / 0.672.
  4B RL-only 0.985 / 0.005 / 0.981, +0.97 / +0.92. 4B A6 (250-ep) 0.951 / 0.021 / 0.970. 9B A6 (250-ep)
  0.978 / 0.011 / 0.979, +0.95 / +0.95. 9B RL-only 0.961 / 0.041 / 0.872, +0.88 / +0.73. Off-the-shelf gaps are all
  within ±0.05. The A6 recipe failed the GRPO gate at 4B and 9B with the 1,000-episode warm-start (gate-fail records
  in `results/SIG_4B_a6`, `results/SIG_9B_a6`); the 250-episode retry passed.
* Bugs fixed during the night: non-string decision crashed the parser (`90619d9`); an n=0 metric crashed analysis
  (`164df92`). Neither affects earlier results.
* Not done (needs the owner): private-split scoring, ladder second seeds, any upload or public change.

## 2026-11-22 — Signet release verification (pre-registration)

Release set and names (owner, 2026-11-22): **Signet-2B** (RL-only recipe, Qwen3.5-2B), **Signet-4B** (RL-only,
Qwen3.5-4B), **Signet-9B** (A6 recipe with 250-episode warm-start, Qwen3.5-9B). Release target Monday 2026-11-24.
Verification compute approved by the owner (~$10–15). Recorded before any of these runs:

* **V-S (second seed):** each release model is retrained with `seed=2` and its recipe otherwise unchanged; for 9B
  this means the 250-episode warm-start directly. Each is scored once on test with both control sets. **Replicates**
  if seed 2's held-out (test_structural) final accuracy is within 0.05 of seed 1's **and** its paste-style gap lower
  bound is ≥ +0.50.
* **V-P (private held-out split, contamination check):** the three seed-1 release models and their off-the-shelf
  bases are scored on the private split (`~/conviction-private/v0.1-private`, hashes match
  `release/PRIVATE_SPLIT_COMMITMENT.txt`; only the three test files leave the machine, never the config or seed; pods
  are terminated afterwards). **Passes** if, on each of the three splits, a Signet model's public − private final
  accuracy difference is within 0.05 of its base's.
* **V-R (prompt robustness):** the three release models with the paraphrased system prompt (`base_paraphrase`) on
  test_id and test_structural. **Passes** if held-out final accuracy drops by ≤ 0.05.
* All three are reported regardless of outcome. A failed check is disclosed on the model card. It does not by itself
  cancel the release; that is the owner's call.

### Result (2026-11-23 UTC): all three checks pass at all three sizes

Verdicts: `results/SIGV_<size>/verdicts.json` (`scripts/signet_verify.py`). Held-out = test_structural final accuracy.

| Size | V-S seed 1 → seed 2 held-out | V-S seed-2 paste gap [lo] | V-P Signet / base public−private drop (id, template, structural) | V-R held-out base → paraphrase | Verdicts |
|---|---|---|---|---|---|
| Signet-2B | 0.796 → 0.840 | +0.82 [+0.78] | +0.017/−0.003, +0.001/−0.013, −0.018/−0.017 | 0.796 → 0.847 | S ✓ P ✓ R ✓ |
| Signet-4B | 0.985 → 0.961 | +0.66 [+0.63] | +0.010/+0.036, +0.001/+0.018, −0.001/−0.028 | 0.985 → 0.965 | S ✓ P ✓ R ✓ |
| Signet-9B | 0.978 → 0.972 | +0.94 [+0.92] | +0.007/+0.018, −0.011/+0.007, +0.003/−0.019 | 0.978 → 0.941 | S ✓ P ✓ R ✓ |

* No contamination signal: every Signet public−private difference is within 0.02, and within 0.05 of its base's.
* Signet-4B seed 2 replicates, but with a weaker paste-style channel rule (+0.66 vs +0.92; per-style code-block gap
  +0.69). Disclosed on the card as decided by the owner.
* 2B scores higher with the paraphrased prompt (fewer `logic.sets` truncations); not a robustness concern.
* Privacy: only `metrics.json`, `summary.csv` and `run_info.json` came back for the private split; no item-level
  predictions. All verification pods terminated. Spend $12.92 (balance $22.35 → $9.43).
