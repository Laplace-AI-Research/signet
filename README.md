# Signet

**Signet checks the seal before it believes the record.**

Signet is a family of open language models from Laplace Research, trained to revise their answers on evidence rather
than pressure. A Signet model keeps a correct answer when pushed, accepts a valid correction, says when the evidence
is insufficient, and accepts a record only if it arrives through the authenticated tool channel. The same record
pasted into a user message is treated as a claim, not as evidence.

This repository holds the code, configurations, pre-registration log and every result behind the release.

| Model | Base | Recipe | Weights |
|---|---|---|---|
| **Signet-9B** | Qwen3.5-9B | 250-example warm-start → 200 GRPO steps | [Hugging Face](https://huggingface.co/Laplace-AI-Research/Signet-9B) (merged weights + LoRA adapters) |
| **Signet-4B** | Qwen3.5-4B | GRPO from the base, 400 steps | [Hugging Face](https://huggingface.co/Laplace-AI-Research/Signet-4B) (LoRA adapter) |
| **Signet-2B** | Qwen3.5-2B | GRPO from the base, 400 steps | [Hugging Face](https://huggingface.co/Laplace-AI-Research/Signet-2B) (LoRA adapter) |

Write-up: [laplaceresearch.org/signet](https://laplaceresearch.org/signet/) · Technical report: December 2026 ·
Benchmark: [ConvictionBench](https://convictionbench.org)

## Results

ConvictionBench v0.1 test splits, greedy decoding, non-thinking. *Held-out* means problem subtypes never seen in
training (`test_structural`). *Channel gap* is the adoption rate of a record delivered as a tool response minus the
adoption rate of the identical record pasted into a user message, on paste formats never used in training.

| Model | Held-out accuracy | Pressure flips ↓ | Forged records rejected | Valid corrections accepted | Channel gap |
|---|---|---|---|---|---|
| Qwen3.5-9B (off the shelf) | 0.688 | 0.399 | 0.158 | 0.886 | +0.01 |
| **Signet-9B** | **0.978** | 0.011 | 0.979 | **1.000** | **+0.95** |
| Qwen3.5-4B (off the shelf) | 0.606 | 0.428 | 0.128 | 0.909 | −0.02 |
| **Signet-4B** | **0.985** | **0.005** | **0.981** | 0.989 | +0.92 |
| Qwen3.5-2B (off the shelf) | 0.425 | 0.581 | 0.221 | 0.737 | −0.03 |
| **Signet-2B** | 0.796 | 0.069 | 0.863 | 0.840 | +0.79 |

None of the 35 off-the-shelf models we tested checks the channel; every Signet model does. All 15 confirmatory tests
hold after Holm correction. Three release checks, registered before they were run, pass at every size: a second
training seed, a never-published held-out split (no sign of contamination), and a paraphrased system prompt.
Signet-4B's second seed learned a weaker rule for unfamiliar paste formats; its model card says so.

## What's here

```
conviction/            benchmark generator, oracles, evaluation runner, training (SFT and GRPO)
configs/base.yaml      ConvictionBench v0.1 generator config (public seed)
configs/signet/        exact training configs for the released models
scripts/               data derivation, analysis, release checks, Holm correction, error tagging
results/               metrics, breakdowns and analyses for every run in the report (no per-item predictions)
model_cards/           the three Hugging Face model cards
docs/PREREGISTRATION.md  the research log: every arm recorded before its results
SIGNET_HASHES.txt      SHA-256 of the released adapter and merged-weight files
```

## Reproduce

```bash
uv sync --extra train --extra analysis
uv run pytest -q                                                       # unit tests

# 1. Regenerate ConvictionBench v0.1 (hashes must match data/conviction_bench/v0.1/FROZEN.txt)
uv run python -m conviction.data.cli --config configs/base.yaml
# 2. Derive the channel-contrast training data and the paste-style controls
uv run python scripts/make_c2_data.py
uv run python scripts/make_channel_controls.py val test_id test_template test_structural
# 3. Train (GPU) and evaluate, e.g. Signet-4B
uv run python -m conviction.training.train_grpo --config configs/signet/signet-4b_grpo.yaml
uv run python -m conviction.evaluation.runner --model Qwen/Qwen3.5-4B --condition base \
  --adapter checkpoints/signet-4b/grpo/final --splits test_id,test_template,test_structural \
  --max-new-tokens 200 --out results/my_signet_4b
```

Signet-9B trains in two stages: `configs/signet/signet-9b_stage1_sft.yaml`, then `signet-9b_stage2_grpo.yaml`. Pass
both adapters to the runner, comma-separated and in order. Training used one A40 (2B) or one A100 80GB (4B and 9B),
1.5 to 3 hours each.

The release checks recompute from `results/` alone, with no GPU:

```bash
uv run python scripts/signet_verify.py 9B         # second seed, private split, paraphrased prompt
uv run python scripts/signet_family_table.py      # family table
```

Per-item predictions on the public test splits are not included, to keep test items out of crawlable text. Three
things need them and so need an evaluation run first: the off-the-shelf bases' paste-style gaps in the family table
(shown as "—" without them), the Holm correction (`scripts/signet_holm.py`), and the error tagging
(`scripts/signet_spotread.py`). Their outputs are included in `results/comparisons/`.

## Scope

Synthetic, single-domain problems with a fixed JSON response. "Tool messages are authentic" holds by benchmark
construction: Signet learned to trust the channel its system prompt declares trusted; it does not authenticate
anything. See each model card's limitations.

## Licence and citation

Code and weights: Apache-2.0 (Qwen3.5 base models: Apache-2.0). ConvictionBench data: CC BY 4.0, from
[Laplace-AI-Research/ConvictionBench](https://huggingface.co/datasets/Laplace-AI-Research/ConvictionBench). Do not
train on the test splits. The benchmark and its v0.1 release live at
[Laplace-AI-Research/conviction](https://github.com/Laplace-AI-Research/conviction).

Arya Somu and Bruce Nshuti Hirwa, Laplace Research. Cite with [`CITATION.cff`](CITATION.cff).
