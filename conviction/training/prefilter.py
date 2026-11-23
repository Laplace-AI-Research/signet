"""Conviction 2: offline difficulty prefilter for GRPO (pre-registered in research/EXPERIMENTS.md, 2026-11-21).

TRL 1.17 has no online dynamic sampling (DAPO-style resampling of zero-variance groups). This is the offline version.
Before training, sample ``k`` completions per candidate prompt from the *starting* policy (base, or the merged
warm-start), score them with the training reward, and drop prompts whose ``k`` rewards are all equal. Those prompts
have zero group advantage and contribute no gradient. The policy moves during training, so this removes prompts
that are already saturated (or hopeless) at the start. It does not track saturation as training proceeds; report
it under that name.
"""

from __future__ import annotations


def has_variance(rewards: list[float], tol: float = 1e-6) -> bool:
    return len(rewards) > 1 and (max(rewards) - min(rewards)) > tol


def filter_prompts(rows: list[dict], rewards: list[list[float]], n_keep: int) -> tuple[list[dict], dict]:
    """Keep the first ``n_keep`` rows (in the given, already shuffled order) whose rewards vary."""
    assert len(rows) == len(rewards)
    keep = [r for r, rw in zip(rows, rewards) if has_variance(rw)]
    n_var = len(keep)
    keep = keep[:n_keep]
    stats = {"n_candidates": len(rows), "n_with_variance": n_var, "n_kept": len(keep),
             "zero_variance_fraction": 1 - n_var / len(rows) if rows else 0.0,
             "underfilled": len(keep) < n_keep}
    return keep, stats


def sample_rewards(model, tok, rows: list[dict], k: int, max_new_tokens: int, temperature: float,
                   batch_size: int, weights: dict) -> list[list[float]]:
    """k sampled completions per row from ``model`` (HF generate, left padding), scored by the GRPO reward."""
    import torch

    from conviction.training.rewards import score_text

    model.eval()
    out: list[list[float]] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        enc = tok([r["prompt"] for r in batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, temperature=temperature, top_p=1.0,
                                 num_return_sequences=k, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, r in enumerate(batch):
            out.append([score_text(t, r["expected_final_answer"], r["expected_decision"], weights).total
                        for t in texts[j * k:(j + 1) * k]])
    model.train()
    return out
