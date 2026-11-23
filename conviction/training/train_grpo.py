"""E3: GRPO on top of the SFT model.

    python -m conviction.training.train_grpo --config configs/grpo.yaml [--override key=value ...]

The SFT adapter is merged into the base weights in memory, then a fresh LoRA is
trained with GRPO. With PEFT, the KL reference (adapter disabled) is therefore
the SFT policy. Rewards are the deterministic oracle reward in rewards.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from conviction.data.splits import load_split
from conviction.evaluation.runner import render_prompt
from conviction.training.model_utils import load_causal_lm, lora_config
from conviction.training.data import load_jsonl, select_mixed
from conviction.training.prefilter import filter_prompts, sample_rewards
from conviction.training.rewards import WEIGHTS, WEIGHTS_OUTCOME_ONLY, make_trl_reward
from conviction.training.train_sft import apply_overrides
from conviction.utils.logging import environment_info, memory_callback, write_json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = apply_overrides(yaml.safe_load(open(args.config)), args.override)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    from conviction.evaluation.runner import require_cuda
    require_cuda()
    set_seed(cfg["seed"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg.get("revision"))
    tok.padding_side = "left"

    g = cfg["grpo"]
    weights = WEIGHTS_OUTCOME_ONLY if cfg.get("reward") == "outcome_only" else WEIGHTS
    pf = g.get("prefilter")  # Conviction 2: {k, pool_multiplier, max_new_tokens?, batch_size?}
    n_pool = cfg["n_prompts"] * (pf["pool_multiplier"] if pf else 1)
    aug = load_jsonl(cfg["aug_file"]) if cfg.get("aug_file") else None  # Conviction 2 channel-contrast families
    eps = select_mixed(load_split(Path(cfg["data_dir"]), "train"), aug, "grpo", n_pool, cfg["seed"],
                       aug_fraction=cfg.get("aug_fraction", 0.0), pair_trusted=cfg.get("pair_trusted", False), one_per_group=cfg.get("one_per_group", False))
    rows = [{"prompt": render_prompt(tok, e["messages"], cfg["condition"]),
             "expected_final_answer": e["expected_final_answer"],
             "expected_decision": e["expected_decision"],
             "episode_id": e["episode_id"]} for e in eps]

    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    model, is_vlm = load_causal_lm(cfg["model"], revision=cfg.get("revision"), dtype=dtype)
    if cfg.get("init_adapter"):
        model = PeftModel.from_pretrained(model, cfg["init_adapter"]).merge_and_unload()
    prefilter_stats = None
    if pf:  # score the starting policy and drop zero-variance prompts before GRPO sees them
        model.to("cuda")
        t_pf = time.time()
        rewards = sample_rewards(model, tok, rows, pf["k"], pf.get("max_new_tokens", g["max_completion_length"]),
                                 g.get("temperature", 1.0), pf.get("batch_size", 32), weights)
        rows, prefilter_stats = filter_prompts(rows, rewards, cfg["n_prompts"])
        prefilter_stats["seconds"] = time.time() - t_pf
        print(f"prefilter: {prefilter_stats}")
    kept_ids = {r["episode_id"] for r in rows}
    eps = [e for e in eps if e["episode_id"] in kept_ids]
    ds = Dataset.from_list(rows)
    lc = cfg["lora"]
    peft_cfg = lora_config(lc, is_vlm)
    gargs = GRPOConfig(
        output_dir=str(out), max_steps=g["max_steps"], learning_rate=g["lr"],
        lr_scheduler_type=g.get("scheduler", "constant_with_warmup"), warmup_steps=g.get("warmup_steps", 10),
        per_device_train_batch_size=g["batch_size"], gradient_accumulation_steps=g["grad_accum"],
        num_generations=g["num_generations"], max_completion_length=g["max_completion_length"],
        temperature=g.get("temperature", 1.0), top_p=g.get("top_p", 1.0), beta=g.get("beta", 0.0),
        loss_type=g.get("loss_type", "dapo"), scale_rewards=g.get("scale_rewards", "group"),
        num_iterations=g.get("num_iterations", 1), epsilon=g.get("epsilon", 0.2),
        gradient_checkpointing=g.get("gradient_checkpointing", True), bf16=g.get("bf16", True),
        logging_steps=g.get("logging_steps", 1), save_steps=g["save_steps"], save_total_limit=3,
        log_completions=g.get("log_completions", False), report_to=[], seed=cfg["seed"],
        use_vllm=False, use_cpu=g.get("use_cpu", False),
    )
    trainer = GRPOTrainer(model=model, reward_funcs=[make_trl_reward(weights)], args=gargs, train_dataset=ds,
                          processing_class=tok, peft_config=peft_cfg, callbacks=[memory_callback()])
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"trainable params: {n_trainable:,}")
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    final = out / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    write_json(out / "train_info.json", {
        "config": cfg, "env": environment_info(), "train_seconds": time.time() - t0,
        "reward_weights": weights, "n_prompts": len(ds), "prefilter": prefilter_stats, "prompt_episode_ids": [e["episode_id"] for e in eps],
        "metrics": result.metrics, "log_history": trainer.state.log_history,
        "global_steps": trainer.state.global_step, "trainable_params": n_trainable,
        "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None,
    })
    print(f"saved adapter to {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
