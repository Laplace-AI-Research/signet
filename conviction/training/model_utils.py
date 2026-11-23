"""Model loading shared by the trainers (Conviction 2 ladder, 2026-11-22).

Qwen3.5 checkpoints are vision-language models (`Qwen3_5ForConditionalGeneration`). If AutoModelForCausalLM has no
mapping for a config, fall back to AutoModelForImageTextToText (as the evaluation runner does), and keep LoRA off the
vision tower. For text-only models (Qwen3), behaviour is unchanged.
"""
from __future__ import annotations

VISION_EXCLUDE = r".*(visual|vision_tower|vision_model|multi_modal_projector|merger).*"


def load_causal_lm(name: str, revision=None, dtype=None):
    """Return (model, is_vlm)."""
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype), False
    except ValueError as e:
        if "Unrecognized configuration class" not in str(e):
            raise
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(name, revision=revision, dtype=dtype), True


def lora_config(lc: dict, is_vlm: bool):
    from peft import LoraConfig
    kw = {"exclude_modules": VISION_EXCLUDE} if is_vlm else {}
    return LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                      target_modules=lc["target_modules"], task_type="CAUSAL_LM", **kw)
