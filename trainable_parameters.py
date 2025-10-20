#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, re, torch
from collections import defaultdict
from typing import Dict, List, Tuple

from transformers import (
    AutoModelForImageTextToText,
    BlipForConditionalGeneration,
)

from peft import LoraConfig, get_peft_model

# -----------------------------
# Defaults mirroring your scripts
# -----------------------------
# MEDGEMMA
MEDGEMMA_ID = "google/medgemma-4b-it"
MEDGEMMA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MEDGEMMA_LORA = dict(r=4, lora_alpha=8, lora_dropout=0.05)
MEDGEMMA_FREEZE_VISION = True
MEDGEMMA_ENABLE_M2S = True
MEDGEMMA_K_LAST = 1
MEDGEMMA_ALSO_SAVE_LM_HEAD = True

# BLIP (LoRA)
BLIP_ID = "Salesforce/blip-image-captioning-base"
BLIP_TARGETS = [
    "crossattention.self.query",
    "crossattention.self.key",
    "crossattention.self.value",
    "crossattention.output.dense",
]
BLIP_K_LAST = 2
BLIP_MODULES_TO_SAVE = (
    ["text_decoder.cls",
     "text_decoder.bert.embeddings.word_embeddings",
     "text_decoder.bert.embeddings.position_embeddings",
     "text_decoder.bert.embeddings.LayerNorm"]
    + [f"text_decoder.bert.encoder.layer.{i}" for i in range(12 - BLIP_K_LAST, 12)]
)
BLIP_LORA = dict(r=8, lora_alpha=16, lora_dropout=0.05)

# -----------------------------
# Helpers
# -----------------------------
def count_params(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total

def group_trainables(model, depth: int = 3) -> Dict[str, int]:
    buckets = defaultdict(int)
    for name, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        parts = name.split(".")
        key = ".".join(parts[:min(depth, len(parts))])
        buckets[key] += p.numel()
    return dict(sorted(buckets.items(), key=lambda kv: kv[1], reverse=True))

def show_sample_params(model, n: int = 30) -> List[str]:
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    return names[:n]

def print_summary(title: str, model):
    trn, tot = count_params(model)
    pct = 100.0 * trn / tot if tot else 0.0
    print(f"\n=== {title} ===")
    print(f"Trainable params: {trn:,} / {tot:,}  ({pct:.2f}%)")

    # vision freeze check
    vision_names = [n for n, _ in model.named_parameters() if ("vision_tower" in n or "vision_model" in n)]
    trn_vision = sum(p.numel() for n, p in model.named_parameters()
                     if ("vision_tower" in n or "vision_model" in n) and p.requires_grad)
    tot_vision = sum(p.numel() for n, p in model.named_parameters()
                     if ("vision_tower" in n or "vision_model" in n))
    if vision_names:
        vpct = 100.0 * trn_vision / tot_vision if tot_vision else 0.0
        print(f"Vision params trainable: {trn_vision:,} / {tot_vision:,} ({vpct:.2f}%)")

    # breakdown
    buckets = group_trainables(model, depth=3)
    print("\nTop trainable buckets (depth=3):")
    for i, (k, v) in enumerate(list(buckets.items())[:20], 1):
        print(f"{i:>2}. {k:<50} {v:,}")

    print("\nSample trainable parameter names:")
    for n in show_sample_params(model, n=30):
        print("  -", n)

def guess_last_k_layer_module_roots(model, k: int) -> List[str]:
    """
    Heuristic used in your MedGemma script: collect module names that look like ...layers.{idx}
    and pick the last K by idx.
    """
    seen = {}
    for name, _ in model.named_modules():
        m = re.search(r"(?:^|\.)([^.]*)layers\.(\d+)(?:$|\.)", name)
        if m:
            idx = int(m.group(2))
            root = name.split(f".{idx}")[0] + f".{idx}"
            seen[root] = idx
    if not seen or k <= 0:
        return []
    ordered = sorted(seen.items(), key=lambda kv: kv[1])
    return [name for name, _ in ordered[-k:]]

def ensure_existing_module_names(model, module_names: List[str]) -> List[str]:
    ok = []
    for name in module_names:
        cur = model
        good = True
        for part in name.split("."):
            if not hasattr(cur, part):
                good = False
                break
            cur = getattr(cur, part)
        if good:
            ok.append(name)
    return ok

# -----------------------------
# Builders matching your setups
# -----------------------------
def build_medgemma_lora(args):
    model = AutoModelForImageTextToText.from_pretrained(
        MEDGEMMA_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False

    # Optionally freeze vision tower (as in your script)
    if MEDGEMMA_FREEZE_VISION:
        for n, p in model.named_parameters():
            if "vision_tower" in n and p.requires_grad:
                p.requires_grad = False

    # modules_to_save guessing
    modules_to_save = None
    if MEDGEMMA_ENABLE_M2S:
        mnames = []
        if MEDGEMMA_ALSO_SAVE_LM_HEAD and hasattr(model, "lm_head"):
            mnames.append("lm_head")
        last_k = guess_last_k_layer_module_roots(model, MEDGEMMA_K_LAST)
        mnames.extend(last_k)
        mnames = ensure_existing_module_names(model, mnames)
        modules_to_save = mnames if mnames else None

    peft_cfg = LoraConfig(
        bias="none",
        target_modules=MEDGEMMA_TARGETS,
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
        **MEDGEMMA_LORA,
    )
    from peft import get_peft_model
    model = get_peft_model(model, peft_cfg)

    # Freeze any LoRA that landed in the vision tower (mirrors your script)
    for n, p in model.named_parameters():
        if "vision_tower" in n and ("lora_" in n or n.endswith(".A") or n.endswith(".B")):
            p.requires_grad = False

    # Print PEFT info
    print(f"[PEFT] target_modules={MEDGEMMA_TARGETS}")
    print(f"[PEFT] modules_to_save={modules_to_save}")
    return model

def build_blip_lora(args):
    base = BlipForConditionalGeneration.from_pretrained(BLIP_ID)

    # Freeze vision tower (as in your script)
    for p in base.vision_model.parameters():
        p.requires_grad = False

    peft_cfg = LoraConfig(
        bias="none",
        target_modules=BLIP_TARGETS,
        modules_to_save=BLIP_MODULES_TO_SAVE,
        **BLIP_LORA,
    )
    model = get_peft_model(base, peft_cfg)

    print(f"[PEFT] target_modules={BLIP_TARGETS}")
    print(f"[PEFT] modules_to_save={BLIP_MODULES_TO_SAVE}")
    return model

def build_blip_full(args):
    # Full fine-tune: everything trainable by default
    model = BlipForConditionalGeneration.from_pretrained(BLIP_ID)
    return model

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Quick inspector for trainable parameters (LoRA / full FT)."
    )
    parser.add_argument(
        "mode",
        choices=["medgemma_lora", "blip_lora", "blip_full"],
        help="Which setup to inspect."
    )
    args = parser.parse_args()

    if args.mode == "medgemma_lora":
        model = build_medgemma_lora(args)
        print_summary("MedGemma 4B — LoRA (completion-only training setup)", model)
    elif args.mode == "blip_lora":
        model = build_blip_lora(args)
        print_summary("BLIP base — LoRA (+modules_to_save, vision frozen)", model)
    elif args.mode == "blip_full":
        model = build_blip_full(args)
        print_summary("BLIP base — Full fine-tune (no freezing)", model)

if __name__ == "__main__":
    main()
