#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MedGemma 4B finetuning with completion-only loss (assistant tokens only),
chat template preserved, image inputs supported, LoRA / (optional) QLoRA,
optional modules_to_save to mimic BLIP's "last K decoder blocks + lm_head".
"""

import os, re, math, random, torch
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from datasets import load_from_disk, Dataset, DatasetDict
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# ===================== CONFIG =====================

MODEL_ID              = "google/medgemma-4b-it"
DATA_DIR              = "eindresultaat_train_data"      # HF load_from_disk path
TEXT_FIELD            = "varied_descriptions"   # caption field name
SYSTEM_PROMPT         = (
    "You are a medical image captioning assistant. Given a microscopy image of a blood cell, "
    "describe the cell's morphological features (size, nuclear shape, chromatin, cytoplasm, etc.) "
    "and mention the cell type and whether the cell type is healthy or leukemia. "
    "if the cell is leukemia, mention the leukemia subtype. Keep the description concise yet informative."
)

OUTPUT_DIR            = "./EINDRESULTAAT_MEDGEMMA_loraONLY"  # PEFT adapters + modules_to_save
PROJECT               = "MedGemma-Finetune"
RUN_NAME              = "eindresultaat_medgemma_lora"

# Training
EPOCHS                = 3
PER_DEVICE_TRAIN_BSZ  = 1
GRAD_ACCUM            = 8
LR                    = 5e-5
WEIGHT_DECAY          = 0.05
WARMUP_RATIO          = 0.1
MAX_SEQ_LEN           = 384
SEED                  = 42

# LoRA / QLoRA
USE_4BIT              = False      # set False for pure bf16 LoRA if VRAM allows
LORA_R                = 4
LORA_ALPHA            = 8
LORA_DROPOUT          = 0.05
TARGET_MODULES        = ["q_proj", "k_proj", "v_proj", "o_proj"]  # narrow & stable, this would also hit vision tower..

# Mimic BLIP capacity bump: also train a thin slice of base model
ENABLE_MODULES_TO_SAVE = False #True     # set False to pure LoRA adapters only
K_LAST_DECODER_LAYERS  = 0 #1        # number of last decoder blocks to save/train
ALSO_SAVE_LM_HEAD      = False #True

#K_LAST = 2
#LAST_TXT_LAYERS = [f"model.language_model.layers.{i}" for i in range(34-K_LAST, 34)]
#MODULES_TO_SAVE = ["lm_head"] + LAST_TXT_LAYERS
#print("modules_to_save ->", MODULES_TO_SAVE)

# Vision freezing (mimic BLIP freezing the vision tower)
FREEZE_VISION          = True

# Engineering knobs
ATTN_IMPL              = "flash_attention_2"     # falls back to "eager" if not available
LOGGING_STEPS          = 10
SAVE_TOTAL_LIMIT       = 2
USE_WANDB              = True    # toggle if you want W&B off


# ==================================================
#                       Utils
# ==================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def process_vision_info(messages: List[Dict[str, Any]]) -> List[Image.Image]:
    imgs = []
    for m in messages:
        for e in m.get("content", []):
            if isinstance(e, dict) and (e.get("type") == "image" or "image" in e):
                img = e.get("image", None)
                if isinstance(img, Image.Image):
                    imgs.append(img.convert("RGB"))
                else:
                    imgs.append(Image.open(img).convert("RGB"))
    return imgs

def build_messages_from_row(img: Image.Image, caption: str) -> List[Dict[str, Any]]:
    """system + user(image) + assistant(text) transcript (teacher-forcing)."""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": [{"type": "image", "image": img}]},
        {"role": "assistant", "content": [{"type": "text", "text": caption}]},
    ]

def freeze_vision_params(model):
    frozen = 0
    for n, p in model.named_parameters():
        if "vision_tower" in n:
            if p.requires_grad:
                p.requires_grad = False
                frozen += 1
    print(f"[Freeze] Vision params frozen: {frozen}")

def find_sublist(hay: List[int], needle: List[int]) -> Optional[int]:
    """Return start index of last occurrence of needle in hay, or None."""
    if not needle or len(needle) > len(hay):
        return None
    # Search from the end to prefer the gold answer at the tail
    for i in range(len(hay) - len(needle), -1, -1):
        if hay[i:i+len(needle)] == needle:
            return i
    return None

def tokenize_answer_only(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False).input_ids

def guess_last_k_layer_module_roots(model, k: int) -> List[str]:
    """
    Heuristic: collect module names that look like ...layers.{idx}
    and pick the last K by idx. Works for Gemma/Llama-style decoders.
    """
    import re
    seen = {}
    for name, _ in model.named_modules():
        m = re.search(r"(?:^|\.)([^.]*)layers\.(\d+)(?:$|\.)", name)
        if m:
            idx = int(m.group(2))
            # get root up to the index (e.g., "language_model.model.layers.31")
            root = name.split(f".{idx}")[0] + f".{idx}"
            seen[root] = idx
    if not seen:
        return []
    ordered = sorted(seen.items(), key=lambda kv: kv[1])
    last = [name for name, _ in ordered[-k:]]
    print(f"[modules_to_save] Guessed last {k} decoder blocks:", last)
    return last

def ensure_existing_module_names(model, module_names: List[str]) -> List[str]:
    """Keep only names that resolve to an actual submodule."""
    ok = []
    for name in module_names:
        # Resolve nested submodule by dotted path
        cur = model
        good = True
        for part in name.split("."):
            if not hasattr(cur, part):
                good = False
                break
            cur = getattr(cur, part)
        if good:
            ok.append(name)
    missing = set(module_names) - set(ok)
    if missing:
        print(f"[modules_to_save] Skipped non-existing: {sorted(missing)}")
    return ok


# ==================================================
#             Completion-only Collator
# ==================================================

class MedGemmaCompletionOnlyCollator:
    """
    Collator that renders full transcripts (system+user+assistant),
    then masks labels so loss is applied ONLY to the assistant reply.
    - Keeps image inputs intact.
    - Robust to chat template variations: finds the assistant gold reply
      by tokenizing the assistant text and locating it inside input_ids.
    - Also masks PAD and common image special tokens.
    """

    def __init__(self, processor):
        self.processor = processor
        self.tok = processor.tokenizer
        self.pad_id = self.tok.pad_token_id

        # try to find image special ids (varies by release)
        self.boi = self.tok.convert_tokens_to_ids(self.tok.special_tokens_map.get("boi_token",""))
        self.eoi = self.tok.convert_tokens_to_ids(self.tok.special_tokens_map.get("eoi_token",""))

        # Common consolidated <image> id sometimes used
        self.image_fallback_id = 262144 if getattr(self.tok, "vocab_size", 0) and 262144 < self.tok.vocab_size else None

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts: List[str] = []
        images: List[List[Image.Image]] = []
        answers: List[str] = []

        # Render full transcripts (assistant included)
        for ex in examples:
            msgs = ex["messages"]
            txt = self.processor.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
            texts.append(txt.strip())
            images.append(process_vision_info(msgs))

            # gather assistant gold answer text (concat all 'text' parts)
            ans_parts = []
            for m in msgs:
                if m.get("role") == "assistant":
                    for c in m.get("content", []):
                        if isinstance(c, dict) and c.get("type") == "text":
                            ans_parts.append(c.get("text", ""))
            answers.append("".join(ans_parts).strip())

        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        labels[:] = -100  # mask everything by default

        # Build per-example answer spans
        for i in range(input_ids.shape[0]):
            ans_ids = tokenize_answer_only(self.tok, answers[i])
            ids = input_ids[i].tolist()

            start = find_sublist(ids, ans_ids) if ans_ids else None

            if start is not None:
                end = start + len(ans_ids)
                # Unmask ONLY the assistant answer token ids
                labels[i, start:end] = input_ids[i, start:end]
            else:
                # Fallback: if we cannot match (e.g., truncation), unmask from last token that's not PAD
                # This at least applies some signal (usually the tail is the assistant)
                last_nonpad = (input_ids[i] != self.pad_id).nonzero().flatten()
                if len(last_nonpad) > 0:
                    first = int(last_nonpad[0].item())
                    labels[i, first:] = -100  # stay conservative
                # (No-op effectively; better to increase MAX_SEQ_LEN to avoid truncation)

        # Always keep PAD and image specials masked
        specials = [self.pad_id, self.boi, self.eoi, self.image_fallback_id]
        for sid in specials:
            if sid is not None and sid != -1:
                labels[input_ids == sid] = -100

        batch["labels"] = labels
        return batch


# ==================================================
#                  Data Preparation
# ==================================================

def load_and_prepare(DATA_DIR: str) -> Tuple[Dataset, Optional[Dataset]]:
    full = load_from_disk(DATA_DIR)

    def ok(ex):
        return (ex.get("image") is not None) and (ex.get(TEXT_FIELD) is not None) and (len(str(ex[TEXT_FIELD]).strip()) > 0)

    if isinstance(full, DatasetDict):
        # Use provided splits if present
        if "train" in full:
            train_raw = full["train"].filter(ok)
            val_raw = full.get("validation", None)
            if val_raw is not None:
                val_raw = val_raw.filter(ok)
            else:
                # If only test exists, treat as val
                if "test" in full:
                    val_raw = full["test"].filter(ok)
        else:
            # Single unnamed dataset in dict
            first_key = list(full.keys())[0]
            ds = full[first_key].filter(ok)
            split = ds.train_test_split(test_size=0.1, seed=SEED)
            train_raw, val_raw = split["train"], split["test"]
    else:
        ds = full.filter(ok)
        split = ds.train_test_split(test_size=0.1, seed=SEED)
        train_raw, val_raw = split["train"], split["test"]

    return train_raw, val_raw

def rows_to_messages(ds: Dataset) -> List[Dict[str, Any]]:
    out = []
    for ex in ds:
        msgs = build_messages_from_row(ex["image"], ex[TEXT_FIELD])
        out.append({"messages": msgs})
    return out


# ==================================================
#                      Main
# ==================================================

def main():
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")

    # ----- Data -----
    train_raw, val_raw = load_and_prepare(DATA_DIR)
    train_msgs = rows_to_messages(train_raw)
    eval_msgs  = rows_to_messages(val_raw) if val_raw is not None else None

    # ----- Model & Processor -----
    dtype = torch.bfloat16

    qconf = None
    if USE_4BIT:
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.uint8,
        )

    attn_impl = ATTN_IMPL
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
            quantization_config=qconf,
            attn_implementation=attn_impl,
        )
    except Exception as e:
        print(f"[Warn] Falling back to eager attention due to: {e}")
        attn_impl = "eager"
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
            quantization_config=qconf,
            attn_implementation=attn_impl,
        )

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.config.use_cache = False

    # Optionally freeze vision tower
    if FREEZE_VISION:
        freeze_vision_params(model)

    # ----- LoRA / modules_to_save -----
    modules_to_save = None
    if ENABLE_MODULES_TO_SAVE:
        mnames = []
        if ALSO_SAVE_LM_HEAD and hasattr(model, "lm_head"):
            mnames.append("lm_head")

        last_k = guess_last_k_layer_module_roots(model, K_LAST_DECODER_LAYERS)
        mnames.extend(last_k)
        mnames = ensure_existing_module_names(model, mnames)
        modules_to_save = mnames if mnames else None
        print(f"[modules_to_save] Final list: {modules_to_save}")

    peft_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save, 
    )

    from peft import get_peft_model
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # Freeze any LoRA adapters that got injected into vision tower
    for n, p in model.named_parameters():
        if "vision_tower" in n and ("lora_" in n or "A" in n or "B" in n):
            p.requires_grad = False

    # ----- Trainer / Collator -----
    collator = MedGemmaCompletionOnlyCollator(processor)

    EVAL_STRATEGY = "no" #"epoch" if eval_msgs is not None else "no"
    SAVE_STRATEGY = "epoch" #if eval_msgs is not None else "no"
    LOAD_BEST     = False #(eval_msgs is not None)

    sft_cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BSZ,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_seq_length=MAX_SEQ_LEN,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        eval_strategy=EVAL_STRATEGY,  
        load_best_model_at_end=LOAD_BEST and EVAL_STRATEGY != "no",
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to=("wandb" if USE_WANDB else "none"),
        optim="adamw_bnb_8bit",
        remove_unused_columns=False,           # IMPORTANT for image columns
        dataset_text_field="",                 # we're using a custom collator
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_msgs,
        eval_dataset=eval_msgs,
        processing_class=processor,
        data_collator=collator,
        peft_config=peft_cfg,
    )

    # ----- Train -----
    if USE_WANDB:
        import wandb
        wandb.init(project=PROJECT, name=RUN_NAME, config=dict(
            model=MODEL_ID, epochs=EPOCHS, per_device_bsz=PER_DEVICE_TRAIN_BSZ,
            grad_accum=GRAD_ACCUM, lr=LR, wd=WEIGHT_DECAY, warmup=WARMUP_RATIO,
            max_seq_len=MAX_SEQ_LEN, target_modules=TARGET_MODULES,
            modules_to_save=modules_to_save, qlora=USE_4BIT, attn=attn_impl
        ))

    trainer.train()

    # ----- Save -----
    trainer.save_model()             # saves LoRA adapters (+ modules_to_save if any)
    processor.save_pretrained(OUTPUT_DIR)

    if USE_WANDB:
        import wandb; wandb.finish()

    print(f"✅ Saved to: {OUTPUT_DIR}")
    if modules_to_save:
        print("✅ Saved LoRA adapters + selected base modules (modules_to_save).")
    else:
        print("✅ Saved LoRA adapters (no modules_to_save).")


# ==================================================
#                     Entry
# ==================================================

if __name__ == "__main__":
    main()
