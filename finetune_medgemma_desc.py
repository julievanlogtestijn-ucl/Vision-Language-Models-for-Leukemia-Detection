#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, torch
from datasets import load_from_disk
from PIL import Image
from typing import Any, Dict, List
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

def process_vision_info(messages):
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

def make_collator(processor):
    # Try to find special image token ids (varies across releases)
    boi = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.special_tokens_map.get("boi_token",""))
    eoi = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.special_tokens_map.get("eoi_token",""))
    def collate_fn(examples: List[Dict[str, Any]]):
        texts = []
        images = []
        for ex in examples:
            txt = processor.apply_chat_template(ex["messages"], add_generation_prompt=False, tokenize=False)
            texts.append(txt.strip())
            images.append(process_vision_info(ex["messages"]))
        batch = processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        if boi is not None and boi != -1: labels[labels == boi] = -100
        if eoi is not None and eoi != -1: labels[labels == eoi] = -100
        # Common consolidated <image> id seen in Gemma VLM guides; guard its presence
        if 262144 < processor.tokenizer.vocab_size:
            labels[labels == 262144] = -100
        batch["labels"] = labels
        return batch
    return collate_fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="google/medgemma-4b-it")
    ap.add_argument("--dataset_path", required=True)  # e.g., train_data_classification
    ap.add_argument("--output_dir", default="./finetuned_medgemma_caption")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--flash_attn", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_wandb", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    dtype = torch.bfloat16  # recommended for Gemma VLMs

    # Load dataset (expects columns: image, caption)
    ds = load_from_disk(args.dataset_path)

    # Build messages: [SYSTEM minimal] + [USER (image only)] -> [ASSISTANT caption]
    #subtypes = "healthy, acute myeloid leukemia, acute lymphoblastic leukemia, chronic myeloid leukemia, chronic lymphocytic leukemia, acute promyelocytic leukemia"

    SYSTEM_PROMPT = (
        "You are a medical image captioning assistant. Given a microscopy image of a blood cell, "
        "describe the cell's morphological features, including size, nuclear shape, chromatin, cytoplasm, "
        "and any diagnostic clues. Mention the cell type and diagnosis (e.g., leukemia subtype) if evident."
    )    
    
    messages_ds = []
    for ex in ds:
        img = ex["image"]
        cap = ex["varied_descriptions"]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user",   "content": [{"type": "image", "image": img}]},
            {"role": "assistant", "content": [{"type": "text", "text": cap}]},
        ]
        messages_ds.append({"messages": messages})

    if args.use_wandb:
        import wandb
        wandb.init(project="MedGemma-Finetune", name="medgemma_lora_caption",
                   config={"model": args.model_id, "epochs": args.epochs,
                           "bsz": args.per_device_train_batch_size, "lr": args.lr})

    # 4-bit quantization for QLoRA
    
    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # compute stays bf16
        bnb_4bit_quant_storage=torch.uint8,     # <- cut VRAM
    )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=qconf,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    """
    for name, p in model.named_parameters():
        if not p.dtype.is_floating_point:
            continue  # Skip non-float tensors (e.g., int, bool)
        if "vision_model" in name:
            p.requires_grad = False
        else:
            p.requires_grad = True
    """
    
    peft_cfg = LoraConfig(
    r=8,                      # smaller rank
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    # Narrower target set to cut memory; still works well:
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    target_modules="all-linear", 
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # minimal; or ["q_proj","k_proj","v_proj","o_proj"]
    task_type="CAUSAL_LM",
    modules_to_save=None,     # <- important: avoids the costly deepcopy
    )

    collator = make_collator(processor)

    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=512,      # captions are short; 512 is plenty
        packing=False,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        optim="adamw_torch_fused",
        bf16=True,
        report_to=("wandb" if args.use_wandb else "none"),
        seed=args.seed,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    model.config.use_cache = False 
    sft_cfg.remove_unused_columns = False

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=messages_ds,
        processing_class=processor,
        data_collator=collator,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model()
    processor.save_pretrained(args.output_dir)

    if args.use_wandb:
        import wandb; wandb.finish()
    print(f"✅ Saved LoRA adapter + processor to: {args.output_dir}")

if __name__ == "__main__":
    main()