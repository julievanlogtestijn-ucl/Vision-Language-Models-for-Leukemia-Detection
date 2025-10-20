# train_blip1_lora_modules_to_save.py
import os, math, torch, wandb
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, List, Dict

from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from torch.amp import GradScaler, autocast

# ===================== Config =====================
MODEL_ID        = "Salesforce/blip-image-captioning-base"
DATA_DIR        = "eindresultaat_train_data"     # HF load_from_disk path
TEXT_FIELD      = "varied_descriptions"          # change if needed
OUTPUT_DIR      = "./EINDRESULTAAT_BLIP1_lora"   # PEFT adapters + modules_to_save
OUTPUT_DIR_MERGED = OUTPUT_DIR + "_merged"       # full merged HF model (optional)
EPOCHS          = 10
BATCH_SIZE      = 4
GRAD_ACCUM      = 8                               # effective batch ~32
LR              = 5e-5
WEIGHT_DECAY    = 0.05
WARMUP_RATIO    = 0.1
LABEL_SMOOTH    = 0.05
SEED            = 42
PROJECT         = "Dissertation-VLM"
RUN_NAME        = "eindresultaat_blip1_lora"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# LoRA target modules (from your module dump)
TARGET_MODULES = [
    "crossattention.self.query",
    "crossattention.self.key",
    "crossattention.self.value",
    "crossattention.output.dense",
]

# Keep these base modules trainable AND saved with the PEFT checkpoint
# Adjust K_LAST if you want more/less decoder depth updated
K_LAST = 2

MODULES_TO_SAVE_BASE = [
    "text_decoder.cls",  # LM head
    "text_decoder.bert.embeddings.word_embeddings",
    "text_decoder.bert.embeddings.position_embeddings",
    "text_decoder.bert.embeddings.LayerNorm",
]
DEC_LAYERS = [f"text_decoder.bert.encoder.layer.{i}" for i in range(12 - K_LAST, 12)]
MODULES_TO_SAVE = MODULES_TO_SAVE_BASE + DEC_LAYERS


# ===================== Repro =====================
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ===================== Dataset =====================
class ImageCaptioningDatasetOld(Dataset):
    def __init__(self, hf_dataset, processor, text_field=TEXT_FIELD):
        self.ds = hf_dataset
        self.processor = processor
        self.text_field = text_field
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"]
        txt = ex[self.text_field]
        enc = self.processor(images=img, text=txt, padding=False, truncation=True, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}

# ===================== Collator =====================
@dataclass
class BlipCaptioningCollator:
    tokenizer: Any
    label_pad_token_id: int = -100
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        pixel_values = [f["pixel_values"] for f in features]
        text_batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True, return_tensors="pt"
        )
        images = torch.stack(pixel_values, dim=0)
        labels = text_batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = self.label_pad_token_id
        return {
            "input_ids": text_batch["input_ids"],
            "attention_mask": text_batch["attention_mask"],
            "pixel_values": images,
            "labels": labels,
        }

# ===================== Label Smoothing =====================
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.0, ignore_index=-100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=-1)
    def forward(self, logits, target):
        log_probs = self.log_softmax(logits)
        ignore = target.eq(self.ignore_index)
        tgt = target.clone(); tgt[ignore] = 0
        nll = -log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        smooth = -log_probs.mean(dim=-1)
        nll = nll.masked_fill(ignore, 0.0)
        smooth = smooth.masked_fill(ignore, 0.0)
        active = (~ignore).sum()
        if active == 0: return logits.new_tensor(0.0)
        loss = (1 - self.smoothing) * nll.sum() + self.smoothing * smooth.sum()
        return loss / active

# ===================== Utilities =====================
def freeze_vision(model):
    for p in model.vision_model.parameters():
        p.requires_grad = False

def add_lora_with_modules_to_save(model):
    cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=TARGET_MODULES,
        modules_to_save=MODULES_TO_SAVE,   # keep & save LM head + last K layers
    )
    model = get_peft_model(model, cfg)
    print(f"[LoRA] Target modules: {TARGET_MODULES}")
    print(f"[PEFT] modules_to_save: {MODULES_TO_SAVE}")
    model.print_trainable_parameters()
    return model

def count_trainable(m):
    tot = sum(p.numel() for p in m.parameters())
    trn = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return trn, tot

# ===================== Main =====================
def main():
    # ----- Load data -----
    full = load_from_disk(DATA_DIR)

    # Filter out bad rows (missing text or image)
    def ok(ex):
        txt = ex.get(TEXT_FIELD)
        return (txt is not None) and (len(str(txt).strip()) > 0) and (ex.get("image") is not None)
    try:
        train_raw = full["train"].filter(ok)
        val_raw   = full["validation"].filter(ok)
    except Exception:
        split = full.train_test_split(test_size=0.1, seed=SEED)
        train_raw, val_raw = split["train"].filter(ok), split["test"].filter(ok)

    # ----- Processor / base model -----
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = BlipForConditionalGeneration.from_pretrained(MODEL_ID)

    tok = processor.tokenizer
    base.config.pad_token_id = tok.pad_token_id
    base.config.eos_token_id = getattr(tok, "sep_token_id", tok.eos_token_id)
    base.config.decoder_start_token_id = getattr(tok, "cls_token_id", tok.bos_token_id)


    # Freeze vision tower
    freeze_vision(base)

    # Wrap with PEFT (adds LoRA + keeps selected base modules trainable)
    model = add_lora_with_modules_to_save(base)
    model.to(DEVICE)

    # Datasets & loaders
    train_ds = ImageCaptioningDatasetOld(train_raw, processor)
    val_ds   = ImageCaptioningDatasetOld(val_raw,   processor)
    collator = BlipCaptioningCollator(tokenizer=processor.tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collator)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    # Optimizer / scheduler
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM)
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Loss & AMP
    criterion = LabelSmoothingCE(LABEL_SMOOTH, ignore_index=-100)
    scaler = GradScaler("cuda") if DEVICE == "cuda" else GradScaler(enabled=False)

    # W&B
    wandb.init(project=PROJECT, name=RUN_NAME, config=dict(
        model=MODEL_ID, epochs=EPOCHS, batch_size=BATCH_SIZE, grad_accum=GRAD_ACCUM,
        lr=LR, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO, label_smoothing=LABEL_SMOOTH,
        lora_targets=TARGET_MODULES, modules_to_save=MODULES_TO_SAVE, k_last=K_LAST
    ))

    trn, tot = count_trainable(model)
    print(f"Trainable params: {trn:,} / {tot:,} ({100*trn/tot:.2f}%)")
    wandb.log({"params/trainable": trn, "params/total": tot})

    # ----- Train -----
    best_val, patience, bad = float("inf"), 2, 0
    global_step = 0
    model.train()

    for epoch in range(1, EPOCHS + 1):
        running = 0.0
        optim.zero_grad(set_to_none=True)
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {epoch}")

        for step, batch in pbar:
            batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            with autocast(device_type="cuda", enabled=(DEVICE=="cuda")):
                out = model(**batch)
                loss = out.loss
                #loss = criterion(out.logits, batch["labels"])

            scaler.scale(loss).backward()
            if step % GRAD_ACCUM == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running += loss.item()
            if step % 10 == 0:
                wandb.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0], "step": global_step})
                pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = running / len(train_loader)
        print(f"Epoch {epoch} train loss: {train_loss:.4f}")
        wandb.log({"train/epoch_loss": train_loss, "epoch": epoch})

        # ----- Validation -----
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for vb in tqdm(val_loader, desc="Validation"):
                vb = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in vb.items()}
                with autocast(device_type="cuda", enabled=(DEVICE=="cuda")):
                    out = model(**vb)
                    vloss = out.loss
                    #vloss = criterion(out.logits, vb["labels"])
                val_sum += vloss.item()
        val_loss = val_sum / len(val_loader)
        print(f"Epoch {epoch} val loss: {val_loss:.4f}")
        wandb.log({"val/loss": val_loss, "epoch": epoch})

        # Early stopping + save best (PEFT adapters + modules_to_save)
        if val_loss < best_val - 1e-4:
            best_val, bad = val_loss, 0
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)               # saves adapters + modules_to_save
            processor.save_pretrained(OUTPUT_DIR)           # save processor alongside
            print(f"✅ Saved PEFT checkpoint to {OUTPUT_DIR}")
        else:
            bad += 1
            if bad >= patience:
                print("⏹️ Early stopping.")
                break
        model.train()

    wandb.finish()

    # Final save (PEFT)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"✅ Final PEFT checkpoint saved to {OUTPUT_DIR}")

    # OPTIONAL: also save a merged full model for plain HF inference
    try:
        merged = model.merge_and_unload()  # merges LoRA into base and removes PEFT wrappers
        os.makedirs(OUTPUT_DIR_MERGED, exist_ok=True)
        merged.save_pretrained(OUTPUT_DIR_MERGED)
        processor.save_pretrained(OUTPUT_DIR_MERGED)
        print(f"✅ Merged full model saved to {OUTPUT_DIR_MERGED}")
    except Exception as e:
        print(f"ℹ️ merge_and_unload not completed: {e}")

if __name__ == "__main__":
    main()
