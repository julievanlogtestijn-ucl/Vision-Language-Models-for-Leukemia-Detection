import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipProcessor
)
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os
from PIL import UnidentifiedImageError
import wandb
from datasets import load_from_disk
from vlm_data_utils import ImageCaptioningDataset, get_train_val_split, collator, collator_old, ImageCaptioningDataset_old, image_captioning_collator

import torch, gc
#print(torch.cuda.memory_summary())

#torch.cuda.empty_cache()
#torch.cuda.ipc_collect() 

#### FIX LATER WITH TMP!!!
OUTPUT_DIR = "./EINDRESULTAAT_BLIP1_fullyfinetuned"
MODEL_ID = "Salesforce/blip-image-captioning-base"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load and split
full_train_dataset = load_from_disk("eindresultaat_train_data") #traindatafinal for varied, train_data_detailed

#debug dataset subset
#full_train_dataset = full_train_dataset.shuffle(seed=42).select(range(10000))

train_dataset_raw, val_dataset_raw = get_train_val_split(full_train_dataset)

processor = AutoProcessor.from_pretrained(MODEL_ID)
#processor.tokenizer.eos_token_id = 2

model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device)

train_dataset = ImageCaptioningDataset_old(train_dataset_raw, processor)
val_dataset = ImageCaptioningDataset_old(val_dataset_raw, processor)

#train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: collator_old(x, processor))
#val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: collator_old(x, processor))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=image_captioning_collator)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=image_captioning_collator)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

#sample = train_dataset[0]
#label_ids = sample["labels"]

# Decode using the tokenizer
#decoded_label = processor.tokenizer.decode(label_ids[label_ids != -100], skip_special_tokens=False)
#print("🔎 Decoded label:", decoded_label)
#print("🧮 Token IDs:", label_ids.tolist())

wandb.init(
    project="Dissertation-VLM",         # change to your project name
    name="eindresultaat_blip1_fullyfinetuned",    # optional: name of this run
    config={
        "learning_rate": 5e-5,
        "epochs": 5,
        "batch_size": 4,
        "model": MODEL_ID,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }
)

model.train()
pad_token_id = processor.tokenizer.pad_token_id

for epoch in range(5):
    print(f"Epoch {epoch+1}")
    running_loss = 0
    for step, batch in enumerate(tqdm(train_loader)):
        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        #labels = input_ids.clone()
        #labels[labels == pad_token_id] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        wandb.log({"train/loss": loss.item(), "epoch": epoch+1, "step": step})

    avg_epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} done — avg loss: {avg_epoch_loss:.4f}")
    wandb.log({"train/avg_epoch_loss": avg_epoch_loss, "epoch": epoch+1})

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_batch in tqdm(val_loader, desc="Validation"):
            input_ids = val_batch["input_ids"].to(device)
            pixel_values = val_batch["pixel_values"].to(device)
            attention_mask = val_batch["attention_mask"].to(device)
            labels = val_batch["labels"].to(device)

            #labels = input_ids.clone()
            #labels[labels == pad_token_id] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )

            val_loss += outputs.loss.item()

    val_loss /= len(val_loader)
    print(f"Validation loss after epoch {epoch+1}: {val_loss:.4f}")
    wandb.log({"val/loss": val_loss, "epoch": epoch+1})

    model.train()

wandb.finish()

model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"✅ Model saved to {OUTPUT_DIR}")
