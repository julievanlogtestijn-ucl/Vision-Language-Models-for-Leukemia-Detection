import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import wandb

from vlm_data_utils import ImageCaptioningDataset, get_train_val_split, collator, collator_old, ImageCaptioningDataset_old, image_captioning_collator

model_id = "Salesforce/blip2-opt-2.7b"
output_dir = "./finetuned_blip2_classification_caption"

device = "cuda" if torch.cuda.is_available() else "cpu"

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q", "v", "q_proj", "v_proj"]
)

# === Load model + processor ===
model = Blip2ForConditionalGeneration.from_pretrained(model_id, load_in_8bit=True, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)
#processor.tokenizer.eos_token_id = 2

# === Load dataset and split ===
full_train_dataset = load_from_disk("train_data_classification")

# Debug dataset subset
#full_train_dataset = full_train_dataset.shuffle(seed=42).select(range(100))

train_dataset_raw, val_dataset_raw = get_train_val_split(full_train_dataset)

train_dataset = ImageCaptioningDataset_old(train_dataset_raw, processor)
val_dataset = ImageCaptioningDataset_old(val_dataset_raw, processor)

#train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: collator_old(x, processor))
#val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: collator_old(x, processor))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=image_captioning_collator)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=image_captioning_collator)

# === Apply LoRA ===
model = get_peft_model(model, config)
model.print_trainable_parameters()

# === Optimizer ===
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

#sample = train_dataset[0]
#label_ids = sample["labels"]

#decoded_label = processor.tokenizer.decode(label_ids[label_ids != -100], skip_special_tokens=False)
#print("🔎 Decoded label:", decoded_label)
#print("🧮 Token IDs:", label_ids.tolist())

# === wandb Init ===
wandb.init(
    project="Dissertation-VLM",
    name="blip2_lora_finetune",
    config={
        "model": model_id,
        "epochs": 5,
        "batch_size": 4,
        "lr": 5e-5,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "lora_r": config.r,
        "lora_alpha": config.lora_alpha,
    }
)

# === Training ===
model.train()
for epoch in range(5):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    total_loss = 0

    for idx, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device, torch.float16)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        #labels = input_ids.clone()
        #labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            pixel_values=pixel_values, 
            labels=labels)

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        wandb.log({"train/loss": loss.item(), "epoch": epoch+1, "step": idx})
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    wandb.log({"train/avg_epoch_loss": avg_train_loss, "epoch": epoch+1})
    print(f"Epoch {epoch+1} — Avg train loss: {avg_train_loss:.4f}")

    # === Validation ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            input_ids = val_batch["input_ids"].to(device)
            pixel_values = val_batch["pixel_values"].to(device, torch.float16)
            attention_mask = val_batch["attention_mask"].to(device)
            labels = val_batch["labels"].to(device)

            # you already do this in collate
            #labels = input_ids.clone()
            #labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                pixel_values=pixel_values, 
                labels=labels)

            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    wandb.log({"val/loss": avg_val_loss, "epoch": epoch+1})
    print(f"Validation loss: {avg_val_loss:.4f}")

    model.train()  # switch back to training

# === Save Model ===
processor.save_pretrained(f"{output_dir}/processor")
model.save_pretrained(output_dir)
wandb.finish()

print(f"Model saved to {output_dir}")