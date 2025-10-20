import torch
from transformers import Blip2ForConditionalGeneration, AutoProcessor, Blip2Processor
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
from torch.utils.data import DataLoader
from vlm_data_utils import VQADataset, collator, collator_blip2, VQADatasetBlip2, collator2
import wandb
from tqdm import tqdm

# === Config ===
model_id = "Salesforce/blip2-opt-2.7b"
output_dir = "./debug_finetuned_blip2_classification_vqa"
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load model + processor ===
base_model = Blip2ForConditionalGeneration.from_pretrained(
    model_id, load_in_8bit=True, device_map="auto"
)

processor = Blip2Processor.from_pretrained(model_id)
processor.tokenizer.eos_token_id = 2

# === Apply LoRA ===
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q", "v", "q_proj", "v_proj", "k_proj", "out_proj"]
)
model = get_peft_model(base_model, config)
model.print_trainable_parameters()

# === Load and split dataset ===
dataset = load_from_disk("train_data_classification")
dataset = dataset.shuffle(seed=42)

#get subset
dataset = dataset.select(range(100))  # Debug subset

val_size = int(0.2 * len(dataset))
val_raw = dataset.select(range(val_size))
train_raw = dataset.select(range(val_size, len(dataset)))

train_dataset = VQADatasetBlip2(train_raw, processor)
val_dataset = VQADatasetBlip2(val_raw, processor)

train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True,collate_fn=lambda x: collator2(x, processor))
val_loader = DataLoader(val_dataset,batch_size=4,collate_fn=lambda x: collator2(x, processor))

# === Optimizer ===
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# === wandb init ===
wandb.init(
    project="Dissertation-VLM",
    name="blip2_vqa_finetune",
    config={
        "model": model_id,
        "batch_size": 4,
        "lr": 5e-5,
        "type": "blip2_vqa_lora",
        "samples": len(train_dataset)
    }
)

# === Training Loop ===
model.train()
for epoch in range(1):
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    running_loss = 0

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device, torch.float16)
        labels = batch["labels"].to(device)

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
        wandb.log({"train/loss": loss.item(), "epoch": epoch+1})

    avg_loss = running_loss / len(train_loader)
    wandb.log({"train/avg_epoch_loss": avg_loss, "epoch": epoch+1})
    print(f"Epoch {epoch+1} avg train loss: {avg_loss:.4f}")

    # === Validation ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device, torch.float16)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    wandb.log({"val/loss": avg_val_loss, "epoch": epoch+1})
    print(f"Validation loss: {avg_val_loss:.4f}")
    model.train()

# === Save model ===
processor.save_pretrained(f"{output_dir}/processor")
model.save_pretrained(output_dir)
wandb.finish()

print(f"VQA BLIP-2 model saved to {output_dir}")
