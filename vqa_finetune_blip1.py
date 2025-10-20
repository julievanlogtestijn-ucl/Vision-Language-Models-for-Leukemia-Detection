import torch
from transformers import BlipForQuestionAnswering, BlipProcessor
from datasets import load_from_disk
from torch.utils.data import DataLoader
from vlm_data_utils import VQADatasetBlip1, vqa_collate_fn, collator_old
import wandb
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Salesforce/blip-vqa-base"
output_dir = "./FINAL_BLIP1_finetuned_1EPOCH_vqa"

# Use BlipProcessor and VQA model
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForQuestionAnswering.from_pretrained(model_id).to(device)

print("Max length:", processor.tokenizer.model_max_length)

# === Load and prepare data ===
dataset = load_from_disk("train_data_final")

#get debug subset of training data
#dataset = dataset.shuffle(seed=42).select(range(5000))

dataset = dataset.shuffle(seed=42)
val_size = int(0.2 * len(dataset))
val_raw = dataset.select(range(val_size))
train_raw = dataset.select(range(val_size, len(dataset)))

train_dataset = VQADatasetBlip1(train_raw, processor)
val_dataset = VQADatasetBlip1(val_raw, processor)

train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True,collate_fn=lambda x: x)
val_loader = DataLoader(val_dataset,batch_size=4,collate_fn=lambda x: x)

#train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: collator_old(x, processor))
#val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: collator_old(x, processor))

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

wandb.init(
    project="Dissertation-VLM",
    name="final_blip1_vqa_finetune",
    config={"lr": 5e-5, "batch_size": 4, "model": model_id, "type": "blip1_vqa"}
)

model.train()
for epoch in range(1):
    running_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        batch = {
            k: torch.nn.utils.rnn.pad_sequence([d[k] for d in batch], batch_first=True)
            for k in batch[0].keys()
        }

        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(device),
            labels=batch["labels"].to(device) 
        )

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        wandb.log({"train/loss": loss.item(), "epoch": epoch+1})

    avg_loss = running_loss / len(train_loader)
    wandb.log({"train/avg_epoch_loss": avg_loss, "epoch": epoch+1})

    # === Validation ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: torch.nn.utils.rnn.pad_sequence([d[k] for d in batch], batch_first=True)
                for k in batch[0].keys()
            }

            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(device),
                labels=batch["labels"].to(device)
            )

            val_loss += outputs.loss.item()

    avg_val = val_loss / len(val_loader)
    wandb.log({"val/loss": avg_val, "epoch": epoch+1})
    print(f"Val Loss: {avg_val:.4f}")
    model.train()

model.save_pretrained(output_dir)
processor.save_pretrained(output_dir + "/processor")
wandb.finish()
print(f"Saved to {output_dir}")