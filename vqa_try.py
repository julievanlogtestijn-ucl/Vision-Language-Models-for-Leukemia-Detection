import os

import requests
from transformers import BlipProcessor, BlipForQuestionAnswering
from datasets import load_dataset
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from datasets import load_from_disk

model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.cuda.empty_cache()
torch.manual_seed(42)

class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get image + text
        question = self.dataset[idx]['vqa_q1']
        answer = self.dataset[idx]['vqa_a1']

        #image_id = self.dataset[idx]['pid']
        #image_path = f"Data/train_fill_in_blank/{image_id}/image.png"
        #image = Image.open(image_path).convert("RGB")

        image = self.dataset[idx]['image']
        text = question
        
        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        
        """
        labels = self.processor.tokenizer.encode(
            answer, max_length= 8, return_tensors='pt'
        )
        encoding["labels"] = labels"""

        # Create padded labels
        label_enc = self.processor.tokenizer.encode(
            answer,
            padding="max_length",
            truncation=True,
            max_length=10,              # choose what you want for answer length
            return_tensors="pt",
        )
        #labels = label_enc.input_ids
        # ignore loss on pad tokens
        #labels[labels == self.processor.tokenizer.pad_token_id] = -100

        labels = label_enc["input_ids"].squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        encoding["labels"] = labels

        # remove batch dimension
        for k,v in encoding.items():  encoding[k] = v.squeeze()
        return encoding

#training_dataset = load_dataset("json", data_files="Data/train.jsonl", split="train[:90%]")
#valid_dataset = load_dataset("json", data_files="Data/train.jsonl", split="train[90%:]")

dataset = load_from_disk("train_data_classification")   
dataset = dataset.shuffle(seed=42)

#subset for debugging
dataset = dataset.select(range(100))  # Debug subset

val_size = int(0.2 * len(dataset))                     
valid_dataset = dataset.select(range(val_size))
training_dataset = dataset.select(range(val_size, len(dataset)))

print("Training sets: {} - Validating set: {}".format(len(training_dataset), len(valid_dataset)))

train_dataset = VQADataset(dataset=training_dataset,
                          processor=processor)
valid_dataset = VQADataset(dataset=valid_dataset,
                          processor=processor)

batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)

num_epochs = 3
patience = 10
min_eval_loss = float("inf")
early_stopping_hook = 0
tracking_information = []
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    for idx, batch in zip(tqdm(range(len(train_dataloader)), desc='Training batch: ...'), train_dataloader):
        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device)
        attention_masked = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        # attention_mask=attention_masked,
                        labels=labels)
            
        loss = outputs.loss
        epoch_loss += loss.item()
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    model.eval()
    eval_loss = 0
    for idx, batch in zip(tqdm(range(len(valid_dataloader)), desc='Validating batch: ...'), valid_dataloader):
        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device)
        attention_masked = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_masked,
                        labels=labels)
        
        loss = outputs.loss
        eval_loss += loss.item()

    tracking_information.append((epoch_loss/len(train_dataloader), eval_loss/len(valid_dataloader), optimizer.param_groups[0]["lr"]))
    print("Epoch: {} - Training loss: {} - Eval Loss: {} - LR: {}".format(epoch+1, epoch_loss/len(train_dataloader), eval_loss/len(valid_dataloader), optimizer.param_groups[0]["lr"]))
    scheduler.step()
    if eval_loss < min_eval_loss:
        model.save_pretrained("Model/blip-saved-model", from_pt=True) 
        print("Saved model to Model/blip-saved-model")
        min_eval_loss = eval_loss
        early_stopping_hook = 0
    else:
        early_stopping_hook += 1
        if early_stopping_hook > patience:
            break
    
pickle.dump(tracking_information, open("tracking_information.pkl", "wb"))
print("The finetuning process has done!")