from torch.utils.data import Dataset
import torch
from datasets import load_from_disk
from PIL import Image
import pandas as pd

class ImageCaptioningDataset_old(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        caption = item["varied_description"] #choose description/caption/desc_detailed

        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        # remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["text"] = caption  # keep for later decoding

        labels = encoding["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        encoding["labels"] = labels

        return encoding

def image_captioning_collator(batch):
    keys = batch[0].keys()
    output = {}

    for key in keys:
        if key == "text":
            # These are just raw strings, keep as list
            output[key] = [item[key] for item in batch]
        else:
            # Stack all tensor items into a single batch tensor
            output[key] = torch.stack([item[key] for item in batch])

    return output

class ImageOnlyDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {
            "pil_image": sample["image"],               # raw PIL
            "image_filename": sample.get("image_filename"),
            "true_description": sample.get("true_description"), #varied_descriptions or "true_description" or 'desc-detailed
        }

def image_only_collator(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),
        "image_filename": [b["image_filename"] for b in batch],
        "true_description": [b["true_description"] for b in batch],
    }

def pil_collator(batch):
    return {
        "images": [b["pil_image"] for b in batch],   # list of PIL
        "image_filename": [b["image_filename"] for b in batch],
        "true_description": [b["true_description"] for b in batch],
    }


from torch.utils.data import Dataset
import torch

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor, max_input_length=64, max_target_length=64):
        self.dataset = dataset
        self.processor = processor
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        caption = item["lineage_label"]  

        # Input: image + empty prompt
        inputs = self.processor(
            images=image,
            text="",  
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt",
            add_special_tokens=True
        )

        
        # 1. Tokenize the caption without special tokens
        tokenized = self.processor.tokenizer(
            caption.strip(),
            add_special_tokens=False,  # don't add [CLS], [SEP]
            return_tensors="pt"
        ).input_ids.squeeze(0)

        # 2. Append EOS token manually
        eos_token_id = self.processor.tokenizer.eos_token_id  # usually 2 for BLIP1
        tokenized = torch.cat([tokenized, torch.tensor([eos_token_id])], dim=0)

        # 3. Truncate and pad to max length
        max_len = self.max_target_length
        if tokenized.size(0) < max_len:
            pad_len = max_len - tokenized.size(0)
            padded = torch.cat([tokenized, torch.full((pad_len,), self.processor.tokenizer.pad_token_id)])
        else:
            padded = tokenized[:max_len]

        # 4. Mask padding tokens for loss
        labels = padded.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        """
        # Labels: tokenized caption only
        labels = self.processor.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt"
        ).input_ids

        labels = labels.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100  # ignore padding in loss
        #"""

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels
        }



def get_train_val_split(dataset, val_fraction=0.2, seed=42):
    dataset = dataset.shuffle(seed=seed)
    val_size = int(val_fraction * len(dataset))
    val_dataset = dataset.select(range(val_size))
    train_dataset = dataset.select(range(val_size, len(dataset)))
    return train_dataset, val_dataset

def collator_old(batch, processor=None):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


def collator_workingforblip1(batch, processor=None):
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch

def collator(batch, processor=None):
    processed_batch = {}

    for key in batch[0].keys():
        if key in ["input_ids", "attention_mask", "pixel_values"]:
            processed_batch[key] = torch.stack([example[key] for example in batch])
        elif key == "text":
            processed_batch["text"] = [example["text"] for example in batch]

    # Create labels (masked version of input_ids)
    labels = processed_batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    processed_batch["labels"] = labels

    return processed_batch

def collator_vqa(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }

class VQADataset(Dataset):
    def __init__(self, dataset, processor):
        self.samples = []
        self.processor = processor

        for example in dataset:
            for i in range(3):  # vqa_q1–3, vqa_a1–3
                q_key = f"vqa_q{i+1}"
                a_key = f"vqa_a{i+1}"
                if q_key in example and a_key in example:
                    q = example[q_key]
                    a = example[a_key]
                    if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
                        self.samples.append({
                            "image": example["image"],
                            "question": q,
                            "answer": a
                        })

    def __len__(self):
        return len(self.samples)

        """
        # pad labels to max length too - only text so just use tokenizer
        labels_enc = self.processor.tokenizer(
            answer,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        #"""

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample["image"]
        question = sample["question"]
        answer = sample["answer"]

        # Process image and question (input to encoder)
        inputs = self.processor(
            images=image,
            text=question,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        # pad labels to max length too - only text so just use tokenizer
        labels_enc = self.processor.tokenizer(
            answer,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        labels = labels_enc["input_ids"].squeeze(0)

        # Replace pad tokens with -100 so they're ignored in loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        """
        # === EOS LOGIC ===
        eos_token_id = self.processor.tokenizer.eos_token_id or 2  # fallback if missing
        max_len = 64  # total max length for labels

        # Tokenize answer (without special tokens)
        tokenized = self.processor.tokenizer(
            answer.strip(),
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        # Append EOS
        tokenized = torch.cat([tokenized, torch.tensor([eos_token_id])], dim=0)

        # Pad/truncate
        if tokenized.size(0) < max_len:
            pad_len = max_len - tokenized.size(0)
            padded = torch.cat([tokenized, torch.full((pad_len,), self.processor.tokenizer.pad_token_id)])
        else:
            padded = tokenized[:max_len]

        # Mask padding tokens
        labels = padded.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        """

        # === Final outputs ===
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels
        }

def collator_blip2(batch, processor):
    # batch comes with keys: input_ids, attention_mask, pixel_values, labels
    out = {}
    out["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])

    # keep input_ids/attention_mask produced from the *prompt sequence* we built (see next section)
    out["input_ids"]      = torch.stack([b["input_ids"] for b in batch])
    out["attention_mask"] = torch.stack([b["attention_mask"] for b in batch])

    # USE the labels that __getitem__ prepared; don't regenerate from input_ids
    labels = torch.stack([b["labels"] for b in batch])
    labels[labels == processor.tokenizer.pad_token_id] = -100
    out["labels"] = labels
    return out

class VQADatasetBlip1(Dataset):
    def __init__(self, dataset, processor):
        self.samples = self.flatten(dataset)
        self.processor = processor

    def flatten(self, dataset):
        samples = []
        for example in dataset:
            for i in range(1, 3):
                q = example.get(f"vqa_q{i}")
                a = example.get(f"vqa_a{i}")
                if pd.notna(q) and pd.notna(a):
                    samples.append({
                        "image": example["image"],
                        "question": q.strip(),
                        "answer": a.strip()
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample["image"]
        question = sample["question"]
        answer = sample["answer"]

        # Process question and image together
        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
            return_attention_mask=True
        )

        # Process answer separately for labels
        labels = self.processor.tokenizer.encode(
            answer,
            max_length=64,
            truncation=True,
            add_special_tokens=True,
        )

        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels
        }

def vqa_collate_fn(batch, processor):
    # Lists for each field
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Stack fixed-length tensors
    input_ids_stacked = torch.stack(input_ids)
    attention_mask_stacked = torch.stack(attention_mask)
    pixel_values_stacked = torch.stack(pixel_values)

    # Pad labels to the longest answer in the batch and use -100 as padding
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    return {
        "input_ids": input_ids_stacked,
        "attention_mask": attention_mask_stacked,
        "pixel_values": pixel_values_stacked,
        "labels": labels_padded,
    }

class VQADatasetBlip2(Dataset):
    def __init__(self, dataset, processor):
        self.samples = []
        self.processor = processor

        for example in dataset:
            for i in range(3):  # vqa_q1–3, vqa_a1–3
                q_key = f"vqa_q{i+1}"
                a_key = f"vqa_a{i+1}"
                if q_key in example and a_key in example:
                    q = example[q_key]
                    a = example[a_key]
                    if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
                        self.samples.append({
                            "image": example["image"],
                            "question": q,
                            "answer": a
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        prompt = f"Question: {ex['question']} Answer:"
        target = f"{prompt} {ex['answer']}"

        # Tokenize full target (will be used as inputs)
        enc = self.processor(
            images=ex["image"],
            text=target,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        input_ids  = enc["input_ids"].squeeze(0)
        attn_mask  = enc["attention_mask"].squeeze(0)
        pixel_vals = enc["pixel_values"].squeeze(0)

        # Get actual (unpadded) length of the prompt sequence using the *same tokenizer*
        tok = self.processor.tokenizer
        prompt_ids = tok(
            prompt,
            add_special_tokens=True,   # match special-tokens behavior of target
            padding=False,             # IMPORTANT: no padding to get real length
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        prompt_len = prompt_ids.size(0)

        # Build labels: copy input_ids and mask prompt + padding
        labels = input_ids.clone()
        labels[:prompt_len] = -100                                 # mask BOS + "Question: ... Answer:"
        labels[labels == tok.pad_token_id] = -100                  # ignore padding

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "pixel_values": pixel_vals,
            "labels": labels,
        }


def collator2(batch, processor=None):
    return {
        "pixel_values":   torch.stack([b["pixel_values"] for b in batch]),
        "input_ids":      torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"] for b in batch]),
    }
