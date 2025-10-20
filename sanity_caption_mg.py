#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ================== Config ==================
#MODEL_ID        = "google/medgemma-4b-it"      # HF id or local finetuned path
MODEL_ID        = "./EINDRESULTAAT_MEDGEMMA_lora"
FOLDER          = "./data/figure_dis" # folder of images or medical_images
CSV_OUT         = ""                           # e.g., "sanity_medgemma.csv" or "" to disable

MAX_NEW_TOKENS  = 120
NUM_BEAMS       = 3
DO_SAMPLE       = False
TEMPERATURE     = 1.0

SYSTEM_PROMPT_ = (
    "You are a medical image captioning assistant. Given a microscopy image of a blood cell, "
    "describe the cell's morphological features (size, nuclear shape, chromatin, cytoplasm) "
    "and mention the cell type and diagnosis, for diagnosis choose between the following: healthy, aml, cml, all, cll, apml. "
    "Give me a concise caption, no more than 2 sentences, do not repeat phrases and only describe each feature once."
)

SYSTEM_PROMPT = (
    "Describe this image in one or two sentences."
)
# ============================================

import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

def _list_images(folder):
    exts = (".png", ".jpg", ".jpeg", ".webp")
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(exts)
    ]

def _load_image(path):
    return Image.open(path).convert("RGB")

def main():
    # device / dtype
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # load model + processor (works for base or local finetuned)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()

    # images
    if not os.path.isdir(FOLDER):
        raise FileNotFoundError(f"Folder not found: {FOLDER}")
    image_paths = _list_images(FOLDER)
    if not image_paths:
        print(f"No images found in: {FOLDER}")
        return

    rows = []
    for path in image_paths:
        fname = os.path.basename(path)
        try:
            image = _load_image(path)

            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user",   "content": [{"type": "image", "image": image}]},
            ]

            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=dtype)

            input_len = inputs["input_ids"].shape[-1]

            gen_kwargs = dict(
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                do_sample=DO_SAMPLE,
            )
            if DO_SAMPLE:
                gen_kwargs["temperature"] = TEMPERATURE

            with torch.inference_mode():
                out_ids = model.generate(**inputs, **gen_kwargs)
                new_tokens = out_ids[0][input_len:]
                caption = processor.decode(new_tokens, skip_special_tokens=True).strip()

            print(f"{fname} \u2192 {caption}")
            rows.append({"image_filename": fname, "prediction": caption})

        except Exception as e:
            print(f"{fname} \u2192 [ERROR] {e}")

    if CSV_OUT:
        try:
            import pandas as pd
            pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
            print(f"Saved {len(rows)} predictions to {CSV_OUT}")
        except Exception as e:
            print(f"[WARN] Could not write CSV: {e}")

if __name__ == "__main__":
    main()
