import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
import os

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_TYPE = "blip1"  # "blip1" or "blip2"
finetuned = True  
model_path = "./EINDRESULTAAT_BLIP1_fullyfinetuned"  # Path to your finetuned model
#model_path = "./FINAL_BLIP1_finetuned_varieddescriptions"  # Path to your finetuned model

if finetuned:
    #processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base") if MODEL_TYPE == "blip1" else Blip2Processor.from_pretrained(model_path + "/processor")
    processor = AutoProcessor.from_pretrained(model_path)
    model_cls = Blip2ForConditionalGeneration if MODEL_TYPE == "blip2" else BlipForConditionalGeneration
    model = model_cls.from_pretrained(model_path).to(device).eval()
else:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

# --- Load model & processor (no PEFT needed) ---

# Ensure BLIP-1 token IDs are set (CLS/SEP/PAD)
tok = processor.tokenizer
model.config.pad_token_id = tok.pad_token_id
model.config.eos_token_id = getattr(tok, "sep_token_id", tok.eos_token_id)
model.config.decoder_start_token_id = getattr(tok, "cls_token_id", tok.bos_token_id)

# Load a test image (other than blood smear for catastrophic testing)
#url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"

folder = "./data/figure_dis"  # Path to your local folder with images

for fname in os.listdir(folder):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        continue  

    path = os.path.join(folder, fname)
    image = Image.open(path).convert("RGB")

    # Preprocess
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Generate caption
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            num_beams=4,
            max_new_tokens=60, min_new_tokens=3,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            length_penalty=1.0,
            early_stopping=True
        )

    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(f"{fname} → {caption}")


