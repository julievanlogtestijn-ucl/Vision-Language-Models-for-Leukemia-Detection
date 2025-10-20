import torch, pandas as pd
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoProcessor, BlipForConditionalGeneration
from vlm_data_utils import ImageOnlyDataset, pil_collator
from tqdm import tqdm

CKPT = "./EINDRESULTAAT_BLIP1_lora_merged"  # Path to your finetuned model ./FINAL_BLIP1_finetuned_lora or FINAL_BLIP1_finetuned_varieddescription
SAMPLE_SIZE = 10
OUTPUT_CSV = "EINDRESULTAAT_BLIP1_lora_captions_external.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load model & processor (no PEFT needed) ---
model = BlipForConditionalGeneration.from_pretrained(CKPT).to(device).eval()
processor = AutoProcessor.from_pretrained(CKPT)

# Ensure BLIP-1 token IDs are set (CLS/SEP/PAD)
tok = processor.tokenizer
model.config.pad_token_id = tok.pad_token_id
model.config.eos_token_id = getattr(tok, "sep_token_id", tok.eos_token_id)
model.config.decoder_start_token_id = getattr(tok, "cls_token_id", tok.bos_token_id)

# --- Data ---
test_data = load_from_disk("eindresultaat_external_data") #.shuffle(seed=42).select(range(SAMPLE_SIZE)) #external_set eindresultaat_external_data
dataset = ImageOnlyDataset(test_data, processor=None)
loader = DataLoader(dataset, batch_size=1, collate_fn=pil_collator)
dataset.processor = processor

results = []
with torch.no_grad():
    for i in tqdm(range(len(dataset))):
        img = dataset[i]["pil_image"]
        prompt = "The shape of the nucleus is "

        inputs = processor(images=img,return_tensors="pt")
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        out_ids = model.generate(
            **inputs,
            max_new_tokens=200,           # shorter to prevent rambling
            do_sample=False, 
            repetition_penalty=1.15,
            num_beams=4, no_repeat_ngram_size=3,
            early_stopping=True, length_penalty=1.25, min_new_tokens=15
        )

        caption = processor.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        """
        results.append({
            "image_filename": test_data[i].get("image_filename"),
            "true_description": test_data[i].get("varied_descriptions"),
            "caption": caption
        })

        """
        results.append({
            "image_filename": test_data[i].get("image_filename"),
            "source": test_data[i].get("source"),
            "true_description": test_data[i].get("true_description"),
            "caption": caption
        })

pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"✅ Captions saved to {OUTPUT_CSV}")
