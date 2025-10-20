import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoProcessor, BlipForConditionalGeneration, Blip2ForConditionalGeneration
from peft import PeftModel
from vlm_data_utils import ImageCaptioningDataset_old, image_captioning_collator, ImageOnlyDataset, image_only_collator, pil_collator
from tqdm import tqdm
import pandas as pd
import os

# === Config ===
MODEL_TYPE = "blip1"  # "blip1" or "blip2"
USE_FINETUNED = True
SAMPLE_SIZE = 5
OUTPUT_CSV = "EINDRESULTAAT_BLIP1_fullyfinetuned_captions_external.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load dataset ===
test_data = load_from_disk("eindresultaat_external_data") #Or "external_test_data" test_data_final 

test_data = test_data.shuffle(seed=42) #.select(range(SAMPLE_SIZE))
#dataset = ImageCaptioningDataset_old(test_data, processor=None)  # processor added later

dataset = ImageOnlyDataset(test_data, processor=None)
loader = DataLoader(dataset, batch_size=1, collate_fn=pil_collator)
#loader = DataLoader(dataset, batch_size=1, collate_fn=image_captioning_collator)

# === Load model and processor ===
if MODEL_TYPE == "blip2":
    model_id = "Salesforce/blip2-opt-2.7b"
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, device_map="auto", load_in_8bit=True)
else:
    model_id = "Salesforce/blip-image-captioning-base"
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)

if USE_FINETUNED:
    if MODEL_TYPE == "blip2":
        model = PeftModel.from_pretrained(model, "./finetuned_blip2_classification_caption")
    else:
        model_path = "./EINDRESULTAAT_BLIP1_fullyfinetuned"  # Path to your finetuned model ./FINAL_BLIP1_finetuned_lora or FINAL_BLIP1_finetuned_varieddescriptions
        model = BlipForConditionalGeneration.from_pretrained(model_path).to(device)

processor_path = "./finetuned_blip2_classification_caption/processor" if USE_FINETUNED and MODEL_TYPE == "blip2" else model_id
processor = AutoProcessor.from_pretrained(processor_path)
#processor.tokenizer.eos_token_id = 2

#model.config.eos_token_id = processor.tokenizer.eos_token_id

model.eval()


# Rebind processor to dataset (after loading)
dataset.processor = processor

# === Inference loop ===
results = []

for item in tqdm(dataset):
    #image = item["pixel_values"].unsqueeze(0).to(device)
    #image = item["pixel_values"].unsqueeze(0).to(device)
    image = item['pil_image']
    desc = item['true_description']

    #input_ids = item.get("input_ids", None)

    prompt_long = ("This is a microscopic image of a peripheral blood cell. Give only the cell type, nothing else")
    
    #prompt = "A microscopic image of a cell classified as a"
    #prompt = f"This is a microscopic image showing a {desc}. The image shows "

    prompt = ""
    #prompt = "Only answer with one of: neutrophil, eosinophil, basophil, monocyte, lymphocyte, blast"
    prompt_inputs = processor(text=prompt, return_tensors="pt").to(device)

    inputs = processor(image, prompt, return_tensors="pt").to(device)
    
    #text_inputs = processor(text=prompt, return_tensors="pt").to(device)

    #inputs = processor(images=item["pixel_values"], text=prompt, return_tensors="pt").to(device)
    #out = model.generate(**inputs)

    
    # Recommended: pass explicit gen params
    out = model.generate(
        **inputs,
        max_new_tokens=200,        # or 80–100 if you want longer
        num_beams=3,              # or use sampling: do_sample=True, top_p=0.9
        #early_stopping=True,
        no_repeat_ngram_size=2,   # optional, to reduce loops
        #eos_token_id=model.config.eos_token_id,
        #pad_token_id=model.config.pad_token_id,
    )
    
    """

    #second try for not mimicing the original training data
    out = model.generate(
        **inputs,
        (
                    pixel_values=pv,
                    max_new_tokens=64,
                    do_sample=True, top_p=0.9, temperature=0.8,
                    repetition_penalty=1.15,
                    num_beams=1,
                    no_repeat_ngram_size=3,
                )
    )

    """
   

    #out = model.generate(
    """
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=image,
            decoder_input_ids=prompt_inputs.input_ids,
            #input_ids=text_inputs.input_ids,
            #attention_mask=text_inputs.attention_mask,
            max_new_tokens=20,
            num_beams=3,
            #no_repeat_ngram_size=2,
            #repetition_penalty=2.0,
            early_stopping=True,
        )

    prefix_len = prompt_inputs.input_ids.shape[-1] if prompt_inputs is not None else 0
    if prefix_len > 0:
        output_ids = output_ids[:, prefix_len:]
    
    #gen_only = output_ids[:, text_inputs.input_ids.shape[-1]:]
    #caption = processor.tokenizer.decode(gen_only[0], skip_special_tokens=True).strip()
    caption = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    """

    #caption = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    caption = processor.decode(out[0], skip_special_tokens=True)

    results.append({
        "image_filename": test_data[results.__len__()]["image_filename"],
        "true_description": test_data[results.__len__()]["true_description"], #description for external test set, varied_descriptions, desc_detailed
        "caption": caption
    })

# === Save results ===
df_out = pd.DataFrame(results)
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Captions saved to {OUTPUT_CSV}")
