import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BlipForConditionalGeneration, BlipForQuestionAnswering, BlipProcessor, Blip2Processor
from datasets import load_from_disk
from tqdm import tqdm
import pandas as pd
import argparse

# === Config ===
MODEL_TYPE = "blip1"         # "blip1" or "blip2"
USE_FINETUNED = True
SAMPLE_SIZE = 10
DATASET_PATH = "test_data_final"
OUTPUT_CSV = "FINAL_BLIP1_finetuned_1EPOCH_vqa.csv"


QUESTIONS = [
    "How would you interpret this cell morphologically?",
    "How would you describe the chromatin?",
    "Describe the cytoplasm."
]

"""
---question options:

What type of cell is shown in this image?

Does this cell indicate a healthy or diseased state?

What leukemia subtype is associated with this image?

How would you classify this cell morphologically?

Is this a blast cell?

---Morphological Features
What is the shape of the nucleus?

Describe the chromatin pattern of this cell.

What can you say about the cytoplasmic texture?

Is the nucleolus visible in this cell?

What is the level of basophilia in this image?

Are there granules or vacuoles visible in the cytoplasm?

What is the size of the cell?

"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Resolve model paths ===
if MODEL_TYPE == "blip2":
    model_id = "Salesforce/blip2-opt-2.7b"
    model_path = "./debug_finetuned_blip2_classification_vqa"
else:
    model_id = "Salesforce/blip-vqa-base" #"Salesforce/blip-image-captioning-base"
    model_path = "./FINAL_BLIP1_finetuned_1EPOCH_vqa"

# === Load dataset ===
dataset = load_from_disk(DATASET_PATH)
dataset = dataset.shuffle(seed=42).select(range(SAMPLE_SIZE))

# === Load model and processor ===
if USE_FINETUNED:
    #processor_path = f"{model_path}/processor" #
    #processor = AutoProcessor.from_pretrained(processor_path)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base") if MODEL_TYPE == "blip1" else Blip2Processor.from_pretrained(model_path + "/processor")
    model_cls = Blip2ForConditionalGeneration if MODEL_TYPE == "blip2" else BlipForQuestionAnswering
    model = model_cls.from_pretrained(model_path).to(DEVICE)
else:
    processor = AutoProcessor.from_pretrained(model_id)
    model_cls = Blip2ForConditionalGeneration if MODEL_TYPE == "blip2" else BlipForQuestionAnswering
    model = model_cls.from_pretrained(model_id).to(DEVICE)

#processor.tokenizer.eos_token_id = 2
#model.config.eos_token_id = processor.tokenizer.eos_token_id

#model.eval()

# === Run VQA Inference ===
results = []

for sample in tqdm(dataset):
    image = sample["image"]
    filename = sample["image_filename"]
    description = sample["description"]
    caption = sample["caption"]  # Fallback to description if no caption

    for question in QUESTIONS:

        if MODEL_TYPE == "blip1":
            #prompt = f"Question: {question} Answer:"
            prompt = question  # Adjusted to match the expected input format

            inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)

            output = model.generate(**inputs, max_new_tokens=30)
            answer = processor.decode(output[0], skip_special_tokens=True)

            #answer = processor.decode(output_ids[0], skip_special_tokens=True)

        else:
            prompt = f"Question: {question} Answer:"
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=8,
                    min_new_tokens=1,        # avoid empty answers
                    num_beams=3,
                    do_sample=False,
                    eos_token_id=processor.tokenizer.eos_token_id or 2,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )

            # Slice off the prompt from the left; keep only continuation tokens
            gen_tokens = outputs.sequences[:, inputs["input_ids"].shape[1]:]

            answer = processor.tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()

        results.append({
                "image_filename": filename,
                "true_description": description,
                "question": question,
                "vqa_answer": answer
            })


# === Save results ===
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"VQA results saved to {OUTPUT_CSV}")
