# pip install -U "transformers>=4.50.0" datasets pillow pandas torch accelerate

import torch
import pandas as pd
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

# ===== Config =====
MODEL_ID = "google/medgemma-4b-it"
DATASET_PATH = "test_data_final" #test_data_final or "external_set"
OUTPUT_CSV = "MG_Base_Explain_rank.csv"
SAMPLE_SIZE = 3  # e.g., 100 for a quick run
MAX_NEW_TOKENS = 200 # short answer (cell type only)

#SYSTEM_PROMPT_vqa = "You are a medical visual question answering assistant. Given the image and question, provide a concise but informative answer."

#SYSTEM_PROMPT = "You are a hematology expert."
#USER_TEXT_celltype = "Examine the provided microscopic image of a peripheral blood smear. Identify the cell type shown, only give the cell type, nothing else."
#USER_TEXT = "Examine the provided microscopic image of a peripheral blood smear. Identify the leukemia subtype shown or report back 'healthy' if and only if it is a healthy cell, only give the subtype or healthy label, nothing else."


#USER_TEXT = "Classify this image as either 'healthy' or 'leukemia', provide leukemia subtype if it is leukemia. Also give the cell type. Only give classification and cell type, nothing else."

SYSTEM_PROMPT = "You are a medical image captioning assistant."
USER_TEXT = (
    "Given a microscopy image of a blood cell, "
    "describe the cell's morphological features, including size, nuclear shape, chromatin, cytoplasm, "
    "and any diagnostic clues. Mention the cell type and diagnosis (e.g., leukemia subtype) if evident. "
    "Give plain text description, no formatting."
)

USER_TEXT_prompt = (
    "Given a microscopy image of a blood cell, "
    "give the cell type and diagnosis (e.g., leukemia subtype or healthy) if evident. "
    "Explain how you arrived at this conclusion "
    "Give plain text description, no formatting."
)

SYSTEM_PROMPT_only         = (
    "You are a medical image captioning assistant. Given a microscopy image of a blood cell, "
    "describe the cell's morphological features (size, nuclear shape, chromatin, cytoplasm) "
    "and mention the cell type and diagnosis, for diagnosis choose between the following: healthy, aml, cml, all, cll, apml. "
    "Give me a concise caption, no more than 2 sentences, do not repeat phrases and only describe each feature once."
)

SYSTEM_PROMPT_explain        = (
    "You are a medical image captioning assistant. Given a microscopy image of a blood cell, "
    "Give me the cell type and diagnosis, for diagnosis choose between the following: healthy, aml, cml, all, cll, apml. "
    "Explain me why you chose that specific cell type and diagnosis. "
)

SYSTEM_PROMPT_celltype        = (
    "Give me the cell type in one word only!"
)

SYSTEM_PROMPT_explain_rank       = (
    "Give me the cell type in one word only! After that word, rank the top three most relevant morphological features that led to your conclusion. "
    "Morphological features include cell size, nuclear shape, chromatin pattern, cytoplasm characteristics, and any diagnostic clues. "
    "Format it like this: 'CELL TYPE - rank1. rank2. rank3.'. "
)

# ===== Load model & processor =====
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model.eval()

# ===== Load dataset =====
ds = load_from_disk(DATASET_PATH)
if SAMPLE_SIZE:
    ds = ds.shuffle(seed=42).select(range(SAMPLE_SIZE))

def get_image(rec):
    img = rec.get("image", None)
    if isinstance(img, Image.Image):
        return img
    p = rec.get("image_path") or rec.get("path") or rec.get("file")
    return Image.open(p).convert("RGB") if p else None

def get_filename(rec):
    return rec.get("image_filename") or rec.get("image_path") or rec.get("filename") or f"idx_{rec.get('id', '')}"

def get_label(rec):
    # Your short label like "healthy Monocyte" or "acute promyelocytic leukemia abnormal promyelocyte"
    return (rec.get("caption") or rec.get("description") or "").strip()

def get_descr(rec):
    # Your short label like "healthy Monocyte" or "acute promyelocytic leukemia abnormal promyelocyte"
    return (rec.get("true_description") or rec.get("description") or "").strip()

def get_reference_caption(rec):
    return rec.get("varied_descriptions") or rec.get("description") or ""


rows = []

def guided_prompt_text(cell_type, leukemia_subtype):
    # clean fallbacks
    ct = (cell_type or "").strip()
    dz = (leukemia_subtype or "").strip()

    parts = []
    if ct: parts.append(f"cell type: {ct}")
    if dz: parts.append(f"diagnosis: {dz}")
    label_line = " | ".join(parts) if parts else "no labels provided"

    return (
        "Given a microscopy image of a blood cell, produce a short description of the "
        "morphological features that are CONSISTENT with the following labels.\n\n"
        f"Labels: {label_line}\n\n"
        "Describe: cell size, nuclear shape, chromatin pattern, cytoplasm amount/texture/color, "
        "granules, nucleolus visibility, and any diagnostic clues that support the labels. "
        "Do not contradict the labels. Use plain text (no lists or bullet points)."
    )

for i in tqdm(range(len(ds)), desc="MedGemma 4B zero-shot"):
    rec = ds[i]
    image = get_image(rec)
    #label = get_label(rec)
    if image is None:
        continue

    #What cell type is this,blast,Is this cell healthy or leukemia

    question = "Is this cell healthy or leukemia"

    #cell_type = rec.get("cell_type")
    #leukemia_subtype = rec.get("leukemia_subtype")

    #prompt = guided_prompt_text(cell_type, leukemia_subtype)
    
    """
    # Build chat messages (as on the HF card)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}, #or USER_TEXT question
                {"type": "image", "image": image},
            ],
        },
    ] 
    
    """

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_explain_rank}]},
        {"role": "user", "content": [{"type": "image", "image": image}]},
    ]

    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text":
                 f"Label: {label}\n"
                 f"Task: Describe morphological features that support this label. "
                 f"Do not include the label text itself."},
                {"type": "image", "image": image},
            ],
        },
    ]

    """

    # Apply chat template -> tokens; keep input length to slice prompt off later
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=dtype)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,     # deterministic
            num_beams=3,         # small beam search for reliability
            # eos handled by model/tokenizer defaults
        )
        new_tokens = out_ids[0][input_len:]  # decode only newly generated text

    pred = processor.decode(new_tokens, skip_special_tokens=True).strip()

    rows.append({
        "image_filename": get_filename(rec),
        #"true_description": get_reference_caption(rec),
        #'prompt': prompt,
        "description": get_descr(rec),
        "prediction": pred
    })

pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved zero-shot predictions to {OUTPUT_CSV} (n={len(rows)})")
