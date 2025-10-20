# eval_medgemma_lora.py
import torch, pandas as pd
from datasets import load_from_disk
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from tqdm import tqdm

# ===== Config =====
BASE_MODEL_ID = "google/medgemma-4b-it"
ADAPTER_DIR = "./EINDRESULTAAT_MEDGEMMA_lora"
DATASET_PATH = "eindresultaat_test_data"     # "external_set" or "test_data_final"     
OUTPUT_CSV = "MG_finetuned_rank.csv"
SAMPLE_SIZE = 3
MAX_NEW_TOKENS = 20


SYSTEM_PROMPT_old = (
    "You are a medical image captioning assistant. Given a microscopy image of a blood cell, "
    "describe the cell's morphological features, including size, nuclear shape, chromatin, cytoplasm, "
    "and any diagnostic clues. Mention the cell type and diagnosis (e.g., leukemia subtype) if evident."
)

SYSTEM_PROMPT_only         = (
    "You are a medical image captioning assistant. Given a microscopy image of a blood cell, "
    "describe the cell's morphological features (size, nuclear shape, chromatin, cytoplasm, etc.) "
    "and mention the cell type and whether the cell type is healthy or leukemia. "
    "if the cell is leukemia, mention the leukemia subtype. Keep the description concise yet informative. "
    "Only give me one or two sentences, nothing else, don't repeat features."
)

SYSTEM_PROMPT_explain        = (
    "You are a medical image captioning assistant. Given a microscopy image of a blood cell, "
    "Give me the cell type and diagnosis, for diagnosis choose between the following: healthy, aml, cml, all, cll, apml. "
    "Explain me why you chose that specific cell type and diagnosis. "
)

SYSTEM_PROMPT_explain_rank       = (
    "Give me the cell type in one word only! After that word, rank the top three most relevant morphological features that led to your conclusion. "
    "Morphological features include cell size, nuclear shape, chromatin pattern, cytoplasm characteristics, and any diagnostic clues. "
    "Format it like this: 'CELL TYPE - rank1. rank2. rank3.'. "
)

SYSTEM_PROMPT_old = "You are a medical image captioning assistant."
USER_TEXT = (
    "Given a microscopy image of a blood cell, "
    "describe the cell's morphological features, including size, nuclear shape, chromatin, cytoplasm, "
    "and any diagnostic clues. Mention the cell type and diagnosis (e.g., leukemia subtype) if evident."
)



SYSTEM_PROMPT = "You are a medical image captioning assistant, caption this image. "   

#for cell type:
#SYSTEM_PROMPT = "You are a hematology expert, answer in English only. Examine the provided microscopic image of a peripheral blood smear. Identify the cell type shown, only give the cell type, nothing else."

#SYSTEM_PROMPT_ = "You are a hematology expert, answer in English only. Examine the provided microscopic image of a peripheral blood smear. Identify the leukemia subtype shown or report back 'healthy' if and only if it is a healthy cell, only give the subtype or healthy label, nothing else."

#SYSTEM_PROMPT = "You are a medical visual question answering assistant. Given the image and question, provide a concise but informative answer. Answer in English"


# ===== Load model & processor (base + LoRA adapter) =====
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
base = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_ID, torch_dtype=dtype, device_map="auto"
)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)  # or ADAPTER_DIR (both ok)
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
    return rec.get("image_filename") or rec.get("image_path") or rec.get("filename") or f"idx_{rec.get('id','')}"

def get_reference_caption(rec):
    return rec.get("true_description") or rec.get("description") or "" #true_description or varied_descriptions

#vqa_a1,vqa_q2,vqa_a2,vqa_q3,vqa_a3
def get_question_1(rec):
    q = rec.get('vqa_q1')
    a = rec.get('vqa_a1')
    pair = q + " " + a
    return pair

def get_question_2(rec):
    q = rec.get('vqa_q2')
    a = rec.get('vqa_a2')
    pair = q + " " + a
    return pair

tok = processor.tokenizer

eos_ids = {tok.eos_token_id}
# Common extra stop tokens for chat models:
for key in ("eot_token", "eom_token", "eod_token"):
    t = tok.special_tokens_map.get(key)
    if t:
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None and tid != -1:
            eos_ids.add(tid)

# Also try the literal if it exists in vocab:
for literal in ("<end_of_turn>", "<|eot_id|>"):
    tid = tok.convert_tokens_to_ids(literal)
    if tid is not None and tid != -1:
        eos_ids.add(tid)

eos_ids = list(eos_ids)


rows = []
for i in tqdm(range(len(ds)), desc="MedGemma LoRA eval"):
    rec = ds[i]
    image = get_image(rec)
    if image is None:
        continue

    
    # Messages: system + user(image only). No label text, to match training.
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_explain_rank}]},
        {"role": "user", "content": [{"type": "image", "image": image}]},
    ]

    """

    #What cell type is this,blast,Is this cell healthy or leukemia

    #question = "Is this cell healthy or leukemia"

    messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_TEXT},
                    {"type": "image", "image": image},
                ],
            },
        ] 
    #"""

    # Tokenize via chat template
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device) #, dtype=dtype)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=eos_ids,
            no_repeat_ngram_size=7,
            repetition_penalty=1.2,
            do_sample=False,   # deterministic; set True for diversity
            num_beams=1,
        )
        new_tokens = out_ids[0][input_len:]

    pred = processor.decode(new_tokens, skip_special_tokens=True).strip()

    rows.append({
        "image_filename": get_filename(rec),
        "true_description": get_reference_caption(rec),
        #"questionpair1": get_question_1(rec),
        #"questionpair2": get_question_2(rec),
        "caption": pred,
    })

pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved predictions to {OUTPUT_CSV} (n={len(rows)})")
