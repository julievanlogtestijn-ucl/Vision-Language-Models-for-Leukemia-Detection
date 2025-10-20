import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BlipForQuestionAnswering, BlipProcessor
from PIL import Image
import requests
from io import BytesIO

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_TYPE = "blip1"  # "blip1" or "blip2"
finetuned = True  # Set to True if using a finetuned model
model_path = "./debug_finetuned_blip1_classification_vqa"

if finetuned:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base") if MODEL_TYPE == "blip1" else Blip2Processor.from_pretrained(model_path + "/processor")
    model_cls = Blip2ForConditionalGeneration if MODEL_TYPE == "blip2" else BlipForQuestionAnswering
    model = model_cls.from_pretrained(model_path).to(device)
else:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")

# Load test image
url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")

# Prompt
question = "What animals are in this image"

inputs = processor(images=image, text=question, return_tensors="pt").to(device)

output = model.generate(**inputs, max_new_tokens=30)
answer = processor.decode(output[0], skip_special_tokens=True)

print(f"✅ Question: {question}")
print(f"🤖 Answer: {answer}")
