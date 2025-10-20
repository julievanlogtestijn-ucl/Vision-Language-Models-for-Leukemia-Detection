# blip_gradcam_crossattn.py — Per-token Grad-CAM over BLIP-1 decoder cross-attention

import os, io, math, warnings, requests
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_from_disk
from transformers import AutoProcessor, BlipForConditionalGeneration

USE_BASE_MODEL  = False
MODEL_PATH_BASE = "Salesforce/blip-image-captioning-base"
MODEL_PATH_FINE = "./FINAL_BLIP1_finetuned_lora_merged"
MODEL_PATH      = MODEL_PATH_BASE if USE_BASE_MODEL else MODEL_PATH_FINE

USE_DATASET      = True
DATASET_PATH     = "test_data_final"
SAMPLE_INDEX     = 137
USE_SINGLE_IMAGE = False
IMAGE_SOURCE     = "./data/sanity_check_images/golden-retriever-tongue-out.jpg"
USE_FOLDER       = False
FOLDER_PATH      = "./data/sanity_check_images"

MAX_NEW_TOKENS   = 80
NUM_BEAMS        = 3
DO_SAMPLE        = False

DROP_CLS         = True
ALPHA            = 0.60
CMAP             = None
OUTPUT_ROOT      = "./blip_gradcam_per_token/finetunedlora"
NORMALIZE_MODE   = "per_token"  # "global" or "per_token"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def safe_text(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_", ".", "+", "'")) or "tok"
def safe_name(s: str) -> str:
    return "".join(ch for ch in os.path.basename(s) if ch.isalnum() or ch in ("-","_",".")) or "img"
def load_image(src: str) -> Image.Image:
    if src.startswith("http://") or src.startswith("https://"):
        r = requests.get(src, timeout=20); r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    return Image.open(src).convert("RGB")
def get_image_from_ds(path, idx):
    ds = load_from_disk(path); rec = ds[idx]
    img = rec.get("image")
    if isinstance(img, Image.Image): return img.convert("RGB")
    p = rec.get("image_path") or rec.get("path") or rec.get("file")
    return Image.open(p).convert("RGB") if p else None
def get_grid_from_model(model):
    enc = model.vision_model.config
    H = W = enc.image_size // enc.patch_size
    return H, W, 1 + H * W
def reshape_to_grid(vec_1d: torch.Tensor, H: int, W: int, drop_cls=True):
    v = vec_1d.to(torch.float32).contiguous().view(-1); S = v.numel()
    if drop_cls and S == 1 + H * W: v = v[1:]; S = v.numel()
    if S == H * W: return v.view(H, W)
    s = int(math.ceil(math.sqrt(max(1, S)))); need = s*s - S
    if need > 0: v = torch.nn.functional.pad(v, (0, need))
    x = v.view(1,1,s,s)
    x = torch.nn.functional.interpolate(x, size=(H,W), mode="bilinear", align_corners=False)
    return x[0,0]
def overlay_with_colorbar(pil_img, heat2d, title, savepath, alpha=ALPHA, cmap=CMAP, vmin=0.0, vmax=None):
    img = np.array(pil_img.convert("RGB"))
    arr = heat2d.detach().to(torch.float32).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img)
    im = ax.imshow(arr, interpolation="bilinear",
                   extent=[0, img.shape[1], img.shape[0], 0],
                   alpha=alpha, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    if title: ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Grad-CAM", rotation=90)
    plt.savefig(savepath, bbox_inches="tight", dpi=180)
    plt.close(fig)
def first_4d(x):
    if torch.is_tensor(x) and x.ndim == 4: return x
    if isinstance(x, (list, tuple)):
        for y in x:
            t = first_4d(y)
            if t is not None: return t
    if isinstance(x, dict):
        for y in x.values():
            t = first_4d(y)
            if t is not None: return t
    return None

def run_one_image(model, processor, image: Image.Image, out_dir: str, device: str):
    ensure_dir(out_dir)
    H, W, expected_src_len = get_grid_from_model(model)
    print(f"[VISION GRID] HxW={H}x{W}, expected src_len={expected_src_len}")

    pixel = processor(images=image, return_tensors="pt").to(device)

    prompt = ""
    text_inputs = processor(text=prompt, return_tensors="pt").to(DEVICE)

    #inputs = processor(image, prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        gen = model.generate(
            **{**pixel, **text_inputs},  
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            do_sample=DO_SAMPLE,
            no_repeat_ngram_size=2, 
            return_dict_in_generate=True,
        )

    #caption = processor.decode(out[0], skip_special_tokens=True)

    seq = gen.sequences[0]
    tokens = processor.tokenizer.convert_ids_to_tokens(seq.tolist())
    decoded = processor.decode(seq, skip_special_tokens=True)
    print("Caption:", decoded)
    with open(os.path.join(out_dir, "caption.txt"), "w", encoding="utf-8") as f:
        f.write(decoded + "\n")

    last_layer_attn = {"tensor": None}
    if not (hasattr(model, "text_decoder") and hasattr(model.text_decoder, "bert")):
        raise RuntimeError("Unexpected BLIP structure. No text_decoder.bert.")
    layers = list(model.text_decoder.bert.encoder.layer)
    if not layers or not hasattr(layers[-1], "crossattention") or not hasattr(layers[-1].crossattention, "self"):
        raise RuntimeError("Could not locate decoder crossattention.self in last layer.")
    def crossattn_forward_hook(module, inp, out):
        t = first_4d(out)
        if t is not None:
            last_layer_attn["tensor"] = t
            t.retain_grad()
    hook = layers[-1].crossattention.self.register_forward_hook(crossattn_forward_hook)

    raw_vecs = []
    ctx_next_pairs = []
    try:
        for pos in range(0, len(seq) - 1):
            prefix_ids = seq[: pos + 1].unsqueeze(0).to(device)  # [1, pos+1]
            target_id  = seq[pos + 1].view(1)                    # [1]
            ctx_tok    = tokens[pos]
            next_tok   = tokens[pos + 1]
            ctx_next_pairs.append((ctx_tok, next_tok))

            last_layer_attn["tensor"] = None

            outputs = model(
                pixel_values=pixel["pixel_values"],
                input_ids=prefix_ids,
                output_attentions=True,
                return_dict=True,
            )
            logits = outputs.logits[0, -1, :]  # [V]
            loss = F.cross_entropy(logits.unsqueeze(0), target_id)

            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=False)

            attn = last_layer_attn["tensor"]
            if attn is None or attn.grad is None:
                raise RuntimeError("Cross-attention tensor or its grad not captured. Check hook.")
            grad = attn.grad
            attn_last = attn[0, :, -1, :]     # [H,S]
            grad_last = grad[0, :, -1, :]     # [H,S]

            if attn_last.shape[-1] != expected_src_len:
                warnings.warn(f"[gradcam] Unexpected src_len={attn_last.shape[-1]} vs {expected_src_len}")

            cam = torch.relu((grad_last * attn_last).mean(dim=0))  # [S]
            raw_vecs.append(cam.detach().cpu())

        if NORMALIZE_MODE == "global" and raw_vecs:
            vmax_global = torch.stack(raw_vecs).max().item() or 1.0
        else:
            vmax_global = None

        per_tok_dir = os.path.join(out_dir, "per_token_gradcam")
        ensure_dir(per_tok_dir)
        for i, vec_cpu in enumerate(raw_vecs):
            vec = vec_cpu.to(torch.float32)
            if NORMALIZE_MODE == "per_token":
                vmax = float(vec.max().item()) or 1.0
                if vmax > 0: vec = vec / vmax
                vmin_plot, vmax_plot = 0.0, 1.0
            else:
                vmin_plot, vmax_plot = 0.0, vmax_global
            grid = reshape_to_grid(vec, H, W, drop_cls=DROP_CLS)
            ctx_tok, next_tok = ctx_next_pairs[i]
            title = f"pos {i:03d} (ctx='{ctx_tok}') → '{next_tok}' [Grad-CAM]"
            savepath = os.path.join(per_tok_dir, f"step_{i:03d}_{safe_text(next_tok)}.png")
            overlay_with_colorbar(image, grid, title, savepath, alpha=ALPHA, cmap=CMAP,
                                  vmin=vmin_plot, vmax=vmax_plot)

        print(f"\n✅ Saved per-token Grad-CAM PNGs to: {os.path.abspath(per_tok_dir)}")

    finally:
        hook.remove()

def main():
    ensure_dir(OUTPUT_ROOT)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    for cfg in (model.config,
                getattr(model, "text_decoder", None) and model.text_decoder.config,
                getattr(model, "text_decoder", None) and getattr(model.text_decoder, "bert", None) and model.text_decoder.bert.config):
        if cfg:
            cfg.output_attentions = True
            cfg.use_cache = False

    items = []
    if USE_DATASET:
        img = get_image_from_ds(DATASET_PATH, SAMPLE_INDEX)
        if img is None: raise RuntimeError("No image from dataset")
        items.append( (img, f"dataset_{SAMPLE_INDEX:05d}.png") )
    if USE_SINGLE_IMAGE:
        img = Image.open(IMAGE_SOURCE).convert("RGB")
        items.append( (img, os.path.basename(IMAGE_SOURCE)) )
    if USE_FOLDER:
        for fname in os.listdir(FOLDER_PATH):
            if fname.lower().endswith((".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff")):
                path = os.path.join(FOLDER_PATH, fname)
                try:
                    img = Image.open(path).convert("RGB")
                    items.append( (img, fname) )
                except Exception as e:
                    print(f"[skip] {fname}: {e}")

    for img, name in items:
        sub = safe_name(name)
        out_dir = os.path.join(OUTPUT_ROOT, sub)
        print(f"\n=== Processing: {name} → {out_dir}")
        run_one_image(model, processor, img, out_dir, DEVICE)

if __name__ == "__main__":
    main()
