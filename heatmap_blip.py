import os, io, math, warnings, requests
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_from_disk
from transformers import AutoProcessor, BlipForConditionalGeneration

# =========================
# Config
# =========================

# Use base checkpoint (no finetune):
USE_BASE_MODEL  = True
MODEL_PATH_BASE = "Salesforce/blip-image-captioning-base"
MODEL_PATH_FINE = "./FINAL_BLIP1_finetuned_varieddescriptions"
MODEL_PATH      = MODEL_PATH_BASE if USE_BASE_MODEL else MODEL_PATH_FINE

# Choose input mode(s)
USE_DATASET     = True
DATASET_PATH    = "test_data_final"
SAMPLE_INDEX    = 15

USE_SINGLE_IMAGE = False
IMAGE_SOURCE     = "./data/sanity_check_images/golden-retriever-tongue-out.jpg"

USE_FOLDER      = False
FOLDER_PATH     = "./data/sanity_check_images"

MAX_NEW_TOKENS  = 120
NUM_BEAMS       = 1
DO_SAMPLE       = False

LAYER_MODE      = "last"   # "last" | "mean" | "max" | "idx:<int>"
HEAD_MODE       = "mean"   # "mean" | "max"
DROP_CLS        = True
ALPHA           = 0.60
CMAP            = None
OUTPUT_ROOT     = "./blip_per_token_heatmaps"

# =========================
# Helpers
# =========================
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def safe_text(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_", ".", "+", "'")) or "tok"
def safe_name(s: str) -> str:
    return "".join(ch for ch in os.path.basename(s) if ch.isalnum() or ch in ("-","_",".")) or "img"

def get_image_from_ds(path, idx):
    ds = load_from_disk(path)
    rec = ds[idx]
    img = rec.get("image")
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    p = rec.get("image_path") or rec.get("path") or rec.get("file")
    return Image.open(p).convert("RGB") if p else None

def reshape_to_grid(attn_1d: torch.Tensor, H: int, W: int, drop_cls=True):
    v = attn_1d.to(torch.float32).contiguous().view(-1)
    S = v.numel()
    if drop_cls and S == 1 + H * W:
        v = v[1:]
        S = v.numel()
    if S == H * W:
        return v.view(H, W)
    s = int(math.ceil(math.sqrt(max(1, S))))
    need = s * s - S
    if need > 0:
        v = torch.nn.functional.pad(v, (0, need))
    x = v.view(1, 1, s, s)
    x = torch.nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    return x[0, 0]

def overlay_with_colorbar(pil_img, heat2d, title, savepath, alpha=ALPHA, cmap=CMAP, vmin=0.0, vmax=None):
    img = np.array(pil_img.convert("RGB"))
    arr = heat2d.detach().to(torch.float32).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    im = ax.imshow(arr, interpolation="bilinear",
                   extent=[0, img.shape[1], img.shape[0], 0],
                   alpha=alpha, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("attention", rotation=90)
    plt.savefig(savepath, bbox_inches="tight", dpi=180)
    plt.close(fig)

def aggregate_layers(stack, mode="last"):
    if mode == "last":
        return stack[-1]
    if mode == "mean":
        return stack.mean(dim=0)
    if mode == "max":
        return stack.max(dim=0).values
    if mode.startswith("idx:"):
        idx = int(mode.split(":")[1]); return stack[idx]
    raise ValueError("LAYER_MODE must be 'last'|'mean'|'max'|'idx:<int>'")

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

def get_grid_from_model(model):
    enc = model.vision_model.config
    H = W = enc.image_size // enc.patch_size
    return H, W, 1 + H * W

# =========================
# Core runner for one image
# =========================
def run_one_image(model, processor, image: Image.Image, out_dir: str, device: str):
    ensure_dir(out_dir)
    H, W, expected_src_len = get_grid_from_model(model)
    print(f"[VISION GRID] HxW={H}x{W}, expected src_len={expected_src_len}")

    pixel_inputs = processor(images=image, return_tensors="pt").to(device)

    # Generate text (no attentions exposed reliably for BLIP)
    with torch.no_grad():
        gen = model.generate(
            **pixel_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            do_sample=DO_SAMPLE,
            return_dict_in_generate=True,
        )

    seq = gen.sequences[0]
    tokens = processor.tokenizer.convert_ids_to_tokens(seq.tolist())
    decoded = processor.decode(seq, skip_special_tokens=True)
    print("Generated:", decoded)
    with open(os.path.join(out_dir, "caption.txt"), "w", encoding="utf-8") as f:
        f.write(decoded + "\n")

    # Hook cross-attn
    collected = []

    def hook_fn(module, inp, out):
        t = first_4d(out)
        if t is not None:
            collected.append(t.detach())

    if not (hasattr(model, "text_decoder") and hasattr(model.text_decoder, "bert")):
        raise RuntimeError("Unexpected BLIP structure. No text_decoder.bert.")
    layers = list(model.text_decoder.bert.encoder.layer)

    hooks = []
    for lyr in layers:
        if hasattr(lyr, "crossattention") and hasattr(lyr.crossattention, "self"):
            hooks.append(lyr.crossattention.self.register_forward_hook(hook_fn))

    per_tok_dir = os.path.join(out_dir, "per_token_reforward")
    ensure_dir(per_tok_dir)

    try:
        for pos in range(0, len(seq) - 1):
            prefix_ids = seq[: pos + 1].unsqueeze(0).to(device)
            ctx_tok  = tokens[pos]
            next_tok = tokens[pos + 1]

            collected.clear()
            with torch.no_grad():
                _ = model(
                    pixel_values=pixel_inputs["pixel_values"],
                    input_ids=prefix_ids,
                    output_attentions=True,
                    return_dict=True,
                )

            if not collected:
                raise RuntimeError("No cross-attention captured. Check output_attentions flags and hooks.")

            stack = torch.stack(collected, dim=0)     # [L,B,H,T,S]
            cross = aggregate_layers(stack, LAYER_MODE)  # [B,H,T,S]
            cross_last = cross[0, :, -1, :]           # [H,S]

            if cross_last.shape[-1] != expected_src_len:
                warnings.warn(f"[ref] Unexpected src_len={cross_last.shape[-1]} vs expected {expected_src_len}")

            if HEAD_MODE == "mean":
                attn_vec = cross_last.mean(0)
            elif HEAD_MODE == "max":
                attn_vec = cross_last.max(0).values
            else:
                raise ValueError("HEAD_MODE must be 'mean' or 'max'.")

            vec = attn_vec.clamp_min(0)
            vmax = float(vec.max().item()) if vec.numel() > 0 else 1.0
            if vmax > 0: vec = vec / vmax
            grid = reshape_to_grid(vec, H, W, drop_cls=DROP_CLS)

            title = f"pos {pos:03d} (ctx='{ctx_tok}') → '{next_tok}'"
            savepath = os.path.join(per_tok_dir, f"step_{pos:03d}_{safe_text(next_tok)}.png")
            overlay_with_colorbar(image, grid, title, savepath, alpha=ALPHA, cmap=CMAP, vmin=0.0, vmax=1.0)

        print(f"✅ Saved per-token PNGs to: {os.path.abspath(per_tok_dir)}")

    finally:
        for h in hooks:
            h.remove()

# =========================
# Main: dataset / single / folder
# =========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(OUTPUT_ROOT)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    # ensure attentions not hidden by cache
    if hasattr(model, "config"):
        model.config.output_attentions = True
        model.config.use_cache = False
    if hasattr(model, "text_decoder") and hasattr(model.text_decoder, "config"):
        model.text_decoder.config.output_attentions = True
        model.text_decoder.config.use_cache = False
    if hasattr(model, "text_decoder") and hasattr(model.text_decoder, "bert") and hasattr(model.text_decoder.bert, "config"):
        model.text_decoder.bert.config.output_attentions = True
        model.text_decoder.bert.config.use_cache = False

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
        run_one_image(model, processor, img, out_dir, device)

if __name__ == "__main__":
    main()
