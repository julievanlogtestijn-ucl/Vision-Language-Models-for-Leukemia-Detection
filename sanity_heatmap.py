# blip_token_heatmaps_single_image.py
import os
import math
import io
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt
from PIL import Image
import requests

from transformers import AutoProcessor, BlipForConditionalGeneration

# =========================
# Config
# =========================
MODEL_PATH     = "./FINAL_BLIP1_finetuned_varieddescriptions"  # your finetuned BLIP-1 folder (processor+model saved here)
MODEL_ID       = "Salesforce/blip-image-captioning-base"  # base model ID
IMAGE_SOURCE   = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"  # path OR URL
OUTPUT_DIR     = "./attn_maps_single"                          # where to dump PNGs
MAX_NEW_TOKENS = 40
LAYER_MODE     = "last"   # "last" or "mean"
HEAD_MODE      = "mean"   # "mean" (average heads) or "none" (you can extend to save per-head)
DROP_CLS       = True     # drop ViT CLS token before reshaping

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Small helpers
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_text(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_", ".", "+")).strip() or "tok"

def _infer_grid(src_len: int):
    """Try to infer (drop_cls, H, W) where H*W matches encoder patch tokens."""
    def sq(n):
        r = int(math.sqrt(n))
        return (r, r) if r * r == n else None

    if sq(src_len - 1):
        H, W = sq(src_len - 1)
        return True, H, W
    if sq(src_len):
        H, W = sq(src_len)
        return False, H, W
    r = int(round(math.sqrt(max(src_len - 1, 1))))
    return False, r, r

def attn_to_heatmap(attn_1d: torch.Tensor, drop_cls: bool = True):
    """
    attn_1d: [src_len] attention over encoder tokens
    returns a normalized 2D heatmap tensor [H,W]
    """
    v = attn_1d
    src_len = v.numel()
    drop, H, W = _infer_grid(src_len)
    if drop_cls and drop and src_len > 1:
        v = v[1:]

    denom = v.sum() + 1e-8
    v = v / denom

    if v.numel() != H * W:
        s = int(round(math.sqrt(v.numel())))
        x = v[: s * s].view(1, 1, s, s)
        x = torch.nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        heat = x[0, 0]
    else:
        heat = v.view(H, W)

    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat

def overlay_heatmap(pil_img: Image.Image, heatmap: torch.Tensor, alpha=0.5, title=None, savepath=None):
    img = np.array(pil_img.convert("RGB"))
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(heatmap.detach().cpu().numpy(), interpolation="bilinear",
               extent=[0, img.shape[1], img.shape[0], 0], alpha=alpha)
    plt.axis("off")
    if title:
        plt.title(title)
    if savepath:
        plt.savefig(savepath, bbox_inches="tight", dpi=180)
        plt.close(fig)
    else:
        return fig

def _first_4d(x):
    """Recursively find the first 4D attention tensor (B, H, T, S)."""
    if torch.is_tensor(x) and x.ndim == 4:
        return x
    if isinstance(x, (list, tuple)):
        for y in x:
            t = _first_4d(y)
            if t is not None:
                return t
    if isinstance(x, dict):
        for y in x.values():
            t = _first_4d(y)
            if t is not None:
                return t
    return None

def load_image(src: str) -> Image.Image:
    if src.startswith("http://") or src.startswith("https://"):
        r = requests.get(src, timeout=20)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    else:
        return Image.open(src).convert("RGB")

# =========================
# Main
# =========================
def main():
    ensure_dir(OUTPUT_DIR)

    # Load processor+model from your finetuned folder (keeps tokenizer & image size aligned)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    model.eval()

    # Make sure attentions are computed in submodules during our re-forward
    model.config.output_attentions = True
    if hasattr(model, "text_decoder") and hasattr(model.text_decoder, "bert"):
        model.text_decoder.config.output_attentions = True
        model.text_decoder.bert.config.output_attentions = True

    # Load image
    image = load_image(IMAGE_SOURCE)

    # 1) Generate caption (greedy) just like your dataset version
    img_inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            **img_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1,
            return_dict_in_generate=True,
        )
    seq = gen.sequences[0]
    decoded = processor.decode(seq, skip_special_tokens=True)
    print("Generated:", decoded)

    # Save caption
    with open(os.path.join(OUTPUT_DIR, "caption.txt"), "w", encoding="utf-8") as f:
        f.write(decoded + "\n")

    tokens = processor.tokenizer.convert_ids_to_tokens(seq.tolist())

    # 2) Attach hooks to decoder cross-attention SELF modules (BLIP-1)
    collected = []  # list of [B, H, T, S] per layer

    def make_hook(layer_idx):
        def hook(module, inputs, outputs):
            attn = _first_4d(outputs)
            if attn is not None:
                collected.append(attn.detach())
        return hook

    if not (hasattr(model, "text_decoder") and hasattr(model.text_decoder, "bert")):
        raise RuntimeError("Could not find text decoder; model structure differs in this install (this script targets BLIP-1).")

    decoder_layers = list(model.text_decoder.bert.encoder.layer)
    hooks = []
    for i, layer in enumerate(decoder_layers):
        if hasattr(layer, "crossattention") and hasattr(layer.crossattention, "self"):
            hooks.append(layer.crossattention.self.register_forward_hook(make_hook(i)))

    try:
        pixel_values = img_inputs["pixel_values"]

        # 3) Re-forward with progressively longer prefixes and save a heatmap per token
        for i in range(1, len(seq)):  # start at 1 to skip BOS
            prefix_ids = seq[: i + 1].unsqueeze(0).to(device)
            collected.clear()

            with torch.no_grad():
                _ = model(
                    pixel_values=pixel_values,
                    input_ids=prefix_ids,
                    output_attentions=True,
                    return_dict=True,
                )

            if not collected:
                raise RuntimeError("No cross-attention captured. Check hooks and output_attentions flags.")

            # [L, B, H, T, S]
            cross = torch.stack(collected, dim=0)

            # layer aggregation
            if LAYER_MODE == "last":
                cross_layer = cross[-1]                 # [B, H, T, S]
            elif LAYER_MODE == "mean":
                cross_layer = cross.mean(dim=0)         # [B, H, T, S]
            else:
                raise ValueError("LAYER_MODE must be 'last' or 'mean'.")

            # take batch 0, last decoder position -> [H, S]
            cross_last_step = cross_layer[0, :, -1, :]

            if HEAD_MODE == "mean":
                attn_1d = cross_last_step.mean(dim=0)   # [S]
            else:
                attn_1d = cross_last_step.mean(dim=0)   # (placeholder for per-head export)

            heat = attn_to_heatmap(attn_1d, drop_cls=DROP_CLS)

            tok_str = tokens[i]
            title = f"Step {i:02d}: {tok_str}"
            fn = f"step_{i:02d}_{safe_text(tok_str)}.png"
            savepath = os.path.join(OUTPUT_DIR, fn)
            overlay_heatmap(image, heat, alpha=0.5, title=title, savepath=savepath)

        print(f"Saved heatmaps to: {os.path.abspath(OUTPUT_DIR)}")

    finally:
        for h in hooks:
            h.remove()

if __name__ == "__main__":
    main()
