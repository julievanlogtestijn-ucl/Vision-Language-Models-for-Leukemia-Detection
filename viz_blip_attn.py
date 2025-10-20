# blip_token_heatmaps_checked.py
# Visualize BLIP decoder cross-attention over image patches, with robust checks.

import os, math, csv, warnings
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_from_disk
from transformers import AutoProcessor, BlipForConditionalGeneration

# =========================
# Config
# =========================
MODEL_PATH      = "./FINAL_BLIP1_finetuned_varieddescriptions"
DATASET_PATH    = "test_data_final"   # HF dataset that has an "image" column
SAMPLE_INDEX    = 0                   # which sample to visualize
OUTPUT_DIR      = "./attn_maps_checked_sample0"
MAX_NEW_TOKENS  = 100
NUM_BEAMS       = 1                   # keep 1 for greedy for reproducibility

# Aggregation knobs
LAYER_MODE      = "last"              # "last" | "mean" | "max"
HEAD_MODE       = "mean"              # "mean" | "per_head" | "max"
HEAD_LIMIT      = 8                   # if HEAD_MODE=="per_head", how many heads to save per token (first N)

# Normalization/display knobs
DROP_CLS        = True                # drop CLS before reshaping
GLOBAL_MINMAX   = True                # True: same vmin/vmax across frames; False: per-frame min-max
ALPHA           = 0.50                # overlay strength
SAVE_RAW_HEAT   = False               # also save raw .npy heatmaps (after reshape, before display scaling)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Small helpers
# =========================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def safe_text(s: str) -> str:
    s = "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_", ".", "+", "'"))
    return s or "tok"

def get_grid_from_model(model):
    """Return (H, W, expected_src_len) using the vision config."""
    if not hasattr(model, "vision_model") or not hasattr(model.vision_model, "config"):
        raise RuntimeError("Model has no vision_model.config; cannot compute patch grid.")
    enc = model.vision_model.config
    if not hasattr(enc, "image_size") or not hasattr(enc, "patch_size"):
        raise RuntimeError("vision_model.config is missing image_size/patch_size.")
    H = W = enc.image_size // enc.patch_size
    expected_src_len = 1 + H * W
    return H, W, expected_src_len

def tokens_from_ids(tokenizer, ids):
    return tokenizer.convert_ids_to_tokens(ids.tolist())

def entropy_from_probs(p: torch.Tensor) -> float:
    """Shannon entropy (base e). p is 1D (src_len), assumed >= 0 and sums to 1."""
    p = p.clamp_min(1e-12)
    return float(-(p * p.log()).sum().item())

def overlay_heatmap(pil_img: Image.Image, heat2d: torch.Tensor, title=None, savepath=None,
                    alpha=0.5, vmin=None, vmax=None):
    img = np.array(pil_img.convert("RGB"))
    H, W = heat2d.shape
    arr = heat2d.detach().cpu().numpy()
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(arr, interpolation="bilinear",
               extent=[0, img.shape[1], img.shape[0], 0],
               alpha=alpha, vmin=vmin, vmax=vmax)
    plt.axis("off")
    if title: plt.title(title)
    if savepath:
        plt.savefig(savepath, bbox_inches="tight", dpi=180)
        plt.close(fig)
    else:
        return fig

def reshape_attn_to_grid(attn_1d: torch.Tensor, H: int, W: int, drop_cls=True):
    """
    attn_1d: [src_len] distribution over encoder tokens (should sum ~1 across S per head/step)
    Returns [H,W] after (optionally) dropping CLS and reshaping; falls back to interpolate if needed.
    """
    v = attn_1d
    src_len = v.numel()
    if drop_cls and src_len == 1 + H * W:
        v = v[1:]
    elif src_len == H * W:
        pass
    else:
        # Fallback with warning + interpolate to exact (H, W)
        warnings.warn(
            f"[reshape_attn_to_grid] Unexpected src_len={src_len} for HxW={H}x{W}. "
            f"Interpolating; check your model image/patch size."
        )
        s = int(round(math.sqrt(max(1, src_len))))
        x = v[: s * s].view(1, 1, s, s)
        x = torch.nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        heat = x[0, 0]
        # normalize to [0,1] for display use; keep raw before display if desired
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-12)
        return heat
    heat = v.view(H, W)
    # Normalize to [0,1] for display, but don't destroy relative structure
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-12)
    return heat

def aggregate_layers_heads(cross_tuple, pos_idx: int, layer_mode="last", head_mode="mean"):
    """
    cross_tuple: tuple of length L; each item shape (B, H, T, S)
    pos_idx: decoder time index to read (usually -1)
    Returns:
        - if head_mode != 'per_head': attn_1d [S] (heads reduced)
        - if head_mode == 'per_head': attn_heads [H, S]
    """
    if not isinstance(cross_tuple, (list, tuple)) or len(cross_tuple) == 0:
        raise RuntimeError("cross_attentions missing or empty; enable output_attentions or fix hooks.")
    # stack -> [L, B, H, T, S]
    cross = torch.stack([c for c in cross_tuple], dim=0)
    B = cross.shape[1]
    cross = cross[:, 0, :, :, :]            # take batch 0 => [L, H, T, S]
    if pos_idx < 0: pos_idx = cross.shape[2] + pos_idx
    cross = cross[:, :, pos_idx, :]          # [L, H, S]

    # Layer reduction
    if layer_mode == "last":
        cross = cross[-1]                    # [H, S]
    elif layer_mode == "mean":
        cross = cross.mean(dim=0)            # [H, S]
    elif layer_mode == "max":
        cross = cross.max(dim=0).values      # [H, S]
    else:
        raise ValueError("layer_mode must be 'last' | 'mean' | 'max'")

    # Head reduction or per-head
    if head_mode == "mean":
        return cross.mean(dim=0)             # [S]
    elif head_mode == "max":
        return cross.max(dim=0).values       # [S]
    elif head_mode == "per_head":
        return cross                          # [H, S]
    else:
        raise ValueError("head_mode must be 'mean' | 'max' | 'per_head'")

# =========================
# Main
# =========================
def main():
    ensure_dir(OUTPUT_DIR)
    print(f"Device: {device}")

    # Load model + processor together (keeps tokenizer/preproc aligned)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    # Turn on attentions at all relevant places
    model.config.output_attentions = True
    if hasattr(model, "text_decoder"):
        model.text_decoder.config.output_attentions = True
        if hasattr(model.text_decoder, "bert"):
            model.text_decoder.bert.config.output_attentions = True

    # Dataset / image
    ds = load_from_disk(DATASET_PATH)
    sample = ds[SAMPLE_INDEX]
    image: Image.Image = sample["image"]

    # Compute exact grid from model config
    H, W, expected_src_len = get_grid_from_model(model)
    print(f"[VISION GRID] image_size={model.vision_model.config.image_size} "
          f"patch_size={model.vision_model.config.patch_size} => HxW={H}x{W} "
          f"(expected src_len={expected_src_len})")

    # ========= 1) Generate a caption =========
    img_inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            **img_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            return_dict_in_generate=True,
        )
    seq = gen.sequences[0]  # [total_len]
    tokens = tokens_from_ids(processor.tokenizer, seq)
    decoded = processor.decode(seq, skip_special_tokens=True)
    print("Generated caption:", decoded)

    with open(os.path.join(OUTPUT_DIR, "caption.txt"), "w", encoding="utf-8") as f:
        f.write(decoded + "\n")

    # ========= 2) Re-forward step-by-step and collect cross-attention =========
    # We will read the attention at the LAST decoder position of each prefix,
    # which is the attention used to PREDICT the NEXT token.
    all_heat_raw = []      # before global scaling, 2D tensors
    meta_rows = []         # rows for CSV

    # For stability, disable KV caching so shapes are always [1, i, ...]
    print("\nRe-forwarding with progressive prefixes to extract cross-attention:")
    for pos in range(0, len(seq) - 1):  # last position has no "next" prediction
        prefix_ids = seq[: pos + 1].unsqueeze(0).to(device)  # [1, pos+1]
        last_ctx_tok = tokens[pos]
        next_tok = tokens[pos + 1]

        with torch.no_grad():
            out = model(
                pixel_values=img_inputs["pixel_values"],
                decoder_input_ids=prefix_ids,   # IMPORTANT: explicitly use the decoder
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

        cross_tuple = getattr(out, "cross_attentions", None)
        if not cross_tuple:
            raise RuntimeError(
                "No cross attentions returned. Make sure model/text_decoder configs have output_attentions=True."
            )

        # Quick shape checks (layer, batch, heads, tgt_len, src_len)
        L = len(cross_tuple)
        b0_shape = tuple(cross_tuple[0].shape)
        src_len = b0_shape[-1]
        tgt_len = b0_shape[-2]
        heads   = b0_shape[1]
        if src_len not in (expected_src_len, H*W):  # allow both (with/without CLS)
            warnings.warn(
                f"[CHECK] src_len={src_len} differs from expected {expected_src_len} (1+H*W) "
                f"or {H*W}. Are image_size/patch_size consistent?"
            )

        attn = aggregate_layers_heads(
            cross_tuple, pos_idx=-1, layer_mode=LAYER_MODE, head_mode=HEAD_MODE
        )

        if HEAD_MODE == "per_head":
            # Save a grid of per-head maps for the first HEAD_LIMIT heads
            save_dir = os.path.join(OUTPUT_DIR, f"per_head_step_{pos:02d}")
            ensure_dir(save_dir)
            head_count = min(attn.shape[0], HEAD_LIMIT)
            # Normalize per head to keep distributions (they already sum to 1 across S)
            entropies = []
            for h in range(head_count):
                h_vec = attn[h]                       # [S]
                entropies.append(entropy_from_probs(h_vec))
                heat2d = reshape_attn_to_grid(h_vec, H, W, drop_cls=DROP_CLS)
                title = f"pos {pos:02d} (ctx='{last_ctx_tok}') → next '{next_tok}' | head {h}"
                overlay_heatmap(
                    image, heat2d, title=title,
                    savepath=os.path.join(save_dir, f"head_{h:02d}.png"),
                    alpha=ALPHA
                )
            meta_rows.append({
                "pos": pos,
                "ctx_token": last_ctx_tok,
                "next_token": next_tok,
                "mode": "per_head",
                "num_heads": int(attn.shape[0]),
                "entropy_mean": float(np.mean(entropies)),
                "entropy_min": float(np.min(entropies)),
                "ent
