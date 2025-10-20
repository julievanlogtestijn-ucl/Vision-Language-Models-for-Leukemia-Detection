import os, math, warnings
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_from_disk
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# =========================
# Config
# =========================
BASE_MODEL_ID   = "google/medgemma-4b-it"

# Use base-only? Set True to ignore ADAPTER_DIR entirely.
USE_BASE_ONLY   = True
ADAPTER_DIR     = "./FINAL_MG_finetuned_varieddescriptions_new"   # ignored if USE_BASE_ONLY=True

# Choose one input mode:
USE_DATASET     = False
DATASET_PATH    = "test_data_final"
SAMPLE_INDEX    = 0

USE_SINGLE_IMAGE = True
IMAGE_SOURCE     = "./data/sanity_check_images/example.jpg"

USE_FOLDER      = False
FOLDER_PATH     = "./data/sanity_check_images"

SYSTEM_PROMPT   = (
    "You are a medical image captioning assistant. Given a microscopy image of a blood cell, "
    "describe the cell's morphological features (size, nuclear shape, chromatin, cytoplasm) "
    "and mention the likely cell type and diagnosis if evident. "
    "Give me a concise caption, no more than 2 sentences, do not repeat phrases."
)

OUTPUT_ROOT     = "./mg_token_attn_maps"   # a subfolder per image will be created
MAX_NEW_TOKENS  = 120
NUM_BEAMS       = 1
PREFER_DURING_GENERATE = True

# Visualization
ALPHA           = 0.60
DROP_CLS        = True
HEAD_AGG        = "mean"       # "mean" | "max" | "per_head"
HEAD_LIMIT      = None
SCALE_MODE      = "shared_per_token"  # "shared_per_token" or "per_map"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Helpers
# =========================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def safe_name(s: str) -> str:
    return "".join(ch for ch in os.path.basename(s) if ch.isalnum() or ch in ("-","_",".")) or "img"

def reshape_attn_to_grid(attn_1d: torch.Tensor, H: int, W: int, drop_cls: bool):
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

def overlay_with_colorbar(pil_img: Image.Image, heat2d: torch.Tensor, title: str, save_path: str,
                          alpha: float = 0.6, vmin=None, vmax=None):
    img = np.array(pil_img.convert("RGB"))
    arr = heat2d.detach().to(torch.float32).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    im = ax.imshow(arr, interpolation="bilinear",
                   extent=[0, img.shape[1], img.shape[0], 0],
                   alpha=alpha, vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Attention", rotation=270, labelpad=12)
    fig.savefig(save_path, bbox_inches="tight", dpi=180)
    plt.close(fig)

def get_image_token_span(input_ids: torch.Tensor, tokenizer):
    ids = input_ids[0].tolist()
    boi_tok = tokenizer.special_tokens_map.get("boi_token", None)
    eoi_tok = tokenizer.special_tokens_map.get("eoi_token", None)
    boi_id = tokenizer.convert_tokens_to_ids(boi_tok) if boi_tok else None
    eoi_id = tokenizer.convert_tokens_to_ids(eoi_tok) if eoi_tok else None
    for lit in ("<image>", "<|image|>"):
        if boi_id in (None, -1):
            tid = tokenizer.convert_tokens_to_ids(lit)
            if tid is not None and tid != -1:
                boi_id = tid
                eoi_id = tid
    if boi_id in (None, -1) or eoi_id in (None, -1):
        return None, None, 0
    try:
        boi_idx = ids.index(boi_id)
        eoi_idx = next(i for i in range(boi_idx, len(ids)) if ids[i] == eoi_id)
        count = (eoi_idx - boi_idx + 1)
        return boi_idx, eoi_idx, count
    except Exception:
        return None, None, 0

def aggregate_heads(tensor_3d: torch.Tensor, mode: str):
    if mode == "mean":
        return tensor_3d.mean(dim=0)
    if mode == "max":
        return tensor_3d.max(dim=0).values
    if mode == "per_head":
        return tensor_3d
    raise ValueError("HEAD_AGG must be 'mean' | 'max' | 'per_head'")

# =========================
# One image runner
# =========================
def run_one_image(model, processor, image: Image.Image, out_dir: str):
    ensure_dir(out_dir)

    # chat
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image}]},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_tensors="pt", return_dict=True
    ).to(DEVICE)

    pixel_values = processor.image_processor(images=[image], return_tensors="pt")["pixel_values"].to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **{**inputs, "pixel_values": pixel_values},
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            return_dict_in_generate=True,
            output_attentions=True,
        )

    has_attn = hasattr(out, "attentions") and out.attentions is not None
    print("✅ Attentions during generate." if has_attn else "⚠️ No generate-time attentions; using re-forward.")

    seq_all = out.sequences[0]
    prompt_len = inputs["input_ids"].shape[1]
    gen_only = seq_all[prompt_len:]
    decoded = processor.decode(gen_only, skip_special_tokens=True)
    print("Caption:", decoded)
    with open(os.path.join(out_dir, "caption.txt"), "w", encoding="utf-8") as f:
        f.write(decoded.strip() + "\n")

    # 56x56 grid for MedGemma 4B
    H = W = 56

    boi_idx, eoi_idx, _ = get_image_token_span(seq_all.unsqueeze(0), processor.tokenizer)
    if boi_idx is None:
        warnings.warn("Could not locate <image> tokens; using full source len.")

    def save_map(attn_vec_1d, pos, now_tok, next_tok, root, head_tag=None):
        if boi_idx is not None and eoi_idx is not None:
            attn_vec_1d = attn_vec_1d[..., boi_idx:eoi_idx + 1]
        grid = reshape_attn_to_grid(attn_vec_1d, H, W, drop_cls=DROP_CLS)
        vmin = 0.0 if SCALE_MODE == "shared_per_token" else None
        vmax = float(grid.max().item()) if SCALE_MODE == "shared_per_token" else None
        title = f"pos {pos:02d} '{now_tok}' → '{next_tok}'"
        if head_tag: title += f" | {head_tag}"
        path = os.path.join(root, f"step_{pos:03d}{'' if head_tag is None else '_' + head_tag.replace('=','-')}.png")
        overlay_with_colorbar(image, grid, title, path, alpha=ALPHA, vmin=vmin, vmax=vmax)
        np.save(path.replace(".png", ".npy"), grid.detach().to(torch.float32).cpu().numpy())

    toks = processor.tokenizer.convert_ids_to_tokens(seq_all.tolist())

    if PREFER_DURING_GENERATE and has_attn:
        save_root = os.path.join(out_dir, "per_token_during_generate")
        ensure_dir(save_root)
        attn_steps = out.attentions  # tuple over gen steps
        for k, step_attn in enumerate(attn_steps):
            last_layer = step_attn[-1][0]         # [H, T_tgt, T_src]
            attn_vec_heads = last_layer[:, -1, :] # [H, T_src]
            if HEAD_AGG == "per_head":
                head_idxs = list(range(attn_vec_heads.shape[0]))
                if HEAD_LIMIT is not None:
                    head_idxs = head_idxs[:HEAD_LIMIT]
                vmax_shared = 0.0
                grids = []
                for h in head_idxs:
                    g = reshape_attn_to_grid(attn_vec_heads[h], H, W, drop_cls=DROP_CLS)
                    grids.append((h, g))
                    vmax_shared = max(vmax_shared, float(g.max().item()))
                for h, g in grids:
                    vmin = 0.0 if SCALE_MODE == "shared_per_token" else None
                    vmax = vmax_shared if SCALE_MODE == "shared_per_token" else None
                    now_tok = processor.tokenizer.convert_ids_to_tokens([gen_only[k].item()])[0]
                    next_tok = processor.tokenizer.convert_ids_to_tokens([gen_only[k + 1].item()])[0] if k + 1 < len(gen_only) else "<eos>"
                    overlay_with_colorbar(image, g, f"pos {k:02d} '{now_tok}' → '{next_tok}' | head {h}",
                                          os.path.join(save_root, f"step_{k:03d}_head_{h:02d}.png"),
                                          alpha=ALPHA, vmin=vmin, vmax=vmax)
                    np.save(os.path.join(save_root, f"step_{k:03d}_head_{h:02d}.npy"),
                            g.detach().to(torch.float32).cpu().numpy())
            else:
                agg = aggregate_heads(attn_vec_heads, HEAD_AGG)
                now_tok = processor.tokenizer.convert_ids_to_tokens([gen_only[k].item()])[0]
                next_tok = processor.tokenizer.convert_ids_to_tokens([gen_only[k + 1].item()])[0] if k + 1 < len(gen_only) else "<eos>"
                save_map(agg, k, now_tok, next_tok, save_root, head_tag=f"heads={HEAD_AGG}")
        print(f"✅ Saved generation-time maps to: {os.path.abspath(save_root)}")
    else:
        # Re-forward fallback
        save_root = os.path.join(out_dir, "per_token_reforward")
        ensure_dir(save_root)
        seq_all_cpu = seq_all
        num_steps = len(seq_all_cpu) - 1
        for pos in range(num_steps):
            prefix_ids = seq_all_cpu[: pos + 1].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = model(
                    input_ids=prefix_ids,
                    pixel_values=pixel_values,
                    output_attentions=True,
                    return_dict=True,
                )
            dec_attns = outputs.attentions
            last_layer = dec_attns[-1][0]
            attn_vec_heads = last_layer[:, -1, :]
            now_tok = toks[pos]
            next_tok = toks[pos + 1] if pos + 1 < len(toks) else "<eos>"
            if HEAD_AGG == "per_head":
                head_idxs = list(range(attn_vec_heads.shape[0]))
                if HEAD_LIMIT is not None:
                    head_idxs = head_idxs[:HEAD_LIMIT]
                vmax_shared = 0.0
                grids = []
                for h in head_idxs:
                    vec = attn_vec_heads[h]
                    if boi_idx is not None and eoi_idx is not None:
                        vec = vec[boi_idx:eoi_idx + 1]
                    g = reshape_attn_to_grid(vec, H, W, drop_cls=DROP_CLS)
                    grids.append((h, g))
                    vmax_shared = max(vmax_shared, float(g.max().item()))
                for h, g in grids:
                    vmin = 0.0 if SCALE_MODE == "shared_per_token" else None
                    vmax = vmax_shared if SCALE_MODE == "shared_per_token" else None
                    title = f"pos {pos:02d} '{now_tok}' → '{next_tok}' | head {h}"
                    path  = os.path.join(save_root, f"step_{pos:03d}_head_{h:02d}.png")
                    overlay_with_colorbar(image, g, title, path, alpha=ALPHA, vmin=vmin, vmax=vmax)
                    np.save(os.path.join(save_root, f"step_{pos:03d}_head_{h:02d}.npy"),
                            g.detach().to(torch.float32).cpu().numpy())
            else:
                agg = aggregate_heads(attn_vec_heads, HEAD_AGG)
                if boi_idx is not None and eoi_idx is not None:
                    agg = agg[boi_idx:eoi_idx + 1]
                save_map(agg, pos, now_tok, next_tok, save_root, head_tag=f"heads={HEAD_AGG}")
        print(f"✅ Saved re-forward maps to: {os.path.abspath(save_root)}")

# =========================
# Main: dataset / single / folder
# =========================
def main():
    ensure_dir(OUTPUT_ROOT)
    print(f"Device: {DEVICE}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    if USE_BASE_ONLY:
        model = base
    else:
        model = PeftModel.from_pretrained(base, ADAPTER_DIR) if (ADAPTER_DIR and os.path.isdir(ADAPTER_DIR)) else base

    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, use_fast=False)
    model.config.output_attentions = True
    model.config.use_cache = False
    model.eval()

    items = []
    if USE_DATASET:
        ds = load_from_disk(DATASET_PATH)
        img = ds[SAMPLE_INDEX].get("image")
        if not isinstance(img, Image.Image):
            p = ds[SAMPLE_INDEX].get("image_path") or ds[SAMPLE_INDEX].get("path") or ds[SAMPLE_INDEX].get("file")
            img = Image.open(p).convert("RGB")
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
        run_one_image(model, processor, img, out_dir)

if __name__ == "__main__":
    main()
