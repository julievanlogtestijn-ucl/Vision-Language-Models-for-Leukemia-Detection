# blip_cross_attn_heatmaps_hooks.py
# Robust cross-attention visualization for BLIP1 using forward hooks.

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
MODEL_PATH      = "./FINAL_BLIP1_finetuned_varieddescriptions"
USE_DATASET     = True                  # set False to use IMAGE_SOURCE below
DATASET_PATH    = "test_data_final"     # HF dataset with an "image" column
SAMPLE_INDEX    = 0
IMAGE_SOURCE    = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"

OUTPUT_DIR      = "./attentionmaps_perhead_hooks"
MAX_NEW_TOKENS  = 120
NUM_BEAMS       = 1

# Layer aggregation: "last" | "mean" | "max" | "idx:<int>"
LAYER_MODE      = "last"

# Head aggregation: "per_head" | "mean" | "max"
HEAD_MODE       = "per_head"
HEAD_LIMIT      = None                  # e.g., 8 to limit saved heads

# Visualization
DROP_CLS        = True                  # drop ViT CLS before reshape to HxW
SCALE_MODE      = "shared_per_token"    # "shared_per_token" | "per_map"
ALPHA           = 0.55

# Extras
SAVE_DIFF       = True                  # save |head - head0| maps
SAVE_STD        = True                  # save std-across-heads map

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Helpers
# =========================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def safe_text(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_", ".", "+", "'")) or "tok"

def load_image(src: str) -> Image.Image:
    if src.startswith("http://") or src.startswith("https://"):
        r = requests.get(src, timeout=20)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    return Image.open(src).convert("RGB")

def get_grid_from_model(model):
    enc = model.vision_model.config
    H = W = enc.image_size // enc.patch_size
    return H, W, 1 + H * W

def reshape_attn_to_grid_raw(attn_1d: torch.Tensor, H: int, W: int, drop_cls=True):
    v = attn_1d
    S = v.numel()
    if drop_cls and S == 1 + H * W:
        v = v[1:]
    elif S != H * W:
        # Fallback interpolate to exact HxW
        s = int(round(math.sqrt(max(1, S))))
        x = v[: s * s].view(1, 1, s, s)
        x = torch.nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x[0, 0]
    return v.view(H, W)

def overlay_heatmap(pil_img: Image.Image, heat2d: torch.Tensor, title=None, savepath=None,
                    alpha=ALPHA, vmin=None, vmax=None):
    img = np.array(pil_img.convert("RGB"))
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

def _first_4d(x):
    # Robustly extract the (B, H, T, S) attention tensor from various return shapes
    if torch.is_tensor(x) and x.ndim == 4:
        return x
    if isinstance(x, (list, tuple)):
        for y in x:
            t = _first_4d(y)
            if t is not None: return t
    if isinstance(x, dict):
        for y in x.values():
            t = _first_4d(y)
            if t is not None: return t
    return None

def aggregate_layers_stacked(cross_stack, layer_mode="last"):
    """
    cross_stack: [L, B, H, T, S]
    """
    if layer_mode == "last":
        return cross_stack[-1]
    elif layer_mode == "mean":
        return cross_stack.mean(dim=0)
    elif layer_mode == "max":
        return cross_stack.max(dim=0).values
    elif layer_mode.startswith("idx:"):
        idx = int(layer_mode.split(":")[1])
        if idx < 0 or idx >= cross_stack.shape[0]:
            raise IndexError(f"LAYER_MODE idx out of range: {idx} not in [0, {cross_stack.shape[0]-1}]")
        return cross_stack[idx]
    else:
        raise ValueError("LAYER_MODE must be 'last'|'mean'|'max'|'idx:<int>'")

# =========================
# Main
# =========================
def main():
    ensure_dir(OUTPUT_DIR)
    print(f"Device: {device}")

    # Load model + processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    # Enable attentions in configs (older versions ignore top-level flags; set nested too)
    for cfg in [
        getattr(model, "config", None),
        getattr(model, "text_decoder", None) and getattr(model.text_decoder, "config", None),
        getattr(model, "text_decoder", None) and getattr(model.text_decoder, "bert", None) and getattr(model.text_decoder.bert, "config", None),
    ]:
        if cfg is not None and hasattr(cfg, "output_attentions"):
            cfg.output_attentions = True
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False  # if available

    # Image
    if USE_DATASET:
        ds = load_from_disk(DATASET_PATH)
        sample = ds[SAMPLE_INDEX]
        image: Image.Image = sample["image"]
    else:
        image = load_image(IMAGE_SOURCE)

    # Grid info
    H, W, expected_src_len = get_grid_from_model(model)
    print(f"[VISION GRID] image_size={model.vision_model.config.image_size} "
          f"patch_size={model.vision_model.config.patch_size} => HxW={H}x{W} "
          f"(expected src_len={expected_src_len})")

    # Generate caption
    img_inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            **img_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            return_dict_in_generate=True,
        )
    seq = gen.sequences[0]
    tokens = processor.tokenizer.convert_ids_to_tokens(seq.tolist())
    decoded = processor.decode(seq, skip_special_tokens=True)
    print("Generated caption:", decoded)
    with open(os.path.join(OUTPUT_DIR, "caption.txt"), "w", encoding="utf-8") as f:
        f.write(decoded + "\n")

    # ---- Hook cross-attention (decoder -> vision) ----
    collected = []  # list of [B, H, T, S], one per layer call (in order)

    def make_hook(li):
        def hook(module, inputs, outputs):
            attn = _first_4d(outputs)
            if attn is not None:
                collected.append(attn.detach())
        return hook

    if not (hasattr(model, "text_decoder") and hasattr(model.text_decoder, "bert")):
        raise RuntimeError("Could not find text decoder; model structure differs.")

    decoder_layers = list(model.text_decoder.bert.encoder.layer)
    hooks = []
    for li, layer in enumerate(decoder_layers):
        if hasattr(layer, "crossattention") and hasattr(layer.crossattention, "self"):
            hooks.append(layer.crossattention.self.register_forward_hook(make_hook(li)))

    try:
        # Step-by-step re-forward
        print("\nRe-forwarding with progressive prefixes to extract cross-attention (via hooks):")
        num_saved = 0
        for pos in range(0, len(seq) - 1):  # last position has no "next" prediction
            prefix_ids = seq[: pos + 1].unsqueeze(0).to(device)  # [1, pos+1]
            last_ctx_tok = tokens[pos]
            next_tok = tokens[pos + 1]

            collected.clear()
            with torch.no_grad():
                _ = model(
                    pixel_values=img_inputs["pixel_values"],
                    input_ids=prefix_ids,       # BLIP uses input_ids for decoder
                    output_attentions=True,     # ensure submodules compute probs
                    return_dict=True,
                )

            if not collected:
                raise RuntimeError("No cross-attention captured via hooks. "
                                   "Check output_attentions flags and hook target.")

            # stack -> [L, B, H, T, S]
            cross_stack = torch.stack(collected, dim=0)
            # choose layer aggregation -> [B, H, T, S]
            cross_layer = aggregate_layers_stacked(cross_stack, LAYER_MODE)
            # take batch 0, last decoder position -> [num_heads, src_len]
            cross_last_step = cross_layer[0, :, -1, :]
            src_len = cross_last_step.shape[-1]
            print(f"[CHECK] pos={pos:02d} ctx='{last_ctx_tok}'→next='{next_tok}': src_len={src_len}, expected {expected_src_len}")

            if src_len not in (expected_src_len, H * W):
                warnings.warn(f"[CHECK] Unexpected src_len={src_len} for HxW={H}x{W}.")

            # Diagnostics
            if src_len == expected_src_len:
                cls_share = cross_last_step[:, 0]
                patch_mass = cross_last_step[:, 1:].sum(dim=1)
                print("  CLS share (first 8):", cls_share[:8].detach().cpu().numpy())
                print("  Patch mass (first 8):", patch_mass[:8].detach().cpu().numpy())
            l1_vs_head0 = (cross_last_step - cross_last_step[0]).abs().mean(dim=1)
            print("  mean |diff| vs head0 (first 8):", l1_vs_head0[:8].detach().cpu().numpy())

            tok_dir = os.path.join(OUTPUT_DIR, f"step_{pos:02d}_{safe_text(next_tok)}")
            ensure_dir(tok_dir)

            # ---- Head aggregation / visualization ----
            if HEAD_MODE == "mean":
                attn_vec = cross_last_step.mean(dim=0)
                grid = reshape_attn_to_grid_raw(attn_vec, H, W, DROP_CLS)
                vmin, vmax = (0.0, float(grid.max().item())) if SCALE_MODE == "shared_per_token" else (None, None)
                overlay_heatmap(
                    image, grid, alpha=ALPHA,
                    title=f"pos {pos:02d} (ctx='{last_ctx_tok}') → '{next_tok}' | heads=mean",
                    savepath=os.path.join(tok_dir, "mean.png"),
                    vmin=vmin, vmax=vmax
                ); num_saved += 1

            elif HEAD_MODE == "max":
                attn_vec = cross_last_step.max(dim=0).values
                grid = reshape_attn_to_grid_raw(attn_vec, H, W, DROP_CLS)
                vmin, vmax = (0.0, float(grid.max().item())) if SCALE_MODE == "shared_per_token" else (None, None)
                overlay_heatmap(
                    image, grid, alpha=ALPHA,
                    title=f"pos {pos:02d} (ctx='{last_ctx_tok}') → '{next_tok}' | heads=max",
                    savepath=os.path.join(tok_dir, "max.png"),
                    vmin=vmin, vmax=vmax
                ); num_saved += 1

            elif HEAD_MODE == "per_head":
                num_heads = cross_last_step.shape[0]
                head_idxs = list(range(num_heads))
                if HEAD_LIMIT is not None:
                    head_idxs = head_idxs[:HEAD_LIMIT]

                # shared scaling per token
                head_grids, vmax_token = [], 0.0
                for h in head_idxs:
                    g = reshape_attn_to_grid_raw(cross_last_step[h], H, W, DROP_CLS)
                    head_grids.append((h, g))
                    vmax_token = max(vmax_token, float(g.max().item()))
                for h, g in head_grids:
                    overlay_heatmap(
                        image, g, alpha=ALPHA,
                        title=f"pos {pos:02d} (ctx='{last_ctx_tok}') → '{next_tok}' | head {h}",
                        savepath=os.path.join(tok_dir, f"head_{h:02d}.png"),
                        vmin=0.0 if SCALE_MODE == "shared_per_token" else None,
                        vmax=vmax_token if SCALE_MODE == "shared_per_token" else None
                    ); num_saved += 1

                if SAVE_DIFF:
                    diff_dir = os.path.join(tok_dir, "diff_vs_head0")
                    ensure_dir(diff_dir)
                    ref = 0
                    diff_grids, vmax_diff = [], 0.0
                    for h in head_idxs:
                        diff_1d = (cross_last_step[h] - cross_last_step[ref]).abs()
                        g = reshape_attn_to_grid_raw(diff_1d, H, W, DROP_CLS)
                        diff_grids.append((h, g))
                        vmax_diff = max(vmax_diff, float(g.max().item()))
                    for h, g in diff_grids:
                        overlay_heatmap(
                            image, g, alpha=0.75,
                            title=f"pos {pos:02d} → '{next_tok}' | |head{h}-head{ref}|",
                            savepath=os.path.join(diff_dir, f"diff_head_{h:02d}.png"),
                            vmin=0.0 if SCALE_MODE == "shared_per_token" else None,
                            vmax=vmax_diff if SCALE_MODE == "shared_per_token" else None
                        )

                if SAVE_STD:
                    std_dir = os.path.join(tok_dir, "std_across_heads")
                    ensure_dir(std_dir)
                    std_1d = cross_last_step.std(dim=0)
                    std_grid = reshape_attn_to_grid_raw(std_1d, H, W, DROP_CLS)
                    overlay_heatmap(
                        image, std_grid, alpha=0.75,
                        title=f"pos {pos:02d} → '{next_tok}' | std across heads",
                        savepath=os.path.join(std_dir, "std_heads.png"),
                        vmin=0.0 if SCALE_MODE == "shared_per_token" else None,
                        vmax=float(std_grid.max().item()) if SCALE_MODE == "shared_per_token" else None
                    )

            else:
                raise ValueError("HEAD_MODE must be 'per_head' | 'mean' | 'max'.")

        print(f"\nSaved maps to: {os.path.abspath(OUTPUT_DIR)}")
    finally:
        for h in hooks:
            h.remove()

if __name__ == "__main__":
    main()
