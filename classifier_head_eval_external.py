#!/usr/bin/env python3
"""
evaluate_external_set.py

Run ALL saved classifier heads (VLM backbones + no-backbone baselines) on an external HF dataset
loaded via load_from_disk("external_set"), and export a single consolidated CSV of predictions.

Assumptions:
- Heads saved by your training scripts at:
    ./classifier_head_output/head_<task>_<backbone>.pt
    ./classifier_head_nobackbone_output/head_<task>_<backbone>.pt
- Optional meta JSONs exist next to eval files (useful for baseline params):
    ./*/eval_<task>_<backbone}_meta.json
- External dataset has columns: image (Image), image_filename (str), true_description (str), source (str)

Outputs:
- ./external_eval/predictions_external.csv  (one row per image × model, with per-class probs)

Author: you :)
"""

import os, json, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image as PILImage

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

import torchvision.models as models
from torchvision import transforms
import dinov2.models.vision_transformer as vits

# -------------------------
# Config (edit as needed)
# -------------------------
CONFIG = {
    # Where to discover saved heads
    "head_dirs": [
        "./backbone_experiment"
        #"./classifier_head_output",
        #"./classifier_head_nobackbone_output",
    ],
    "dinobloom_b_path": "./DinoBloom-B.pth",

    # Limit to these tasks; heads must match one of these
    "tasks": ["leukemia_subtype", "cell_type"],

    # If not None, include only these backbone suffixes (filenames use head_<task>_<backbone>.pt)
    # e.g., ["blip_base","blip_ft","medgemma_base","medgemma_ft","nobackbone_random_proj"]
    "backbones_filter": ["blip_base","blip_ft","medgemma_base","nobackbone_random_proj", "resnet50", "dinobloom_b"],

    # External dataset path for load_from_disk
    "external_disk_dir": "eindresultaat_external_data",

    # Batch size for feature extraction
    "batch_size": 32,
    "num_workers": 4,

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Model IDs / paths for VLM extractors
    "blip_base_path": "Salesforce/blip-image-captioning-base",
    "blip_ft_path": "./EINDRESULTAAT_BLIP1_fullyfinetuned",  # your finetuned BLIP-1 path

    "medgemma_base_path": "google/medgemma-4b-it",
    #"medgemma_ft_path": "./FINAL_MG_finetuned_varieddescriptions_new",

    # Output
    "out_dir": "./external_eval",
    "out_csv": "predictions_external.csv",
}

# -------------------------
# Classes & utilities
# -------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

# --- Dataset (no labels) ---

class ExternalImageDataset(Dataset):
    """
    Wrap external HF dataset; returns (PIL.Image, meta_dict).
    Expects columns: image, image_filename (optional), true_description (optional), source (optional)
    """
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item.get("image", None)
        if isinstance(img, PILImage.Image):
            pil = img.convert("RGB")
        elif isinstance(img, dict) and "path" in img:
            pil = PILImage.open(img["path"]).convert("RGB")
        elif isinstance(img, str):
            pil = PILImage.open(img).convert("RGB")
        else:
            pil = PILImage.fromarray(np.array(img)).convert("RGB")
        meta = {
            "image_filename": item.get("image_filename", None),
            "true_description": item.get("true_description", None),
            "source": item.get("source", None),
        }
        return pil, meta

def collate_images_with_meta(batch):
    imgs, metas = zip(*batch)
    return list(imgs), list(metas)

# --- Heads (same as your training scripts) ---

class LinearOrMLPHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: Optional[int] = None, p: float = 0.1):
        super().__init__()
        if hidden is None:
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, n_classes))
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(p),
                nn.Linear(hidden, n_classes),
            )
    def forward(self, x): return self.net(x)

class CosineHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, init_s: float = 10.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, in_dim))
        nn.init.kaiming_uniform_(self.W, a=np.sqrt(5))
        self.log_s = nn.Parameter(torch.log(torch.tensor(init_s, dtype=torch.float32)))
    def forward(self, x):
        x = nn.functional.normalize(x, dim=1)
        W = nn.functional.normalize(self.W, dim=1)
        return (x @ W.t()) * torch.exp(self.log_s)

class CosineMarginHead(CosineHead):
    def __init__(self, in_dim: int, n_classes: int, init_s: float = 10.0, m: float = 0.2):
        super().__init__(in_dim, n_classes, init_s)
        self.m = m
    def forward(self, x, target=None):
        x = nn.functional.normalize(x, dim=1)
        W = nn.functional.normalize(self.W, dim=1)
        logits = x @ W.t()
        if target is not None:
            logits = logits.clone()
            logits[torch.arange(x.size(0), device=logits.device), target] -= self.m
        return logits * torch.exp(self.log_s)

# --- Extractors ---

# BLIP vision-only extractor
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModel, AutoProcessor, AutoImageProcessor
from transformers.utils.logging import set_verbosity_error
set_verbosity_error()

class BlipVisionExtractor(nn.Module):
    def __init__(self, model_name_or_path: str, device: str):
        super().__init__()
        self.processor = BlipProcessor.from_pretrained(model_name_or_path)
        m = BlipForConditionalGeneration.from_pretrained(model_name_or_path)
        m.eval()
        self.vision = m.vision_model.to(device)
        self.device = device
        self.out_dim = self.vision.config.hidden_size

    @torch.no_grad()
    def forward(self, pil_images: List[PILImage.Image]) -> torch.Tensor:
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel = inputs["pixel_values"].to(self.vision.device)
        out = self.vision(pixel_values=pixel)
        return out.last_hidden_state[:, 0, :]

# MedGemma / generic VLM extractor
class VLMVisionExtractor(nn.Module):
    def __init__(self, model_name_or_path: str, device: str, load_on_cpu: bool = True):
        super().__init__()
        map_arg = "cpu" if load_on_cpu else "auto"
        try:
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                device_map=map_arg,              # keep model on CPU by default
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        except torch.cuda.OutOfMemoryError:
            print("OOM with device_map=auto; retrying with CPU load…")
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                device_map="cpu",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        self.model.eval()

        self.vision = (
            getattr(self.model, "vision_tower", None)
            or getattr(self.model, "vision_model", None)
            or getattr(self.model, "image_encoder", None)
            or getattr(self.model, "vision_encoder", None)
        )
        if self.vision is None:
            raise RuntimeError("This checkpoint doesn't expose a vision tower (use a VLM variant).")

        # move only the vision tower to the requested device
        self.vision.to(device)
        self.vision_device = next(self.vision.parameters()).device
        self.dtype = next(self.vision.parameters()).dtype

        try:
            proc = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=True)
            self.image_processor = getattr(proc, "image_processor", None)
        except Exception:
            self.image_processor = None
        if self.image_processor is None:
            self.image_processor = AutoImageProcessor.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )

        cfg = getattr(self.vision, "config", None)
        self.out_dim = getattr(cfg, "hidden_size", -1)

    @torch.no_grad()
    def forward(self, pil_images: List[PILImage.Image]) -> torch.Tensor:
        ip_out = self.image_processor(images=pil_images, return_tensors="pt")
        pixel = ip_out["pixel_values"].to(self.vision_device, dtype=self.dtype)
        out = self.vision(pixel_values=pixel)
        last = getattr(out, "last_hidden_state", None)
        if last is None:
            if isinstance(out, (tuple, list)): last = out[0]
            elif isinstance(out, dict) and "last_hidden_state" in out: last = out["last_hidden_state"]
            else: raise RuntimeError("Could not find last_hidden_state in vision output.")
        return last[:, 0, :]


# No-backbone baseline extractor
class NoBackboneExtractor(nn.Module):
    def __init__(self, feature_mode: str = "random_proj", size: int = 96, proj_dim: int = 1024, seed: int = 42):
        super().__init__()
        assert feature_mode in ("random_proj", "color_stats")
        self.mode = feature_mode
        self.size = int(size)
        self.proj_dim = int(proj_dim)
        self.rng = np.random.RandomState(seed)
        self.register_buffer("proj", None)
        self.out_dim = (3 * 16 + 6) if self.mode == "color_stats" else self.proj_dim

    @torch.no_grad()
    def _to_numpy_rgb(self, pil_img: PILImage.Image):
        arr = np.asarray(pil_img, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr

    @torch.no_grad()
    def forward(self, pil_images: List[PILImage.Image]) -> torch.Tensor:
        if self.mode == "color_stats":
            feats = []
            for im in pil_images:
                arr = self._to_numpy_rgb(im)
                arr_f = arr.astype(np.float32) / 255.0
                means = arr_f.reshape(-1, 3).mean(axis=0)
                stds  = arr_f.reshape(-1, 3).std(axis=0) + 1e-8
                hists = []
                for c in range(3):
                    hist, _ = np.histogram(arr[:, :, c], bins=16, range=(0, 255), density=True)
                    hists.append(hist.astype(np.float32))
                feat = np.concatenate([means, stds] + hists, axis=0)  # 54-D
                feats.append(feat)
            return torch.from_numpy(np.stack(feats, axis=0)).float()

        # random projection
        ds = self.size
        flats = []
        for im in pil_images:
            im_small = im.resize((ds, ds), resample=PILImage.BICUBIC)
            arr = self._to_numpy_rgb(im_small).astype(np.float32) / 255.0
            arr = arr - 0.5
            flats.append(arr.reshape(-1))
        X = np.stack(flats, axis=0)  # (B, D_in)
        D_in = X.shape[1]
        if self.proj is None:
            scale = np.float32(1.0 / np.sqrt(np.float32(D_in)))
            W = (self.rng.randn(D_in, self.proj_dim).astype(np.float32)) * scale
            self.proj = torch.from_numpy(W)
        return torch.from_numpy(X.astype(np.float32)).matmul(self.proj.float())

class ResNetExtractor(nn.Module):
    def __init__(self, arch="resnet50", pretrained=True, device="cpu"):
        super().__init__()
        resnet = getattr(models, arch)(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()
        self.device = device
        self.out_dim = resnet.fc.in_features
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def forward(self, pil_images: List[PILImage.Image]) -> torch.Tensor:
        batch = torch.stack([self.transform(img) for img in pil_images]).to(self.device)
        feats = self.backbone(batch).squeeze(-1).squeeze(-1)
        return feats

# ---------------------------------
# DinoBloom extractor
# ---------------------------------
class DinoBloomExtractor(nn.Module):
    def __init__(self, ckpt_path: str, arch="base", device="cpu"):
        super().__init__()
        if arch == "small":
            self.model = vits.vit_small(patch_size=14); self.out_dim = 384
        elif arch == "base":
            self.model = vits.vit_base(patch_size=14); self.out_dim = 768
        elif arch == "large":
            self.model = vits.vit_large(patch_size=14); self.out_dim = 1024
        elif arch == "giant":
            self.model = vits.vit_giant2(patch_size=14); self.out_dim = 1536
        else:
            raise ValueError(f"Unknown DinoBloom arch: {arch}")

        state_dict = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().to(device)
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def forward(self, pil_images: List[PILImage.Image]) -> torch.Tensor:
        batch = torch.stack([self.transform(img) for img in pil_images]).to(self.device)
        out = self.model(batch)
        # handle dict output if necessary
        if isinstance(out, dict):
            out = out.get("x_norm_clstoken", list(out.values())[0])
        return out


# --- Utilities ---

def build_head_from_ckpt(ckpt: Dict) -> nn.Module:
    in_dim = int(ckpt["in_dim"])
    n_classes = int(ckpt["num_classes"])
    head_type = ckpt.get("head_type", "cosine")
    if head_type == "linear":
        head = LinearOrMLPHead(in_dim=in_dim, n_classes=n_classes, hidden=None)
    elif head_type == "mlp":
        head = LinearOrMLPHead(in_dim=in_dim, n_classes=n_classes, hidden=512, p=0.1)
    elif head_type == "cosine":
        head = CosineHead(in_dim=in_dim, n_classes=n_classes)
    elif head_type == "cosine_margin":
        head = CosineMarginHead(in_dim=in_dim, n_classes=n_classes, m=0.2)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return head

def apply_saved_norm(X: torch.Tensor, norm_meta: Dict) -> torch.Tensor:
    mode = norm_meta.get("feature_norm", "none")
    if mode in ("zscore", "zscore_l2"):
        mean = torch.tensor(norm_meta["zscore_mean"], dtype=X.dtype)
        std  = torch.tensor(norm_meta["zscore_std"], dtype=X.dtype).clamp_min(1e-6)
        X = (X - mean) / std
    if mode in ("l2", "zscore_l2"):
        X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return X

@torch.no_grad()
def extract_features(extractor: nn.Module, loader: DataLoader, device: str) -> Tuple[torch.Tensor, List[Dict]]:
    feats, metas_all = [], []
    for images, metas in loader:
        f = extractor(images)  # some extractors run on GPU internally
        feats.append(f.float().cpu())
        metas_all.extend(metas)
    X = torch.cat(feats, dim=0).contiguous()
    return X, metas_all

def discover_heads(head_dirs: List[str], tasks: List[str], backbones_filter: Optional[List[str]] = None):
    """
    Discover head files: head_<task>_<backbone>.pt
    Returns list of dicts with keys: task, backbone, head_path, eval_meta_path (optional)
    """
    found = []
    for d in head_dirs:
        dpath = Path(d)
        if not dpath.exists():
            continue
        for task in tasks:
            for head_file in dpath.glob(f"head_{task}_*.pt"):
                m = re.match(rf"head_{re.escape(task)}_(.+)\.pt$", head_file.name)
                if not m:
                    continue
                backbone = m.group(1)
                if backbones_filter and backbone not in backbones_filter:
                    continue
                # try to find matching eval meta (for baseline params)
                eval_meta = None
                meta_candidate = head_file.with_name(f"eval_{task}_{backbone}_meta.json")
                if meta_candidate.exists():
                    eval_meta = meta_candidate
                found.append({
                    "task": task,
                    "backbone": backbone,
                    "head_path": head_file,
                    "eval_meta_path": eval_meta,
                    "dir": dpath,
                })
    return found

def build_extractor_for_backbone(backbone: str, cfg: Dict, eval_meta_path: Optional[Path]) -> nn.Module:
    device = cfg["device"]
    if backbone == "blip_base":
        return BlipVisionExtractor(cfg["blip_base_path"], device)
    if backbone == "blip_ft":
        return BlipVisionExtractor(cfg["blip_ft_path"], device)
    if backbone == "medgemma_base":
        return VLMVisionExtractor(cfg["medgemma_base_path"], device, load_on_cpu=True)
    if backbone == "medgemma_ft":
        return VLMVisionExtractor(cfg["medgemma_ft_path"], device, load_on_cpu=True)
    if backbone == "resnet50":
        return ResNetExtractor("resnet50", pretrained=True, device=device)

    if backbone.startswith("dinobloom_"):
        arch = backbone.split("_")[1]  # e.g. dinobloom_b -> "b"
        ckpt_key = f"dinobloom_{arch}_path"
        ckpt_path = cfg.get(ckpt_key, f"./DinoBloom-{arch.upper()}.pth")
        return DinoBloomExtractor(ckpt_path, arch={
            "s": "small", "b": "base", "l": "large", "g": "giant"
        }[arch], device=device)


    # No-backbone baselines
    if backbone.startswith("nobackbone_"):
        # parse feature_mode from suffix if present
        feature_mode = "random_proj" if "random_proj" in backbone else ("color_stats" if "color_stats" in backbone else "random_proj")
        size = 96
        proj_dim = 1024
        # Prefer meta JSON if present
        if eval_meta_path is not None:
            try:
                with open(eval_meta_path, "r") as f:
                    meta = json.load(f)
                size = int(meta.get("downsample_size", size))
                proj_dim = int(meta.get("proj_dim", proj_dim if feature_mode == "random_proj" else 54))
                feature_mode = meta.get("feature_mode", feature_mode)
            except Exception:
                pass
        return NoBackboneExtractor(feature_mode=feature_mode, size=size, proj_dim=proj_dim, seed=42)

    raise ValueError(f"Unknown backbone: {backbone}")

def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)

# -------------------------
# Main
# -------------------------

def main(cfg: Dict):
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / cfg["out_csv"]

    # Load external dataset
    hf_test = load_from_disk(cfg["external_disk_dir"])
    ds = ExternalImageDataset(hf_test)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], collate_fn=collate_images_with_meta,
                        pin_memory=torch.cuda.is_available(), persistent_workers=False)

    # Discover all heads
    runs = discover_heads(cfg["head_dirs"], cfg["tasks"], cfg.get("backbones_filter"))
    if not runs:
        print("No heads found. Check head_dirs/tasks.")
        return
    print(f"Discovered {len(runs)} heads:")
    for r in runs:
        print(f"  - [{r['task']}] {r['backbone']}  ({r['head_path']})")

    all_rows = []

    for r in runs:
        task = r["task"]
        backbone = r["backbone"]
        head_path = r["head_path"]
        eval_meta_path = r["eval_meta_path"]

        # Load head checkpoint
        ckpt = torch.load(head_path, map_location="cpu")
        classes = ckpt.get("classes", None)
        if classes is None:
            raise RuntimeError(f"{head_path} missing 'classes' in checkpoint.")
        head = build_head_from_ckpt(ckpt)
        norm_meta = ckpt.get("norm_meta", {"feature_norm": "none"})
        in_dim = int(ckpt["in_dim"])
        num_classes = int(ckpt["num_classes"])

        # Build extractor
        extractor = build_extractor_for_backbone(backbone, cfg, eval_meta_path)
        # Sanity: if extractor out_dim unknown, infer via one image
        if getattr(extractor, "out_dim", -1) == -1:
            tmp_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_images_with_meta)
            imgs, _ = next(iter(tmp_loader))
            with torch.no_grad():
                emb = extractor(imgs)
            extractor.out_dim = emb.shape[-1]

        if extractor.out_dim != in_dim:
            raise RuntimeError(f"Dim mismatch for {head_path}: head in_dim={in_dim}, extractor out_dim={extractor.out_dim}")

        # Extract features on external set
        print(f"\n[{task}] {backbone}: extracting features on external set…")
        X, metas = extract_features(extractor, loader, cfg["device"])
        # Apply saved normalization
        Xn = apply_saved_norm(X, norm_meta)

        # Get logits & probabilities
        head.eval()
        with torch.no_grad():
            logits = head(Xn).cpu().numpy()
        probs = softmax_np(logits)
        pred_idx = probs.argmax(axis=1)
        pred_label = [classes[i] for i in pred_idx]

        # Build rows
        for i in range(len(metas)):
            row = {
                "task": task,
                "backbone": backbone,
                "image_filename": metas[i].get("image_filename"),
                "source": metas[i].get("source"),
                "true_description": metas[i].get("true_description"),
                "pred_idx": int(pred_idx[i]),
                "pred_label": pred_label[i],
            }
            # Add per-class probs
            for j, cname in enumerate(classes):
                row[f"prob::{cname}"] = float(probs[i, j])
            all_rows.append(row)

    # Write consolidated CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote external predictions: {out_csv} ({len(df)} rows)")

if __name__ == "__main__":
    main(CONFIG)
