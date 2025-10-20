#!/usr/bin/env python3
"""
frozen_backbone_head.py

Train the SAME lightweight classifier head on top of frozen backbones:
- BLIP base
- BLIP finetuned
- MedGemma-4B VLM base
- ResNet (torchvision)
- DinoBloom (DINOv2 pretrained on WBC images)

Tasks:
- "leukemia_subtype" | "cell_type"

Assumptions:
- HuggingFace Datasets saved on disk with 'image' column and labels.
"""

import os, json, random, warnings
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm
from PIL import Image as PILImage

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from datasets import load_from_disk, ClassLabel
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoModel, AutoImageProcessor
)
from transformers.utils.logging import set_verbosity_error
set_verbosity_error()
warnings.filterwarnings("ignore")

import torchvision.models as models
from torchvision import transforms
import dinov2.models.vision_transformer as vits

# =========================
# config
# =========================
config = {
    # backbone: "blip_base" | "blip_ft" | "medgemma_base" | "resnet50" | "dinobloom_s|b|l|g"
    "backbone": "blip_ft",

    # task: "leukemia_subtype" | "cell_type"
    "task": "leukemia_subtype",

    # model paths
    "blip_base_path": "Salesforce/blip-image-captioning-base",
    "blip_ft_path": "./EINDRESULTAAT_BLIP1_fullyfinetuned",
    "medgemma_base_path": "google/medgemma-4b-it",
    "dinobloom_s_path": "./DinoBloom-S.pth",
    "dinobloom_b_path": "./DinoBloom-B.pth",
    "dinobloom_l_path": "./DinoBloom-L.pth",
    "dinobloom_g_path": "./DinoBloom-G.pth",

    # training
    "epochs": 20,
    "batch_size": 16,
    "num_workers": 4,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "seed": 42,

    # head / loss / normalization
    "head_type": "mlp",       # "linear" | "mlp" | "cosine" | "cosine_margin"
    "cosine_margin": 0.2,
    "label_smoothing": 0.0,
    "use_class_weights": True,
    "use_focal": False,
    "focal_gamma": 2.0,
    "early_stop_patience": 6,
    "feature_norm": "zscore_l2",

    # misc
    "out_dir": "./backbone_experiment",
    "cache_features": True,
    "refresh_cache": True,

    # HF datasets
    "train_disk_dir": "eindresultaat_train_data",
    "test_disk_dir": "eindresultaat_test_data",
    "val_frac": 0.15,
}

# =========================
# class lists
# =========================
LEUKEMIA_CLASSES = ["ALL", "AML", "CLL", "CML", "APML", "Healthy"]
CELLTYPE_CLASSES = [
    "Myeloblast", "Monoblast", "Lymphoblast",
    "Neutrophil", "Eosinophil", "Monocyte", "Lymphocyte", "Basophil",
    "Myelocyte", "Abnormal Promyelocyte", "Metamyelocyte", "Atypical Lymphocyte", "Promonocyte"
]
TASK_TO_CLASSES = {"leukemia_subtype": LEUKEMIA_CLASSES, "cell_type": CELLTYPE_CLASSES}

# =========================
# dataset + label canonicalization
# =========================
def get_datasets(task: str):
    hf_train = load_from_disk(config["train_disk_dir"])
    hf_test  = load_from_disk(config["test_disk_dir"])

    def _norm(s: str) -> str:
        return s.replace("-", " ").replace("_", " ").strip().lower()

    SUBTYPE_ALIASES = {
        "acute lymphoblastic leukemia": "ALL",
        "acute lymphocytic leukemia":  "ALL",
        "acute lymphoid leukemia":     "ALL",
        "all":                         "ALL",
        "acute myeloid leukemia":      "AML",
        "acute myelogenous leukemia":  "AML",
        "aml":                         "AML",
        "chronic lymphocytic leukemia":"CLL",
        "cll":                         "CLL",
        "chronic myeloid leukemia":    "CML",
        "cml":                         "CML",
        "acute promyelocytic leukemia":"APML",
        "acute promyelocytic leukaemia":"APML",
        "apl":                         "APML",
        "apml":                        "APML",
        "healthy":                     "Healthy",
        "normal":                      "Healthy",
        "control":                     "Healthy",
    }

    CELL_ALIASES = {
        "myeloblast": "Myeloblast",
        "monoblast": "Monoblast",
        "lymphoblast": "Lymphoblast",
        "neutrophil": "Neutrophil",
        "eosinophil": "Eosinophil",
        "monocyte": "Monocyte",
        "lymphocyte": "Lymphocyte",
        "basophil": "Basophil",
        "myelocyte": "Myelocyte",
        "promyelocyte": "Promyelocyte",
        "metamyelocyte": "Metamyelocyte",
        "atypical lymphocyte": "Atypical Lymphocyte",
        "promonocyte": "Promonocyte",
        "abnormal promyelocyte": "Abnormal Promyelocyte",
    }

    def canon_subtype(x: str) -> str:
        key = _norm(x).replace("leukaemia", "leukemia")
        return SUBTYPE_ALIASES.get(key, x.strip())

    def canon_cell(x: str) -> str:
        key = _norm(x)
        return CELL_ALIASES.get(key, x.strip())

    def _batched_norm(batch):
        batch["leukemia_subtype"] = [canon_subtype(v) for v in batch["leukemia_subtype"]]
        batch["cell_type"]        = [canon_cell(v)     for v in batch["cell_type"]]
        return batch

    hf_train = hf_train.map(_batched_norm, batched=True)
    hf_test  = hf_test.map(_batched_norm,  batched=True)

    hf_train = hf_train.filter(lambda ex: ex["leukemia_subtype"] in set(LEUKEMIA_CLASSES)
                                         and ex["cell_type"] in set(CELLTYPE_CLASSES))
    hf_test  = hf_test.filter(lambda ex: ex["leukemia_subtype"] in set(LEUKEMIA_CLASSES)
                                        and ex["cell_type"] in set(CELLTYPE_CLASSES))

    names = TASK_TO_CLASSES[task]
    name2idx = {n: i for i, n in enumerate(names)}
    label_feature = ClassLabel(names=names)

    hf_train = hf_train.add_column("stratify_label", [name2idx[v] for v in hf_train[task]])
    hf_train = hf_train.cast_column("stratify_label", label_feature)

    split = hf_train.train_test_split(
        test_size=config["val_frac"], seed=config["seed"], stratify_by_column="stratify_label"
    )
    split["train"] = split["train"].remove_columns(["stratify_label"])
    split["test"]  = split["test"].remove_columns(["stratify_label"])
    return split["train"], split["test"], hf_test

# =========================
# utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ColumnTargetDataset(Dataset):
    def __init__(self, base_ds, target_col: str, class_to_idx: Dict[str, int]):
        self.base = base_ds
        self.target_col = target_col
        self.class_to_idx = class_to_idx
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        item = self.base[idx]
        img = item.get("image", None)
        if isinstance(img, PILImage.Image): pil_img = img.convert("RGB")
        elif isinstance(img, dict) and "path" in img: pil_img = PILImage.open(img["path"]).convert("RGB")
        elif isinstance(img, str): pil_img = PILImage.open(img).convert("RGB")
        else: pil_img = PILImage.fromarray(np.array(img)).convert("RGB")
        label_val = item.get(self.target_col, None)
        label_idx = self.class_to_idx[label_val]
        return pil_img, label_idx

def collate_images(batch):
    imgs, labels = zip(*batch)
    return list(imgs), torch.tensor(labels, dtype=torch.long)

# =========================
# extractors
# =========================
class BlipVisionExtractor(nn.Module):
    def __init__(self, model_name_or_path: str, device: str):
        super().__init__()
        self.processor = BlipProcessor.from_pretrained(model_name_or_path)
        m = BlipForConditionalGeneration.from_pretrained(model_name_or_path)
        self.vision = m.vision_model.to(device).eval()
        self.out_dim = self.vision.config.hidden_size
    @torch.no_grad()
    def forward(self, pil_images: List[PILImage.Image]) -> torch.Tensor:
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel = inputs["pixel_values"].to(self.vision.device)
        out = self.vision(pixel_values=pixel)
        return out.last_hidden_state[:, 0, :]

class VLMVisionExtractor(nn.Module):
    def __init__(self, model_name_or_path: str, device: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True,
            device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).eval()
        self.vision = getattr(self.model, "vision_tower", None) or getattr(self.model, "vision_model", None)
        self.vision.to(device)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.out_dim = self.vision.config.hidden_size
    @torch.no_grad()
    def forward(self, pil_images: List[PILImage.Image]) -> torch.Tensor:
        ip_out = self.image_processor(images=pil_images, return_tensors="pt")
        pixel = ip_out["pixel_values"].to(next(self.vision.parameters()).device)
        out = self.vision(pixel_values=pixel)
        return out.last_hidden_state[:, 0, :]

class ResNetExtractor(nn.Module):
    def __init__(self, arch="resnet50", pretrained=True, device="cpu"):
        super().__init__()
        resnet = getattr(models, arch)(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()
        self.device = device
        self.out_dim = resnet.fc.in_features
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])
    @torch.no_grad()
    def forward(self, pil_images: List[PILImage.Image]) -> torch.Tensor:
        batch = torch.stack([self.transform(img) for img in pil_images]).to(self.device)
        feats = self.backbone(batch).squeeze(-1).squeeze(-1)
        return feats

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
        state_dict = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().to(device)
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])
    @torch.no_grad()
    def forward(self, pil_images: List[PILImage.Image]) -> torch.Tensor:
        batch = torch.stack([self.transform(img) for img in pil_images]).to(self.device)
        out = self.model(batch)
        return out

# =========================
# heads
# =========================
class LinearOrMLPHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: Optional[int] = None, p: float = 0.1):
        super().__init__()
        if hidden is None:
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, n_classes))
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(p),
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
        super().__init__(in_dim, n_classes, init_s); self.m = m
    def forward(self, x, target=None):
        x = nn.functional.normalize(x, dim=1); W = nn.functional.normalize(self.W, dim=1)
        logits = x @ W.t()
        if target is not None:
            logits = logits.clone()
            logits[torch.arange(x.size(0), device=logits.device), target] -= self.m
        return logits * torch.exp(self.log_s)

# =========================
# feature extraction helpers
# =========================
@torch.no_grad()
def extract_split_features(extractor: nn.Module, loader: DataLoader, desc: str = "extract"):
    feats, labels = [], []
    for images, y in tqdm(loader, desc=desc):
        f = extractor(images)
        feats.append(f.float().cpu())
        labels.append(y)
    X = torch.cat(feats, dim=0).contiguous()
    y = torch.cat(labels, dim=0).contiguous()
    return X, y

# =========================
# training / eval
# =========================
def make_loss(ytr_tensor: torch.Tensor, n_classes_model: int, device: str):
    weight = None
    if config["use_class_weights"]:
        y = ytr_tensor.cpu().numpy()
        uniq = np.unique(y)
        cw = compute_class_weight(class_weight="balanced", classes=uniq, y=y).astype(np.float32)
        weights_full = np.ones(n_classes_model, dtype=np.float32)
        weights_full[uniq] = cw
        weight = torch.tensor(weights_full, dtype=torch.float32, device=device)
    ls = float(config.get("label_smoothing", 0.0))
    base_ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=ls).to(device)
    if not config.get("use_focal", False): return base_ce
    gamma = float(config.get("focal_gamma", 2.0))
    def focal(logits, target):
        logp = nn.functional.log_softmax(logits, dim=1)
        ce = nn.functional.nll_loss(logp, target, weight=weight, reduction="none")
        with torch.no_grad(): p = torch.exp(-ce).clamp_min(1e-8)
        return ((1 - p) ** gamma * ce).mean()
    return focal

def apply_feature_normalization(Xtr, Xva, Xte):
    mode = config.get("feature_norm", "none")
    mean = std = None
    if mode in ("zscore", "zscore_l2"):
        mean = Xtr.mean(0); std = Xtr.std(0).clamp_min(1e-6)
        Xtr = (Xtr - mean) / std; Xva = (Xva - mean) / std; Xte = (Xte - mean) / std
    if mode in ("l2", "zscore_l2"):
        def l2(x): return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        Xtr = l2(Xtr); Xva = l2(Xva); Xte = l2(Xte)
    meta = {"feature_norm": mode}
    if mean is not None:
        meta["zscore_mean"] = mean.cpu().numpy().tolist()
        meta["zscore_std"]  = std.cpu().numpy().tolist()
    return Xtr, Xva, Xte, meta

def train_head(head: nn.Module, Xtr, ytr, Xva, yva, epochs=10, lr=5e-4, wd=1e-4,
               device="cpu", select_by="macroF1", n_classes_model: Optional[int]=None):
    if n_classes_model is None:
        if isinstance(head, (CosineHead, CosineMarginHead)): n_classes_model = head.W.size(0)
        else: n_classes_model = [m for m in head.modules() if isinstance(m, nn.Linear)][-1].out_features
    head.to(device)
    opt = optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.1)
    loss_fn = make_loss(ytr, n_classes_model=n_classes_model, device=device)
    best_metric, best_state, wait = -1.0, None, 0
    for ep in range(1, epochs+1):
        head.train()
        idx = torch.randperm(Xtr.size(0))
        bs = min(2048, Xtr.size(0))
        losses = []

        for i in tqdm(range(0, Xtr.size(0), bs), desc=f"epoch {ep:02d}"):
            j = idx[i:i+bs]
            xb = Xtr[j].to(device)
            yb = ytr[j].to(device)
            logits = head(xb, yb) if isinstance(head, CosineMarginHead) else head(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        head.eval()
        with torch.no_grad():
            v_logits = head(Xva.to(device))
            v_pred = v_logits.argmax(1).cpu().numpy()
            v_true = yva.cpu().numpy()
            acc = accuracy_score(v_true, v_pred)
            f1  = f1_score(v_true, v_pred, average="macro")

        print(f"val_acc {acc:.4f} | val_macroF1 {f1:.4f} | lr {sched.get_last_lr()[0]:.2e}")

        metric = f1 if select_by == "macroF1" else acc
        if metric > best_metric: best_metric, best_state, wait = metric, {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= int(config.get("early_stop_patience", 6)):
                print(f"early stop @ epoch {ep:02d} (best {select_by}={best_metric:.4f})"); break
        sched.step()
    if best_state is not None: head.load_state_dict(best_state)
    return head

@torch.no_grad()
def evaluate(head: nn.Module, X: torch.Tensor, y: torch.Tensor, device: str = "cpu"):
    head.eval().to(device); logits = head(X.to(device)); pred = logits.argmax(1).cpu().numpy(); true = y.cpu().numpy()
    acc  = accuracy_score(true, pred); f1   = f1_score(true, pred, average="macro")
    print(f"\nTEST accuracy: {acc:.4f} | TEST macro-F1: {f1:.4f}")
    return {"y_true": true, "y_pred": pred, "logits": logits.cpu().numpy(),
            "test_accuracy": float(acc), "test_macro_f1": float(f1)}

# =========================
# main
# =========================
def main(cfg: dict):
    set_seed(cfg["seed"]); os.makedirs(cfg["out_dir"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"; print("device:", device)

    class_names = TASK_TO_CLASSES[cfg["task"]]; n_classes = len(class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    base_train, base_val, base_test = get_datasets(cfg["task"])
    train_ds = ColumnTargetDataset(base_train, cfg["task"], class_to_idx)
    val_ds   = ColumnTargetDataset(base_val,   cfg["task"], class_to_idx)
    test_ds  = ColumnTargetDataset(base_test,  cfg["task"], class_to_idx)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], collate_fn=collate_images, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], collate_fn=collate_images, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], collate_fn=collate_images, pin_memory=pin)

    bb = cfg["backbone"]
    if bb == "blip_base": extractor = BlipVisionExtractor(cfg["blip_base_path"], device); cache_id = "blip_base"
    elif bb == "blip_ft": extractor = BlipVisionExtractor(cfg["blip_ft_path"], device); cache_id = "blip_ft"
    elif bb == "medgemma_base": extractor = VLMVisionExtractor(cfg["medgemma_base_path"], device); cache_id = "medgemma_base"
    elif bb == "resnet50": extractor = ResNetExtractor("resnet50", pretrained=True, device=device); cache_id = "resnet50"
    elif bb == "dinobloom_s": extractor = DinoBloomExtractor(cfg["dinobloom_s_path"], arch="small", device=device); cache_id = "dinobloom_s"
    elif bb == "dinobloom_b": extractor = DinoBloomExtractor(cfg["dinobloom_b_path"], arch="base", device=device); cache_id = "dinobloom_b"
    elif bb == "dinobloom_l": extractor = DinoBloomExtractor(cfg["dinobloom_l_path"], arch="large", device=device); cache_id = "dinobloom_l"
    elif bb == "dinobloom_g": extractor = DinoBloomExtractor(cfg["dinobloom_g_path"], arch="giant", device=device); cache_id = "dinobloom_g"
    else: raise ValueError("Unknown backbone")

    def cache_path(split: str):
        safe = str(cache_id).replace("/", "__").replace(":", "_")
        return os.path.join(cfg["out_dir"], f"feats_{cfg['task']}_{bb}_{safe}_{split}.pt")
    def get_feats(loader, split):
        path = cache_path(split)
        if cfg["cache_features"] and os.path.exists(path) and not cfg.get("refresh_cache", False):
            blob = torch.load(path, map_location="cpu"); X = blob["X"].float(); y = blob["y"].long()
            print(f"loaded cached {split} from {path}"); return X, y
        X, y = extract_split_features(extractor, loader, desc=f"{split} feats"); X = X.float()
        if cfg["cache_features"]: torch.save({"X": X, "y": y}, path); print(f"saved {split} to {path}")
        return X, y

    print("extracting / loading features…")
    Xtr, ytr = get_feats(train_loader, "train"); Xva, yva = get_feats(val_loader, "val"); Xte, yte = get_feats(test_loader, "test")

    Xtr, Xva, Xte, norm_meta = apply_feature_normalization(Xtr, Xva, Xte)

    if config["head_type"] == "linear": head = LinearOrMLPHead(in_dim=extractor.out_dim, n_classes=n_classes, hidden=None)
    elif config["head_type"] == "mlp": head = LinearOrMLPHead(in_dim=extractor.out_dim, n_classes=n_classes, hidden=512, p=0.1)
    elif config["head_type"] == "cosine": head = CosineHead(in_dim=extractor.out_dim, n_classes=n_classes)
    elif config["head_type"] == "cosine_margin": head = CosineMarginHead(in_dim=extractor.out_dim, n_classes=n_classes, m=float(config["cosine_margin"]))
    else: raise ValueError("unknown head_type")

    print("\ntraining head…")
    head = train_head(head, Xtr, ytr, Xva, yva,
                      epochs=cfg["epochs"], lr=cfg["lr"], wd=cfg["weight_decay"], device=device)

    print("\nTEST evaluation")
    eval_blob = evaluate(head, Xte, yte, device=device)

    eval_path_npz = os.path.join(cfg["out_dir"], f"eval_{cfg['task']}_{bb}.npz")
    np.savez_compressed(eval_path_npz, **eval_blob)
    meta = {"classes": class_names, "task": cfg["task"], "backbone": bb,
            "embed_dim": extractor.out_dim, "head_type": cfg["head_type"],
            "label_smoothing": cfg["label_smoothing"], "use_class_weights": cfg["use_class_weights"],
            "use_focal": cfg["use_focal"], "feature_norm": norm_meta.get("feature_norm")}
    if "zscore_mean" in norm_meta: meta["zscore_mean"] = norm_meta["zscore_mean"]; meta["zscore_std"]  = norm_meta["zscore_std"]
    with open(os.path.join(cfg["out_dir"], f"eval_{cfg['task']}_{bb}_meta.json"), "w") as f: json.dump(meta, f, indent=2)
    print(f"saved eval arrays to: {eval_path_npz}")

    save_path = os.path.join(cfg["out_dir"], f"head_{cfg['task']}_{bb}.pt")
    torch.save({"state_dict": head.cpu().state_dict(), "in_dim": extractor.out_dim,
                "num_classes": n_classes, "classes": class_names,
                "head_type": cfg["head_type"], "norm_meta": norm_meta}, save_path)
    print(f"\nsaved head to: {save_path}")

if __name__ == "__main__":
    main(config)
