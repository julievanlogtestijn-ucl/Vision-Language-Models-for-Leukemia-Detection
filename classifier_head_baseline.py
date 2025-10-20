#!/usr/bin/env python3
"""
classifier_head_no_backbone.py

Baseline: train the SAME lightweight classifier head as in the VLM experiments,
but without any learned vision backbone.

We extract fixed, non-learned features from images and train only the head.
Two feature modes:
  - "random_proj": downsampled pixels -> fixed Gaussian random projection to proj_dim
  - "color_stats": per-channel mean/std + 16-bin hist per channel (54-D total)

tasks via config["task"]:
- "leukemia_subtype"  (ALL, AML, CLL, CML, APML, Healthy)
- "cell_type"         (Blast, Neutrophil, Eosinophil, Monocyte,
                       Lymphocyte, Basophil, Myelocyte, Promyelocyte, Metamyelocyte)
"""

import os, json, random, warnings
from typing import List, Dict, Optional, Tuple

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

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

config = {
    # No learned vision backbone in this baseline
    "backbone": "none",

    # pick one: "leukemia_subtype" | "cell_type"
    "task": "cell_type",

    # feature extractor (non-learned) for the baseline
    # "random_proj" | "color_stats"
    "feature_mode": "random_proj",
    "downsample_size": 96,     # used if feature_mode == "random_proj"
    "proj_dim": 1024,          # output dim for random projection features

    # training
    "epochs": 20,
    "batch_size": 32,
    "num_workers": 4,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "seed": 42,

    # head / loss / normalization
    "head_type": "cosine",            # "linear" | "mlp" | "cosine" | "cosine_margin"
    "cosine_margin": 0.2,             # used if head_type == "cosine_margin"
    "label_smoothing": 0.0,
    "use_class_weights": True,
    "use_focal": False,
    "focal_gamma": 2.0,
    "early_stop_patience": 6,
    "feature_norm": "zscore_l2",      # "none" | "zscore" | "l2" | "zscore_l2"

    # misc
    "out_dir": "./classifier_head_nobackbone_output",
    "cache_features": True,           # saves embeddings to disk for reuse
    "refresh_cache": True,           # force re-embed even if cache exists

    # HF datasets on disk
    "train_disk_dir": "eindresultaat_train_data",
    "test_disk_dir":  "eindresultaat_test_data",
    "val_frac": 0.15,
}

# =========================
# class lists
# =========================

LEUKEMIA_CLASSES = ["ALL", "AML", "CLL", "CML", "APML", "Healthy"]
#CELLTYPE_CLASSES = ["Blast", "Neutrophil", "Eosinophil", "Monocyte",
#                   "Lymphocyte", "Basophil", "Myelocyte", "Promyelocyte", "Metamyelocyte"]


CELLTYPE_CLASSES = [
    "Myeloblast", "Monoblast", "Lymphoblast",
    "Neutrophil", 
    "Eosinophil", "Monocyte", "Lymphocyte", "Basophil",
    "Myelocyte", "Abnormal Promyelocyte", "Metamyelocyte", "Atypical Lymphocyte", "Promonocyte"
]


TASK_TO_CLASSES = {
    "leukemia_subtype": LEUKEMIA_CLASSES,
    "cell_type": CELLTYPE_CLASSES,
}

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
        "neutrophil cell": "Neutrophil",
        "segmented neutrophil": "Neutrophil",

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

    # keep only recognized
    hf_train = hf_train.filter(lambda ex: ex["leukemia_subtype"] in set(LEUKEMIA_CLASSES)
                                         and ex["cell_type"] in set(CELLTYPE_CLASSES))
    hf_test  = hf_test.filter(lambda ex: ex["leukemia_subtype"] in set(LEUKEMIA_CLASSES)
                                        and ex["cell_type"] in set(CELLTYPE_CLASSES))

    names = TASK_TO_CLASSES[task]
    name2idx = {n: i for i, n in enumerate(names)}
    label_feature = ClassLabel(names=names)

    # helper column for stratified split
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

        if isinstance(img, PILImage.Image):
            pil_img = img.convert("RGB")
        elif isinstance(img, dict) and "path" in img:
            pil_img = PILImage.open(img["path"]).convert("RGB")
        elif isinstance(img, str):
            pil_img = PILImage.open(img).convert("RGB")
        else:
            pil_img = PILImage.fromarray(np.array(img)).convert("RGB")

        label_val = item.get(self.target_col, None)
        if label_val is None:
            raise KeyError(f"Missing target column '{self.target_col}'.")
        if label_val not in self.class_to_idx:
            raise KeyError(f"Label '{label_val}' not in allowed: {list(self.class_to_idx.keys())}")
        label_idx = self.class_to_idx[label_val]
        return pil_img, label_idx

def collate_images(batch):
    imgs, labels = zip(*batch)
    return list(imgs), torch.tensor(labels, dtype=torch.long)

# =========================
# non-learned feature extractors
# =========================

class NoBackboneExtractor(nn.Module):
    """
    Non-learned feature extractor.
      - "random_proj": resize -> flatten -> fixed Gaussian random projection
      - "color_stats": per-channel mean/std + 16-bin histogram per channel
    """
    def __init__(self, feature_mode: str = "random_proj", size: int = 96, proj_dim: int = 1024, seed: int = 42):
        super().__init__()
        assert feature_mode in ("random_proj", "color_stats")
        self.mode = feature_mode
        self.size = int(size)
        self.proj_dim = int(proj_dim)
        self.rng = np.random.RandomState(seed)
        self.register_buffer("proj", None)  # created on first forward if needed
        self.out_dim = (3 * 16 + 6) if self.mode == "color_stats" else self.proj_dim

    @torch.no_grad()
    def _to_numpy_rgb(self, pil_img: PILImage.Image):
        arr = np.asarray(pil_img, dtype=np.uint8)
        if arr.ndim == 2:  # grayscale
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr

    @torch.no_grad()
    def forward(self, pil_images: List[PILImage.Image]) -> torch.Tensor:
        if self.mode == "color_stats":
            feats = []
            for im in pil_images:
                arr = self._to_numpy_rgb(im)
                arr_f = arr.astype(np.float32) / 255.0
                # per-channel stats
                means = arr_f.reshape(-1, 3).mean(axis=0)         # (3,)
                stds  = arr_f.reshape(-1, 3).std(axis=0) + 1e-8   # (3,)
                # 16-bin hist per channel
                hists = []
                for c in range(3):
                    hist, _ = np.histogram(arr[:, :, c], bins=16, range=(0, 255), density=True)
                    hists.append(hist.astype(np.float32))
                feat = np.concatenate([means, stds] + hists, axis=0)  # (54,)
                feats.append(feat)
            X = torch.from_numpy(np.stack(feats, axis=0)).float()
            return X

        # random projection over downsampled pixels
        ds = self.size
        flats = []
        for im in pil_images:
            im_small = im.resize((ds, ds), resample=PILImage.BICUBIC)
            arr = self._to_numpy_rgb(im_small).astype(np.float32) / 255.0
            arr = arr - 0.5  # center
            flat = arr.reshape(-1)  # (ds*ds*3,)
            flats.append(flat)
        X = np.stack(flats, axis=0)  # (B, D_in)

        D_in = X.shape[1]
        if self.proj is None:
            # Keep everything in float32 to avoid promotion to float64
            scale = np.float32(1.0 / np.sqrt(np.float32(D_in)))
            W = (self.rng.randn(D_in, self.proj_dim).astype(np.float32)) * scale
            self.proj = torch.from_numpy(W)  # float32
        # ensure X and proj dtypes match (float32)
        Xp = torch.from_numpy(X.astype(np.float32)).matmul(self.proj.float())
        return Xp


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
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(p),
                nn.Linear(hidden, n_classes),
            )
    def forward(self, x): return self.net(x)

class CosineHead(nn.Module):
    """Angle-based classifier with learnable temperature."""
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
    """Cosine classifier with additive margin on target logits."""
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
    X = torch.cat(feats,  dim=0).contiguous()
    y = torch.cat(labels, dim=0).contiguous()
    return X, y

# =========================
# training / eval
# =========================

# --- make_loss: pad class weights to the model's output size ---
def make_loss(ytr_tensor: torch.Tensor, n_classes_model: int, device: str):
    weight = None
    if config["use_class_weights"]:
        y = ytr_tensor.cpu().numpy()
        uniq = np.unique(y)  # only classes present in the *train* split
        # weights for present classes
        cw = compute_class_weight(class_weight="balanced", classes=uniq, y=y).astype(np.float32)
        # pad to full length (classes absent in train get weight=1.0)
        weights_full = np.ones(n_classes_model, dtype=np.float32)
        weights_full[uniq] = cw
        weight = torch.tensor(weights_full, dtype=torch.float32, device=device)

    ls = float(config.get("label_smoothing", 0.0))
    base_ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=ls).to(device)

    if not config.get("use_focal", False):
        return base_ce

    gamma = float(config.get("focal_gamma", 2.0))
    def focal(logits, target):
        logp = nn.functional.log_softmax(logits, dim=1)
        ce = nn.functional.nll_loss(logp, target, weight=weight, reduction="none")
        with torch.no_grad():
            p = torch.exp(-ce).clamp_min(1e-8)
        return ((1 - p) ** gamma * ce).mean()
    return focal


def apply_feature_normalization(Xtr, Xva, Xte):
    mode = config.get("feature_norm", "none")
    mean = std = None

    if mode in ("zscore", "zscore_l2"):
        mean = Xtr.mean(0)
        std  = Xtr.std(0).clamp_min(1e-6)
        Xtr = (Xtr - mean) / std
        Xva = (Xva - mean) / std
        Xte = (Xte - mean) / std

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

    # infer n_classes from the head if not provided
    if n_classes_model is None:
        if isinstance(head, (CosineHead, CosineMarginHead)):
            n_classes_model = head.W.size(0)
        else:
            # grab the last Linear layer's out_features
            last_linear = [m for m in head.modules() if isinstance(m, nn.Linear)][-1]
            n_classes_model = last_linear.out_features

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
            xb = Xtr[j].to(device, non_blocking=True); yb = ytr[j].to(device, non_blocking=True)
            if isinstance(head, CosineMarginHead):
                logits = head(xb, yb)
            else:
                logits = head(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
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

        if metric > best_metric:
            best_metric = metric
            best_state  = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= int(config.get("early_stop_patience", 6)):
                print(f"early stop @ epoch {ep:02d} (best {select_by}={best_metric:.4f})")
                break
        sched.step()

    if best_state is not None:
        head.load_state_dict(best_state)
    return head

@torch.no_grad()
def evaluate(head: nn.Module, X: torch.Tensor, y: torch.Tensor, device: str = "cpu"):
    head.eval().to(device)
    logits = head(X.to(device))
    pred = logits.argmax(1).cpu().numpy()
    true = y.cpu().numpy()
    acc  = accuracy_score(true, pred)
    f1   = f1_score(true, pred, average="macro")
    print(f"\nTEST accuracy: {acc:.4f} | TEST macro-F1: {f1:.4f}")
    return {"y_true": true, "y_pred": pred, "logits": logits.cpu().numpy(),
            "test_accuracy": float(acc), "test_macro_f1": float(f1)}

# =========================
# main
# =========================

def main(cfg: dict):
    set_seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # classes
    class_names = TASK_TO_CLASSES[cfg["task"]]
    n_classes = len(class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # datasets
    base_train, base_val, base_test = get_datasets(cfg["task"])
    train_ds = ColumnTargetDataset(base_train, cfg["task"], class_to_idx)
    val_ds   = ColumnTargetDataset(base_val,   cfg["task"], class_to_idx)
    test_ds  = ColumnTargetDataset(base_test,  cfg["task"], class_to_idx)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], collate_fn=collate_images,
                              pin_memory=pin, persistent_workers=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], collate_fn=collate_images,
                              pin_memory=pin, persistent_workers=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], collate_fn=collate_images,
                              pin_memory=pin, persistent_workers=False)

    # non-learned extractor
    feature_mode = cfg.get("feature_mode", "random_proj")
    extractor = NoBackboneExtractor(
        feature_mode=feature_mode,
        size=int(cfg.get("downsample_size", 96)),
        proj_dim=int(cfg.get("proj_dim", 1024)),
        seed=int(cfg.get("seed", 42)),
    )
    print(f"Using NoBackboneExtractor(mode='{feature_mode}') with out_dim={extractor.out_dim}")

    # caching helpers
    def cache_path(split: str):
        safe = f"nobackbone_{feature_mode}_S{cfg.get('downsample_size',96)}_D{extractor.out_dim}"
        return os.path.join(cfg["out_dir"], f"feats_{cfg['task']}_{safe}_{split}.pt")

    def get_feats(loader, split):
        path = cache_path(split)
        if cfg["cache_features"] and os.path.exists(path) and not cfg.get("refresh_cache", False):
            blob = torch.load(path, map_location="cpu")
            X = blob["X"].float(); y = blob["y"].long()
            print(f"loaded cached {split} from {path}")
            return X, y
        X, y = extract_split_features(extractor, loader, desc=f"{split} feats")
        X = X.float()
        if cfg["cache_features"]:
            torch.save({"X": X, "y": y}, path)
            print(f"saved {split} to {path}")
        return X, y

    print("extracting / loading features…")
    Xtr, ytr = get_feats(train_loader, "train")
    Xva, yva = get_feats(val_loader,   "val")
    Xte, yte = get_feats(test_loader,  "test")

    # feature normalization
    Xtr, Xva, Xte, norm_meta = apply_feature_normalization(Xtr, Xva, Xte)

    # head
    n_classes = len(class_names)
    if config["head_type"] == "linear":
        head = LinearOrMLPHead(in_dim=extractor.out_dim, n_classes=n_classes, hidden=None)
    elif config["head_type"] == "mlp":
        head = LinearOrMLPHead(in_dim=extractor.out_dim, n_classes=n_classes, hidden=512, p=0.1)
    elif config["head_type"] == "cosine":
        head = CosineHead(in_dim=extractor.out_dim, n_classes=n_classes)
    elif config["head_type"] == "cosine_margin":
        head = CosineMarginHead(in_dim=extractor.out_dim, n_classes=n_classes, m=float(config["cosine_margin"]))
    else:
        raise ValueError("unknown head_type")

    print("\ntraining head…")
    head = train_head(head, Xtr, ytr, Xva, yva,
                      epochs=cfg["epochs"], lr=cfg["lr"], wd=cfg["weight_decay"], device=device)

    print("\nTEST evaluation")
    eval_blob = evaluate(head, Xte, yte, device=device)

    # save evaluation payload
    bb = "nobackbone_" + feature_mode
    eval_path_npz = os.path.join(cfg["out_dir"], f"eval_{cfg['task']}_{bb}.npz")
    np.savez_compressed(eval_path_npz, **eval_blob)
    meta = {
        "classes": class_names,
        "task": cfg["task"],
        "backbone": "none",
        "feature_mode": feature_mode,
        "embed_dim": extractor.out_dim,
        "head_type": cfg["head_type"],
        "label_smoothing": cfg["label_smoothing"],
        "use_class_weights": cfg["use_class_weights"],
        "use_focal": cfg["use_focal"],
        "feature_norm": norm_meta.get("feature_norm"),
        "downsample_size": cfg.get("downsample_size", 96),
        "proj_dim": cfg.get("proj_dim", None),
    }
    if "zscore_mean" in norm_meta:
        meta["zscore_mean"] = norm_meta["zscore_mean"]
        meta["zscore_std"]  = norm_meta["zscore_std"]

    with open(os.path.join(cfg["out_dir"], f"eval_{cfg['task']}_{bb}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved eval arrays to: {eval_path_npz}")

    # save head
    save_path = os.path.join(cfg["out_dir"], f"head_{cfg['task']}_{bb}.pt")
    torch.save({
        "state_dict": head.cpu().state_dict(),
        "in_dim": extractor.out_dim,
        "num_classes": n_classes,
        "classes": class_names,
        "head_type": cfg["head_type"],
        "norm_meta": norm_meta,
    }, save_path)
    print(f"\nsaved head to: {save_path}")

if __name__ == "__main__":
    main(config)
