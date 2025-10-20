#!/usr/bin/env python3
"""
Evaluate all saved runs across one or more directories and write CSV summaries
(+ consolidated per-sample predictions). Autodiscovers backbones (including baselines).

Works with files named:
  eval_<task>_<backbone>.npz
  eval_<task>_<backbone>_meta.json

Examples found in:
  ./classifier_head_output/eval_leukemia_subtype_blip_base.npz
  ./classifier_head_nobackbone_output/eval_leukemia_subtype_nobackbone_random_proj.npz
"""

import os, json, datetime, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix
)

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

# =========================
# CONFIG (edit this)
# =========================
config = {
    # Search these directories for eval_<task>_*.npz
    "pred_dirs": [
        "./classifier_head_output",
        "./classifier_head_nobackbone_output",   # <-- baseline(s)
    ],

    "out_dir": "./eval_summaries",

    # Limit to these tasks (filenames must start with eval_<task>_...)
    "tasks": ["leukemia_subtype", "cell_type"],

    # If empty or None, autodiscover all backbones present in the files.
    # Otherwise, include only these (e.g., ["blip_base","blip_ft","medgemma_base","medgemma_ft","nobackbone_random_proj"])
    "backbones_filter": ["blip_base","blip_ft","medgemma_base","nobackbone_random_proj"],

    # Optional pretty-name mapping for reporting
    "backbone_alias": {
        "blip_base": "BLIP base",
        "blip_ft": "BLIP finetuned",
        "medgemma_base": "MedGemma-4B base",
        #"medgemma_ft": "MedGemma-4B finetuned",
        "nobackbone_random_proj": "No-backbone (random proj)",
        #"nobackbone_color_stats": "No-backbone (color stats)",
    },

    # Artifacts (kept minimal; consolidated CSVs are the main output)
    "save_plots": False,
    "normalize_cm": True,
    "save_both_cm_plots": False,

    "save_predictions_csv": False,
    "save_confusion_csv": False,
    "save_per_class_csvs": False,

    # Consolidated exports
    "export_consolidated": True,
}

# Canonical class lists (fallback if *_meta.json missing classes)
LEUKEMIA_CLASSES = ["ALL", "AML", "CLL", "CML", "APML", "Healthy"]
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
# Helpers
# =========================

def _safe_pull_same_len(extras: Dict[str, np.ndarray], key: str, n: int):
    arr = extras.get(key, None)
    if isinstance(arr, np.ndarray) and len(arr) == n:
        return arr
    return None

def safe_load_meta(meta_path: Path, task: str) -> Dict:
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if "classes" not in meta:
                meta["classes"] = TASK_TO_CLASSES.get(task)
            return meta
        except Exception:
            pass
    return {"task": task, "classes": TASK_TO_CLASSES.get(task)}

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

def per_class_table(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> pd.DataFrame:
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    return pd.DataFrame({
        "class": class_names,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": sup.astype(int),
    })

def plot_confusion(cm: np.ndarray, classes: List[str], title: str, out_path: Path, normalize: bool = True):
    if normalize:
        cm = cm.astype("float")
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1e-12)
        cm = cm / row_sums

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes, yticklabels=classes,
        ylabel="True label", xlabel="Predicted label",
        title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# NEW: save raw counts & row-% confusion matrices as CSVs
def save_confusion_csvs(cm_counts: np.ndarray, classes: List[str], out_prefix: Path):
    # raw counts
    df_counts = pd.DataFrame(cm_counts, index=pd.Index(classes, name="True"),
                             columns=pd.Index(classes, name="Predicted"))
    df_counts.to_csv(out_prefix.with_name(out_prefix.name + "_counts.csv"))

    # row-normalized percentages
    row_sums = cm_counts.sum(axis=1, keepdims=True).clip(min=1e-12)
    cm_rowpct = (cm_counts / row_sums) * 100.0
    df_rowpct = pd.DataFrame(cm_rowpct, index=pd.Index(classes, name="True"),
                             columns=pd.Index(classes, name="Predicted"))
    df_rowpct.to_csv(out_prefix.with_name(out_prefix.name + "_rowpct.csv"))

# NEW: save top off-diagonal confusions
def save_top_confusions(cm_counts: np.ndarray, classes: List[str], out_csv: Path, top_k: int = 20):
    rows = []
    for i, tname in enumerate(classes):
        for j, pname in enumerate(classes):
            if i == j:  # skip diagonal
                continue
            cnt = int(cm_counts[i, j])
            if cnt > 0:
                rows.append({"true": tname, "pred": pname, "count": cnt})
    if not rows:
        pd.DataFrame(columns=["true","pred","count"]).to_csv(out_csv, index=False)
        return
    df = pd.DataFrame(rows).sort_values("count", ascending=False).head(top_k)
    df.to_csv(out_csv, index=False)

# NEW: export per-sample predictions CSV (with names and indices)
def export_predictions_csv(
    out_csv: Path,
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    class_names: List[str],
    extras: Optional[Dict[str, np.ndarray]] = None,
):
    extras = extras or {}
    n = len(y_true_idx)
    def safe_pull(key: str) -> Optional[np.ndarray]:
        arr = extras.get(key, None)
        return arr if (isinstance(arr, np.ndarray) and len(arr) == n) else None

    df = pd.DataFrame({
        "y_true_idx": y_true_idx.astype(int),
        "y_pred_idx": y_pred_idx.astype(int),
        "y_true": [class_names[i] for i in y_true_idx],
        "y_pred": [class_names[j] for j in y_pred_idx],
        "correct": (y_true_idx == y_pred_idx).astype(int),
    })
    # Attach any optional columns if present in the .npz (e.g., filenames/ids)
    for key in ["filenames", "ids", "image_ids", "image_paths", "uids"]:
        col = safe_pull(key)
        if col is not None:
            df[key] = col
    df.to_csv(out_csv, index=False)

# =========================
# Discovery
# =========================

def discover_runs(pred_dirs: List[str], tasks: List[str], backbones_filter: Optional[List[str]]):
    """
    Return list of (task, backbone, npz_path, meta_path, dir_path).
    Autodiscovers by filename pattern eval_<task>_<backbone>.npz
    """
    discovered = []
    for d in pred_dirs:
        dpath = Path(d)
        if not dpath.exists():
            continue
        for task in tasks:
            for npz in dpath.glob(f"eval_{task}_*.npz"):
                # Extract backbone suffix after eval_<task>_
                m = re.match(rf"eval_{re.escape(task)}_(.+)\.npz$", npz.name)
                if not m:
                    continue
                backbone = m.group(1)
                if backbones_filter and backbone not in backbones_filter:
                    continue
                meta = npz.with_name(f"eval_{task}_{backbone}_meta.json")
                discovered.append((task, backbone, npz, meta, dpath))
    return discovered

# =========================
# Main
# =========================

def main(cfg: Dict):
    out_dir  = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(
        pred_dirs=cfg["pred_dirs"],
        tasks=cfg["tasks"],
        backbones_filter=cfg.get("backbones_filter") or None
    )
    if not runs:
        print("No runs found. Check pred_dirs/tasks.")
        return

    print(f"Discovered {len(runs)} runs:")
    for task, bb, npz, _, d in runs:
        print(f"  - [{task}] {bb}  ({npz})")

    summary_rows = []
    per_sample_rows = []

    for task, backbone, npz_path, meta_path, dir_path in runs:
        blob = np.load(npz_path, allow_pickle=True)
        y_true = blob["y_true"]
        y_pred = blob["y_pred"]
        n = int(y_true.shape[0])

        # Meta + classes
        meta = safe_load_meta(meta_path, task)
        class_names = meta.get("classes") or TASK_TO_CLASSES.get(task)
        if class_names is None:
            ncls = int(max(y_true.max(), y_pred.max()) + 1)
            class_names = [f"class_{i}" for i in range(ncls)]
        n_classes = len(class_names)

        if y_true.max() >= n_classes or y_pred.max() >= n_classes:
            raise ValueError(f"Label out of range for {task}/{backbone}: "
                             f"max(y_true)={y_true.max()}, max(y_pred)={y_pred.max()}, n_classes={n_classes}")

        # Metrics
        computed = compute_metrics(y_true, y_pred)
        acc = float(blob["test_accuracy"]) if "test_accuracy" in blob else computed["accuracy"]
        macro_f1 = float(blob["test_macro_f1"]) if "test_macro_f1" in blob else computed["macro_f1"]

        mtime = datetime.datetime.fromtimestamp(npz_path.stat().st_mtime).isoformat(timespec="seconds")
        summary_rows.append({
            "task": task,
            "backbone": cfg.get("backbone_alias", {}).get(backbone, backbone),
            "backbone_raw": backbone,
            "n_samples": n,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "micro_f1": computed["micro_f1"],
            "weighted_f1": computed["weighted_f1"],
            "file": str(npz_path),
            "dir": str(dir_path),
            "modified": mtime,
        })

        # ------- collect per-sample long-form rows
        extras = {k: blob[k] for k in blob.files if k not in {"y_true", "y_pred"}}
        opt_cols = {}
        for key in ["filenames", "ids", "image_ids", "image_paths", "uids"]:
            arr = _safe_pull_same_len(extras, key, n)
            if arr is not None:
                opt_cols[key] = arr

        for i in range(n):
            row = {
                "task": task,
                "backbone": cfg.get("backbone_alias", {}).get(backbone, backbone),
                "backbone_raw": backbone,
                "y_true_idx": int(y_true[i]),
                "y_pred_idx": int(y_pred[i]),
                "y_true": class_names[int(y_true[i])],
                "y_pred": class_names[int(y_pred[i])],
                "correct": int(y_true[i] == y_pred[i]),
                "file": str(npz_path),
            }
            for key, arr in opt_cols.items():
                val = arr[i]
                row[key] = val if isinstance(val, (str, int, float)) else str(val)
            per_sample_rows.append(row)

        # Optional per-run artifacts (left disabled by default)
        if cfg.get("save_plots") or cfg.get("save_confusion_csv") or cfg.get("save_per_class_csvs"):
            cm_counts = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
            tag_dir = out_dir / "artifacts" / task / backbone
            tag_dir.mkdir(parents=True, exist_ok=True)

            if cfg.get("save_plots"):
                plot_confusion(
                    cm_counts.copy(), class_names,
                    title=f"{task} — {backbone}",
                    out_path=tag_dir / "confusion_counts.png",
                    normalize=False
                )
                if cfg.get("save_both_cm_plots") or cfg.get("normalize_cm"):
                    plot_confusion(
                        cm_counts.copy(), class_names,
                        title=f"{task} — {backbone} (row %)",
                        out_path=tag_dir / "confusion_rowpct.png",
                        normalize=True
                    )

            if cfg.get("save_confusion_csv"):
                save_confusion_csvs(cm_counts, class_names, tag_dir / "confusion")

            if cfg.get("save_per_class_csvs"):
                pct = per_class_table(y_true, y_pred, class_names)
                pct.to_csv(tag_dir / "per_class_metrics.csv", index=False)
                save_top_confusions(cm_counts, class_names, tag_dir / "top_confusions.csv")

    # ------- Write consolidated CSVs
    if per_sample_rows and cfg.get("export_consolidated", True):
        preds_df = pd.DataFrame(per_sample_rows)
        preds_csv = out_dir / "predictions_all.csv"
        preds_df.to_csv(preds_csv, index=False)
        print(f"Wrote: {preds_csv}")
    else:
        print("No per-sample predictions found or export disabled.")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(["task", "macro_f1"], ascending=[True, False])
        summary_csv = out_dir / "run_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print("\n=== Overall results (sorted by task, macro-F1) ===")
        print(summary_df[["task","backbone","n_samples","accuracy","macro_f1"]].to_string(index=False))
        print(f"\nWrote: {summary_csv}")
    else:
        print("No results found to summarize.")

if __name__ == "__main__":
    main(config)
