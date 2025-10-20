import pandas as pd
import numpy as np
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import torch

# === Config ===
PRED_CSV = "FINAL_BLIP1_finetuned_descriptions.csv" 
OUT_CSV = "FINAL_BLIP1_finetuned_descriptions_results.csv"

# === Load predictions ===
df = pd.read_csv(PRED_CSV)
references = df["true_description"].tolist()
hypotheses = df["caption"].tolist()

print(f"Evaluating {len(hypotheses)} predictions...")

# === BLEU ===
smoothie = SmoothingFunction().method4
bleu_scores = [
    sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoothie)
    for ref, hyp in zip(references, hypotheses)
]
print(f"Mean BLEU: {np.mean(bleu_scores):.4f}")

# === ROUGE ===
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
rouge1_scores = []
rougeL_scores = []

for ref, hyp in zip(references, hypotheses):
    scores = scorer.score(ref, hyp)
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)

print(f"Mean ROUGE-1: {np.mean(rouge1_scores):.4f}")
print(f"Mean ROUGE-L: {np.mean(rougeL_scores):.4f}")

# === BERTScore with BioBERT ===
print("Computing BERTScore (BioBERT)...")
P, R, F1 = bert_score(
    cands=hypotheses,
    refs=references,
    lang="en",
    model_type="dmis-lab/biobert-base-cased-v1.1",
    num_layers=12,
    idf=False,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Mean BERTScore (F1): {F1.mean():.4f}")

# Store NLP metrics in dataframe
df["bleu"] = bleu_scores
df["rouge1"] = rouge1_scores
df["rougeL"] = rougeL_scores
df["bertscore_f1"] = F1.tolist()

# === Feature Extraction Function ===
def extract_features(text):
    features = {
        "disease": None,
        "cell_type": None,
        "size": None,
        "shape": None,
        "nucleus": None,
        "ratio": None,
        "chromatin": None,
        "cytoplasm": None,
        "granules": None
    }

    # Unified label mapping to abbreviations
    disease_abbrev_map = {
        "acute myeloid leukemia": "AML",
        "acute lymphoblastic leukemia": "ALL",
        "acute promyelocytic leukemia": "APL",
        "chronic myeloid leukemia": "CML",
        "chronic lymphocytic leukemia": "CLL",
        "aml": "AML",
        "all": "ALL",
        "apl": "APL",
        "apml": "APL",
        "cml": "CML",
        "cll": "CLL",
        "healthy": "HEALTHY",
        "healthy case": "HEALTHY",
        "healthy specimen": "HEALTHY",
        "healthy cell": "HEALTHY"
    }

    disease_pattern = (
        r"(acute myeloid leukemia|acute lymphoblastic leukemia|acute promyelocytic leukemia|"
        r"chronic myeloid leukemia|chronic lymphocytic leukemia|AML|ALL|APL|APML|CML|CLL|"
        r"healthy cell|healthy specimen|healthy case|\bhealthy\b|leukemia)"
    )
    cell_pattern = r"(blast|lymphocyte|monocyte|neutrophil|myelocyte|metamyelocyte|basophil|promyelocyte|eosinophil)"
    size_pattern = r"(small size|medium size|big size|large size)"
    shape_pattern = r"(irregular overall shape|round overall shape)"
    nucleus_pattern = r"(oval nucleus|round nucleus|bilobed nucleus|band nucleus|segmented nucleus|indented nucleus|multilobed nucleus|irregular nucleus|unsegmented\s*-\s*band nucleus)"
    ratio_pattern = r"(nuclear\s*-\s*to\s*-\s*cytoplasmic ratio|cytoplasmic\s*-\s*to\s*-\s*nuclear ratio)"
    chromatin_pattern = r"(fine chromatin|coarse chromatin|clumped chromatin|loosely chromatin|densely chromatin)"
    cytoplasm_pattern = r"(scanty cytoplasm|moderate cytoplasm|light blue cytoplasm|purple blue cytoplasm|vacuolated cytoplasm|clear cytoplasmic texture|frosted cytoplasmic texture|blue cytoplasm)"
    granules_pattern = r"(small pink granules|round red granules|coarse purple granules)"

    patterns = {
        "disease": disease_pattern,
        "cell_type": cell_pattern,
        "size": size_pattern,
        "shape": shape_pattern,
        "nucleus": nucleus_pattern,
        "ratio": ratio_pattern,
        "chromatin": chromatin_pattern,
        "cytoplasm": cytoplasm_pattern,
        "granules": granules_pattern
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(0).lower().strip()

            if key == "disease":
                value = disease_abbrev_map.get(value, value.upper())  # default to upper-case form
            features[key] = normalize_feature_value(key, value if key != "disease" else value.upper())

    return features

def normalize_feature_value(key, value):
    """
    Clean up formatting inconsistencies in extracted feature values.
    """
    if not isinstance(value, str):
        return value

    value = value.lower().strip()

    # Specific cleanup rules
    if key == "nucleus":
        # unify inconsistent hyphens/spaces
        value = value.replace(" - ", "-").replace(" – ", "-").replace("—", "-")
        if value == "unsegmented-band nucleus":
            value = "unsegmented-band nucleus"  # standard form

    return value


# === Extract Features ===
df["features_gt"] = df["true_description"].str.lower().apply(extract_features)
df["features_pred"] = df["caption"].str.lower().apply(extract_features)

# === Prepare for Detailed Evaluation ===
feature_columns = list(df["features_gt"].iloc[0].keys())
exact_match_flags = []
per_feature_correct = {key: 0 for key in feature_columns}
per_feature_total_given = {key: 0 for key in feature_columns}
per_feature_missed = {key: 0 for key in feature_columns}

#for capital letters
def normalize(value):
        return value.strip().lower() if isinstance(value, str) else value

# Per-feature columns
for key in feature_columns:
    df[f"{key}_gt"] = df["features_gt"].apply(lambda x: x[key])
    df[f"{key}_pred"] = df["features_pred"].apply(lambda x: x[key])
    df[f"{key}_correct"] = df.apply(
        lambda row: normalize(row[f"{key}_pred"]) == normalize(row[f"{key}_gt"]) if row[f"{key}_pred"] is not None else False,
        axis=1
    )

# Exact match flag
for i, row in df.iterrows():
    gt = row["features_gt"]
    pred = row["features_pred"]

    all_match = all(
        normalize(pred[k]) == normalize(gt[k])
        for k in feature_columns
    )
    exact_match_flags.append(all_match)

    for key in feature_columns:
        if pred[key] is not None:
            per_feature_total_given[key] += 1
            if normalize(pred[key]) == normalize(gt[key]):
                per_feature_correct[key] += 1
        else:
            per_feature_missed[key] += 1

df["exact_match"] = exact_match_flags
exact_match_count = sum(exact_match_flags)

# === Print Summary ===
print("\n🔍 Feature-Level Accuracy (only when predicted):")
for key in feature_columns:
    total = per_feature_total_given[key]
    missed = per_feature_missed[key]
    correct = per_feature_correct[key]
    acc = correct / total if total > 0 else 0.0
    print(f"- {key}: {acc:.2%} accuracy ({correct}/{total}), missed in {missed} samples")

print(f"\n✅ Exact Match Accuracy: {exact_match_count}/{len(df)} = {exact_match_count / len(df):.2%}")

# === Save everything ===
df.to_csv(OUT_CSV, index=False)
print(f"\n📁 Full results saved to {OUT_CSV}")
