import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import torch

# === Config ===
PRED_CSV = "captions_blip_debug.csv"  
OUT_CSV = "evaluation_results.csv"

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
print("🔬 Computing BERTScore (BioBERT)...")
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

# === Save with scores ===
df["bleu"] = bleu_scores
df["rouge1"] = rouge1_scores
df["rougeL"] = rougeL_scores
df["bertscore_f1"] = F1.tolist()
df.to_csv(OUT_CSV, index=False)

print(f"Scores saved to {OUT_CSV}")
