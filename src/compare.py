from __future__ import annotations
import json
from pathlib import Path

REPORTS_DIR = Path("reports")

FILES = [
    ("Logistic Regression", "metrics_val_lr.json",  "Val"),
    ("Logistic Regression", "metrics_test_lr.json", "Test"),
    ("Random Forest",       "metrics_val_rf.json",  "Val"),
    ("Random Forest",       "metrics_test_rf.json", "Test"),
    ("Linear SVM",          "metrics_val_svm.json", "Val"),
    ("Linear SVM",          "metrics_test_svm.json","Test"),
]

def read_metrics(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    acc = d.get("accuracy")
    pr_auc = d.get("pr_auc")
    atk = d["report"].get("1", {})
    return {
        "accuracy": acc,
        "precision_1": atk.get("precision"),
        "recall_1": atk.get("recall"),
        "f1_1": atk.get("f1-score"),
        "pr_auc": pr_auc,
    }

def fmt(x):
    if x is None:
        return "—"
    return f"{x:.4f}"

def main():
    rows = []
    for model, filename, split in FILES:
        p = REPORTS_DIR / filename
        if not p.exists():
            continue
        m = read_metrics(p)
        rows.append((model, split, m))

    # Print Markdown table
    print("| Model | Split | Accuracy | Precision (Attack=1) | Recall (Attack=1) | F1 (Attack=1) | PR-AUC |")
    print("|------|-------|----------|----------------------|-------------------|---------------|--------|")
    for model, split, m in rows:
        print(
            f"| {model} | {split} | {fmt(m['accuracy'])} | {fmt(m['precision_1'])} | {fmt(m['recall_1'])} | {fmt(m['f1_1'])} | {fmt(m['pr_auc'])} |"
        )

if __name__ == "__main__":
    main()
