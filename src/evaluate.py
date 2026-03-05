\
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, average_precision_score

from .data import locate_train_test, load_nslkdd, to_binary

REPORTS_DIR = Path("reports")
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = Path("models")

def _save_cm(y_true, y_pred, title, outpath: Path):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, values_format="d")
    plt.title(title)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

def _save_pr(y_true, scores, title, outpath: Path):
    p, r, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} (AP={ap:.3f})")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    _, test_path = locate_train_test()
    test_df = load_nslkdd(test_path)
    X_test, y_test = to_binary(test_df)

    for name, model_file in [("LR", "lr_model.joblib"), ("RF", "rf_model.joblib")]:
        model_path = MODELS_DIR / model_file
        if not model_path.exists():
            print(f"Missing {model_path}. Run: python -m src.train")
            continue

        model = joblib.load(model_path)
        pred = model.predict(X_test)
        scores = model.predict_proba(X_test)[:, 1]

        _save_cm(y_test, pred, f"{name} Confusion Matrix (KDDTest+)", FIG_DIR / f"cm_{name.lower()}.png")
        _save_pr(y_test, scores, f"{name} Precision-Recall (KDDTest+)", FIG_DIR / f"pr_{name.lower()}.png")

    print(f"Figures saved to {FIG_DIR}")

if __name__ == "__main__":
    main()
