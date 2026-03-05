\
from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data import locate_train_test, load_nslkdd, to_binary
from .features import build_pre_lr, build_pre_rf

REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")

def _write_metrics(path: Path, y_true, y_pred, scores=None, extra=None):
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "report": classification_report(y_true, y_pred, output_dict=True),
    }
    if scores is not None:
        try:
            m["pr_auc"] = float(average_precision_score(y_true, scores))
        except Exception:
            pass
    if extra:
        m.update(extra)
    path.write_text(json.dumps(m, indent=2), encoding="utf-8")

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_path, test_path = locate_train_test()
    train_df = load_nslkdd(train_path)
    test_df  = load_nslkdd(test_path)

    X_full, y_full = to_binary(train_df)
    X_test, y_test = to_binary(test_df)

    # Validation split from training
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, stratify=y_full, random_state=42
    )

    # --- Logistic Regression ---
    lr = LogisticRegression(
        max_iter=3000,
        solver="saga",
        C=3.0,
        class_weight="balanced",
        n_jobs=-1
    )
    lr_pipe = Pipeline([("pre", build_pre_lr(X_train)), ("clf", lr)])
    lr_pipe.fit(X_train, y_train)

    lr_val_pred = lr_pipe.predict(X_val)
    lr_val_scores = lr_pipe.predict_proba(X_val)[:, 1]
    _write_metrics(
        REPORTS_DIR / "metrics_val_lr.json",
        y_val, lr_val_pred, scores=lr_val_scores,
        extra={"model":"LogisticRegression", "split":"val"}
    )

    lr_test_pred = lr_pipe.predict(X_test)
    lr_test_scores = lr_pipe.predict_proba(X_test)[:, 1]
    _write_metrics(
        REPORTS_DIR / "metrics_test_lr.json",
        y_test, lr_test_pred, scores=lr_test_scores,
        extra={"model":"LogisticRegression", "split":"test"}
    )
    joblib.dump(lr_pipe, MODELS_DIR / "lr_model.joblib")

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        class_weight="balanced_subsample"
    )
    rf_pipe = Pipeline([("pre", build_pre_rf(X_train)), ("clf", rf)])
    rf_pipe.fit(X_train, y_train)

    rf_val_pred = rf_pipe.predict(X_val)
    # RF has predict_proba
    rf_val_scores = rf_pipe.predict_proba(X_val)[:, 1]
    _write_metrics(
        REPORTS_DIR / "metrics_val_rf.json",
        y_val, rf_val_pred, scores=rf_val_scores,
        extra={"model":"RandomForest", "split":"val"}
    )

    rf_test_pred = rf_pipe.predict(X_test)
    rf_test_scores = rf_pipe.predict_proba(X_test)[:, 1]
    _write_metrics(
        REPORTS_DIR / "metrics_test_rf.json",
        y_test, rf_test_pred, scores=rf_test_scores,
        extra={"model":"RandomForest", "split":"test"}
    )
    joblib.dump(rf_pipe, MODELS_DIR / "rf_model.joblib")

    print("Done. Metrics written to reports/*.json and models saved to models/*.joblib")

if __name__ == "__main__":
    main()
