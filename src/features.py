\
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from .data import CAT_COLS

def build_pre_lr(X: pd.DataFrame) -> ColumnTransformer:
    """
    Logistic Regression preprocessor:
      - OneHot for categoricals (handle_unknown=ignore)
      - Standardize numeric (with_mean=False due to sparse output)
    """
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", StandardScaler(with_mean=False), num_cols),
        ],
        remainder="drop"
    )

def build_pre_rf(X: pd.DataFrame) -> ColumnTransformer:
    """
    Random Forest preprocessor:
      - Ordinal encode categoricals; unknown -> -1
      - Pass numeric through unscaled
    """
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    return ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop"
    )
