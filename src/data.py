\
from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd

FEATURE_NAMES = [
  "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
  "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
  "root_shell","su_attempted","num_root","num_file_creations","num_shells",
  "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
  "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
  "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
  "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
  "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
  "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"
]
COLS = FEATURE_NAMES + ["label", "difficulty"]
CAT_COLS = ["protocol_type", "service", "flag"]

def _find_file(filename: str) -> Optional[str]:
    """
    Find dataset files either in Kaggle (/kaggle/input) or local (data/raw).
    Returns first match or None.
    """
    # Kaggle mount
    kaggle_root = "/kaggle/input"
    if os.path.isdir(kaggle_root):
        hits = glob.glob(f"{kaggle_root}/**/{filename}", recursive=True)
        if hits:
            return hits[0]

    # Local repo path
    local_root = Path("data/raw")
    if local_root.exists():
        candidate = local_root / filename
        if candidate.exists():
            return str(candidate)

        hits = glob.glob(str(local_root / f"**/{filename}"), recursive=True)
        if hits:
            return hits[0]

    return None

def locate_train_test() -> Tuple[str, str]:
    train_path = _find_file("KDDTrain+.txt")
    test_path = _find_file("KDDTest+.txt")

    if not train_path or not test_path:
        raise FileNotFoundError(
            "Could not locate KDDTrain+.txt and KDDTest+.txt.\n"
            "Local: put them under data/raw/\n"
            "Kaggle: add the NSL-KDD dataset as an Input."
        )

    return train_path, test_path

def load_nslkdd(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLS)
    df["label"] = df["label"].astype(str).str.strip().str.rstrip(".")
    return df

def to_binary(df: pd.DataFrame):
    """
    Binary label: 0=normal, 1=attack
    Drops 'difficulty' to avoid leakage / non-traffic feature.
    """
    y = (df["label"] != "normal").astype(int)
    X = df.drop(columns=["label", "difficulty"], errors="ignore")
    return X, y
