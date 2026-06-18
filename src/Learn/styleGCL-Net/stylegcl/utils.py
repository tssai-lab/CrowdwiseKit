from __future__ import annotations

import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_csv_rows(path: str | os.PathLike[str]) -> list[dict[str, str]]:
    """Read CSV rows with common encodings used in English/Chinese datasets."""
    encodings = ("utf-8-sig", "utf-8", "gb18030")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"failed to decode {path!s}; last error: {last_error}",
    )


def write_csv_rows(
    path: str | os.PathLike[str],
    rows: Sequence[dict[str, object]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError("fieldnames are required when writing an empty CSV")
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: str | os.PathLike[str], data: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def stable_softmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    denom = float(np.sum(exps))
    if denom == 0.0 or not math.isfinite(denom):
        return np.full_like(values, 1.0 / len(values), dtype=float)
    return exps / denom


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), eps)
    return float(np.dot(a, b) / denom)


def batched(iterable: Sequence[object], batch_size: int) -> Iterable[Sequence[object]]:
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]
