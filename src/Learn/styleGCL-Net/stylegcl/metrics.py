from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np


def binary_classification_metrics(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    threshold: float | None = None,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(y) == 0:
        return {}
    if threshold is None:
        threshold = choose_best_f1_threshold(y, s)
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / len(y)),
        "auc_roc": float(roc_auc_score(y, s)),
        "fp_at_recall_0.8": float(false_positive_rate_at_recall(y, s, 0.8)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def choose_best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(s)) == 1:
        return float(s[0])
    candidates = np.unique(s)
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        pred = (s >= threshold).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold


def roc_auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = 0.0
    total = len(pos) * len(neg)
    for p in pos:
        greater += float((p > neg).sum())
        greater += 0.5 * float((p == neg).sum())
    return greater / total


def false_positive_rate_at_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_recall: float,
) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    negatives = max(int((y == 0).sum()), 1)
    best_fp_rate = 1.0
    for threshold in np.unique(s):
        pred = (s >= threshold).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        recall = tp / max(tp + fn, 1)
        if recall >= target_recall:
            fp = ((pred == 1) & (y == 0)).sum()
            best_fp_rate = min(best_fp_rate, fp / negatives)
    return best_fp_rate


def normalized_mutual_info(labels_true: list[object] | np.ndarray, labels_pred: list[object] | np.ndarray) -> float:
    true = list(labels_true)
    pred = list(labels_pred)
    n = len(true)
    if n == 0:
        return 0.0
    ct = Counter(true)
    cp = Counter(pred)
    joint = Counter(zip(true, pred))
    mi = 0.0
    for (a, b), count in joint.items():
        pa = ct[a] / n
        pb = cp[b] / n
        pab = count / n
        mi += pab * math.log(max(pab / max(pa * pb, 1e-12), 1e-12))
    ht = -sum((count / n) * math.log(count / n) for count in ct.values())
    hp = -sum((count / n) * math.log(count / n) for count in cp.values())
    denom = math.sqrt(max(ht * hp, 1e-12))
    return float(mi / denom) if denom else 0.0


def adjusted_rand_index(labels_true: list[object] | np.ndarray, labels_pred: list[object] | np.ndarray) -> float:
    true = list(labels_true)
    pred = list(labels_pred)
    n = len(true)
    if n < 2:
        return 1.0
    joint = Counter(zip(true, pred))
    ct = Counter(true)
    cp = Counter(pred)

    def comb2(value: int) -> float:
        return value * (value - 1) / 2.0

    sum_joint = sum(comb2(v) for v in joint.values())
    sum_true = sum(comb2(v) for v in ct.values())
    sum_pred = sum(comb2(v) for v in cp.values())
    total = comb2(n)
    expected = sum_true * sum_pred / max(total, 1e-12)
    maximum = 0.5 * (sum_true + sum_pred)
    denom = maximum - expected
    if abs(denom) < 1e-12:
        return 0.0
    return float((sum_joint - expected) / denom)


def best_cluster_binary_mapping(y_true: list[int] | np.ndarray, clusters: list[int] | np.ndarray) -> np.ndarray:
    """Map clusters to binary labels by majority vote for evaluation only."""
    y = np.asarray(y_true, dtype=int)
    c = np.asarray(clusters, dtype=int)
    pred = np.zeros(len(y), dtype=int)
    for cluster in sorted(set(c.tolist())):
        idx = c == cluster
        pred[idx] = int(y[idx].mean() >= 0.5)
    return pred


def all_pair_agreement(labels: list[object] | np.ndarray) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for i, j in itertools.combinations(range(len(labels)), 2):
        if labels[i] == labels[j]:
            result.append((i, j))
    return result
