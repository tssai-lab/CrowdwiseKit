from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clustering import pairwise_cosine
from .data import CrowdTextRecord, LabelRecord
from .features import HashingSentenceEncoder, tokenize, word_count
from .label_correlation import LabelCorrelationDetector
from .utils import l2_normalize


@dataclass
class BaselineResult:
    worker_ids: list[str]
    scores: np.ndarray
    embeddings: np.ndarray
    name: str


def text_cosine_baseline(records: list[CrowdTextRecord]) -> BaselineResult:
    worker_ids, embeddings = _worker_mean_answer_embeddings(records)
    sim = pairwise_cosine(embeddings)
    np.fill_diagonal(sim, 0.0)
    scores = _minmax(np.sort(sim, axis=1)[:, -min(5, max(1, len(worker_ids) - 1)) :].mean(axis=1))
    return BaselineResult(worker_ids, scores, embeddings, "cosine")


def stylometric_baseline(records: list[CrowdTextRecord]) -> BaselineResult:
    worker_ids = sorted({rec.worker_id for rec in records})
    features = []
    for worker in worker_ids:
        worker_records = [rec for rec in records if rec.worker_id == worker]
        rows = [_stylometric_features(rec.answer_text) for rec in worker_records]
        features.append(np.mean(rows, axis=0))
    embeddings = l2_normalize(np.vstack(features), axis=1)
    sim = pairwise_cosine(embeddings)
    np.fill_diagonal(sim, 0.0)
    scores = _minmax(np.sort(sim, axis=1)[:, -min(5, max(1, len(worker_ids) - 1)) :].mean(axis=1))
    return BaselineResult(worker_ids, scores, embeddings, "stylometric")


def nkcd_baseline(records: list[CrowdTextRecord]) -> BaselineResult:
    worker_ids = sorted({rec.worker_id for rec in records})
    worker_to_idx = {worker: i for i, worker in enumerate(worker_ids)}
    task_workers: dict[str, set[int]] = {}
    for rec in records:
        task_workers.setdefault(rec.task_id, set()).add(worker_to_idx[rec.worker_id])
    adjacency = np.zeros((len(worker_ids), len(worker_ids)), dtype=np.float64)
    for workers in task_workers.values():
        workers = sorted(workers)
        for i, src in enumerate(workers):
            for dst in workers[i + 1 :]:
                adjacency[src, dst] += 1.0
                adjacency[dst, src] += 1.0
    core = _weighted_k_core(adjacency)
    degree = adjacency.sum(axis=1)
    scores = _minmax(0.7 * _minmax(core) + 0.3 * _minmax(degree))
    embeddings = np.column_stack([core, degree, adjacency.mean(axis=1)])
    return BaselineResult(worker_ids, scores, embeddings, "nkcd")


def label_correlation_baseline(records: list[LabelRecord]) -> BaselineResult:
    result = LabelCorrelationDetector().fit_predict(records)
    return BaselineResult(result.worker_ids, result.scores, result.embeddings, "label_correlation")


def _worker_mean_answer_embeddings(records: list[CrowdTextRecord]) -> tuple[list[str], np.ndarray]:
    encoder = HashingSentenceEncoder()
    worker_ids = sorted({rec.worker_id for rec in records})
    embeddings = []
    for worker in worker_ids:
        texts = [rec.answer_text for rec in records if rec.worker_id == worker]
        embeddings.append(encoder.encode(texts).mean(axis=0))
    return worker_ids, l2_normalize(np.vstack(embeddings), axis=1)


def _stylometric_features(text: str) -> np.ndarray:
    tokens = tokenize(text)
    words = [tok for tok in tokens if any(ch.isalnum() for ch in tok)]
    chars = max(len(text), 1)
    wc = max(len(words), 1)
    punctuation = ".,;:!?，。！？；："
    return np.asarray(
        [
            word_count(text),
            len(set(words)) / wc,
            sum(len(w) for w in words) / wc,
            sum(ch in punctuation for ch in text) / chars,
            text.count(",") / chars,
            text.count(";") / chars,
            text.count(":") / chars,
            sum(ch.isupper() for ch in text) / chars,
            sum(ch.isdigit() for ch in text) / chars,
            text.lower().count("therefore"),
            text.lower().count("however"),
            text.lower().count("because"),
            text.lower().count("overall"),
            text.lower().count("notably"),
        ],
        dtype=np.float64,
    )


def _weighted_k_core(adjacency: np.ndarray) -> np.ndarray:
    n = len(adjacency)
    remaining = set(range(n))
    strength = adjacency.sum(axis=1)
    core = np.zeros(n, dtype=np.float64)
    current_level = 0.0
    while remaining:
        node = min(remaining, key=lambda idx: strength[idx])
        current_level = max(current_level, float(strength[node]))
        core[node] = current_level
        remaining.remove(node)
        for neighbor in list(remaining):
            strength[neighbor] -= adjacency[neighbor, node]
    return core


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or values.max() <= values.min():
        return np.zeros_like(values)
    return (values - values.min()) / (values.max() - values.min())
