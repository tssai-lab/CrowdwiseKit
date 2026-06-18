from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .clustering import cluster_suspicion_scores, spectral_cluster
from .data import LabelRecord


@dataclass
class LabelCorrelationConfig:
    min_overlap: int = 3
    n_clusters: int | None = None
    min_clusters: int = 2
    max_clusters: int = 8
    random_seed: int = 13


@dataclass
class LabelCorrelationResult:
    worker_ids: list[str]
    task_ids: list[str]
    matrix: np.ndarray
    correlation: np.ndarray
    embeddings: np.ndarray
    clusters: np.ndarray
    scores: np.ndarray
    cluster_info: dict[str, Any]


class LabelCorrelationDetector:
    """Complementary path for pure categorical-label crowdsourcing tasks."""

    def __init__(self, config: LabelCorrelationConfig | None = None):
        self.config = config or LabelCorrelationConfig()

    def fit_predict(self, records: list[LabelRecord]) -> LabelCorrelationResult:
        if not records:
            raise ValueError("records cannot be empty")
        worker_ids = sorted({rec.worker_id for rec in records})
        task_ids = sorted({rec.task_id for rec in records})
        worker_to_idx = {worker: i for i, worker in enumerate(worker_ids)}
        task_to_idx = {task: i for i, task in enumerate(task_ids)}
        label_to_id = {label: i for i, label in enumerate(sorted({rec.label for rec in records}))}

        matrix = np.full((len(worker_ids), len(task_ids)), np.nan, dtype=np.float64)
        for rec in records:
            matrix[worker_to_idx[rec.worker_id], task_to_idx[rec.task_id]] = label_to_id[rec.label]

        corr = self._pairwise_agreement_correlation(matrix)
        embeddings = self._spectral_embedding_from_correlation(corr)
        clusters, cluster_info = spectral_cluster(
            embeddings,
            n_clusters=self.config.n_clusters,
            min_clusters=self.config.min_clusters,
            max_clusters=self.config.max_clusters,
            seed=self.config.random_seed,
            affinity="cosine",
        )
        scores = self._label_suspicion_scores(corr, clusters, embeddings)
        cluster_info["path"] = "label_correlation"
        return LabelCorrelationResult(
            worker_ids=worker_ids,
            task_ids=task_ids,
            matrix=matrix,
            correlation=corr,
            embeddings=embeddings,
            clusters=clusters,
            scores=scores,
            cluster_info=cluster_info,
        )

    def _pairwise_agreement_correlation(self, matrix: np.ndarray) -> np.ndarray:
        n = matrix.shape[0]
        corr = np.eye(n, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                both = ~np.isnan(matrix[i]) & ~np.isnan(matrix[j])
                overlap = int(both.sum())
                if overlap < self.config.min_overlap:
                    value = 0.0
                else:
                    xi = matrix[i, both]
                    xj = matrix[j, both]
                    if np.std(xi) < 1e-12 or np.std(xj) < 1e-12:
                        value = float(np.mean(xi == xj))
                    else:
                        value = float(np.corrcoef(xi, xj)[0, 1])
                        if not np.isfinite(value):
                            value = 0.0
                corr[i, j] = corr[j, i] = max(value, 0.0)
        return corr

    def _spectral_embedding_from_correlation(self, corr: np.ndarray) -> np.ndarray:
        sim = np.asarray(corr, dtype=np.float64).copy()
        np.fill_diagonal(sim, 0.0)
        degrees = sim.sum(axis=1)
        inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
        laplacian = np.eye(len(sim)) - inv_sqrt[:, None] * sim * inv_sqrt[None, :]
        values, vectors = np.linalg.eigh(laplacian)
        order = np.argsort(values)
        dims = min(max(self.config.max_clusters, 2), len(sim))
        return vectors[:, order[:dims]]

    def _label_suspicion_scores(
        self,
        corr: np.ndarray,
        clusters: np.ndarray,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        base = cluster_suspicion_scores(embeddings, clusters)
        cluster_corr = np.zeros(len(corr), dtype=np.float64)
        row_corr = np.zeros(len(corr), dtype=np.float64)
        for i in range(len(corr)):
            same = np.where(clusters == clusters[i])[0]
            same = same[same != i]
            if len(same):
                cluster_corr[i] = float(np.mean(corr[i, same]))
            others = np.delete(corr[i], i)
            if len(others):
                top = np.sort(others)[-min(5, len(others)) :]
                row_corr[i] = float(np.mean(top))
        return _minmax(0.45 * base + 0.35 * _minmax(cluster_corr) + 0.20 * _minmax(row_corr))


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or values.max() <= values.min():
        return np.zeros_like(values)
    return (values - values.min()) / (values.max() - values.min())
