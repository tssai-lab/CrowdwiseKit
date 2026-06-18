from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .clustering import cluster_suspicion_scores, pairwise_cosine, spectral_cluster
from .data import CrowdTextRecord
from .features import HashingSentenceEncoder, word_count
from .utils import cosine, l2_normalize, set_seed, stable_softmax


@dataclass
class StyleGCLConfig:
    embedding_dim: int = 768
    latent_dim: int = 64
    gat_layers: int = 2
    gat_heads: int = 4
    mask_prob: float = 0.1
    noise_std: float = 0.05
    temperature: float = 0.5
    content_weight: float = 1.0
    orthogonal_weight: float = 0.1
    consistency_weight: float = 0.5
    graph_weight: float = 0.65
    random_seed: int = 13
    min_clusters: int = 2
    max_clusters: int = 8
    n_clusters: int | None = None


@dataclass
class StyleGCLResult:
    worker_ids: list[str]
    embeddings: np.ndarray
    clusters: np.ndarray
    scores: np.ndarray
    style_vectors: np.ndarray
    content_vectors: np.ndarray
    losses: dict[str, float]
    cluster_info: dict[str, Any]
    task_contributions: list[dict[str, object]] = field(default_factory=list)


class StyleGCLDetector:
    """Dependency-light reproduction of StyleGCL-Net.

    This class keeps the paper's computational stages while avoiding a hard
    dependency on PyTorch/Sentence-BERT. It is intended for reproducibility,
    data-pipeline validation, and CPU smoke tests. The trainable neural variant
    can reuse the same data schema and clustering utilities.
    """

    def __init__(self, config: StyleGCLConfig | None = None):
        self.config = config or StyleGCLConfig()
        self.encoder = HashingSentenceEncoder(dim=self.config.embedding_dim)
        rng = set_seed(self.config.random_seed)
        # Fixed random projections approximate the paper's linear projection
        # heads in environments where neural training dependencies are absent.
        scale = 1.0 / np.sqrt(self.config.embedding_dim)
        self._content_proj = rng.normal(
            0.0,
            scale,
            size=(self.config.embedding_dim * 2, self.config.latent_dim),
        )
        self._style_proj = rng.normal(
            0.0,
            scale,
            size=(self.config.embedding_dim * 2, self.config.latent_dim),
        )
        self._decoder = rng.normal(
            0.0,
            1.0 / np.sqrt(self.config.latent_dim),
            size=(self.config.latent_dim, self.config.embedding_dim),
        )
        self._pool_query = rng.normal(0.0, 1.0, size=(self.config.latent_dim,))

    def fit_predict(self, records: list[CrowdTextRecord]) -> StyleGCLResult:
        if not records:
            raise ValueError("records cannot be empty")
        worker_ids = sorted({rec.worker_id for rec in records})
        worker_to_index = {worker: i for i, worker in enumerate(worker_ids)}

        task_texts = [rec.task_text or rec.answer_text for rec in records]
        answer_texts = [rec.answer_text for rec in records]
        task_embeddings = self.encoder.encode(task_texts)
        answer_embeddings = self.encoder.encode(answer_texts)
        content_vectors, style_vectors = self._disentangle(
            answer_embeddings,
            task_embeddings,
        )
        worker_initial = self._attention_pool(records, worker_ids, style_vectors)
        adjacency = self._worker_adjacency(records, worker_to_index)
        worker_embeddings = self._graph_attention(worker_initial, adjacency)
        worker_embeddings = self._contrastive_smooth(worker_embeddings)
        clusters, cluster_info = spectral_cluster(
            worker_embeddings,
            n_clusters=self.config.n_clusters,
            min_clusters=self.config.min_clusters,
            max_clusters=self.config.max_clusters,
            seed=self.config.random_seed,
        )
        scores = self._text_suspicion_scores(
            records=records,
            worker_ids=worker_ids,
            embeddings=worker_embeddings,
            clusters=clusters,
            style_vectors=style_vectors,
            adjacency=adjacency,
        )
        losses = self._diagnostic_losses(
            records=records,
            task_embeddings=task_embeddings,
            content_vectors=content_vectors,
            style_vectors=style_vectors,
            worker_ids=worker_ids,
            worker_embeddings=worker_embeddings,
        )
        contributions = self._task_contributions(
            records=records,
            style_vectors=style_vectors,
            worker_ids=worker_ids,
            worker_clusters=clusters,
        )
        return StyleGCLResult(
            worker_ids=worker_ids,
            embeddings=worker_embeddings,
            clusters=clusters,
            scores=scores,
            style_vectors=style_vectors,
            content_vectors=content_vectors,
            losses=losses,
            cluster_info=cluster_info,
            task_contributions=contributions,
        )

    def _disentangle(
        self,
        answer_embeddings: np.ndarray,
        task_embeddings: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        attended_content = []
        for ans, task in zip(answer_embeddings, task_embeddings):
            # Vector-level cross attention surrogate: keep answer dimensions that
            # respond to task semantics, then pass residual to the style branch.
            gate = 1.0 / (1.0 + np.exp(-4.0 * ans * task))
            content_signal = gate * ans + (1.0 - gate) * task
            style_signal = ans - gate * task
            content_input = np.concatenate([content_signal, task])
            style_input = np.concatenate([style_signal, ans - task])
            attended_content.append((content_input, style_input))

        content_inputs = np.vstack([item[0] for item in attended_content])
        style_inputs = np.vstack([item[1] for item in attended_content])
        content = l2_normalize(content_inputs @ self._content_proj, axis=1)
        raw_style = style_inputs @ self._style_proj
        leakage = np.sum(raw_style * content, axis=1, keepdims=True) * content
        style = l2_normalize(raw_style - leakage, axis=1)
        return content, style

    def _attention_pool(
        self,
        records: list[CrowdTextRecord],
        worker_ids: list[str],
        style_vectors: np.ndarray,
    ) -> np.ndarray:
        pooled = np.zeros((len(worker_ids), self.config.latent_dim), dtype=np.float64)
        for worker_idx, worker_id in enumerate(worker_ids):
            indices = [i for i, rec in enumerate(records) if rec.worker_id == worker_id]
            styles = style_vectors[indices]
            lengths = np.asarray([word_count(records[i].answer_text) for i in indices], dtype=float)
            length_weight = np.log1p(lengths) / max(np.log1p(lengths).max(), 1e-12)
            logits = styles @ self._pool_query + 0.35 * length_weight
            weights = stable_softmax(logits)
            pooled[worker_idx] = np.sum(styles * weights[:, None], axis=0)
        return l2_normalize(pooled, axis=1)

    def _worker_adjacency(
        self,
        records: list[CrowdTextRecord],
        worker_to_index: dict[str, int],
    ) -> np.ndarray:
        n = len(worker_to_index)
        task_workers: dict[str, set[int]] = {}
        for rec in records:
            task_workers.setdefault(rec.task_id, set()).add(worker_to_index[rec.worker_id])
        adjacency = np.zeros((n, n), dtype=np.float64)
        for workers in task_workers.values():
            worker_list = sorted(workers)
            for i, src in enumerate(worker_list):
                for dst in worker_list[i + 1 :]:
                    adjacency[src, dst] += 1.0
                    adjacency[dst, src] += 1.0
        if adjacency.max() > 0:
            adjacency = adjacency / adjacency.max()
        return adjacency

    def _graph_attention(self, embeddings: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        h = l2_normalize(embeddings, axis=1)
        for _layer in range(self.config.gat_layers):
            sim = (pairwise_cosine(h) + 1.0) / 2.0
            weighted = self.config.graph_weight * sim + (1.0 - self.config.graph_weight) * adjacency
            weighted = weighted * (adjacency > 0)
            np.fill_diagonal(weighted, 1.0)
            rows = []
            for i in range(len(h)):
                weights = stable_softmax(weighted[i])
                rows.append(weights @ h)
            h = l2_normalize(np.vstack(rows), axis=1)
        return h

    def _contrastive_smooth(self, embeddings: np.ndarray) -> np.ndarray:
        rng = set_seed(self.config.random_seed + 17)
        x = np.asarray(embeddings, dtype=np.float64)
        mask = (rng.random(x.shape) > self.config.mask_prob).astype(float)
        noise = rng.normal(0.0, self.config.noise_std, size=x.shape)
        view = l2_normalize(x * mask + noise, axis=1)
        # One closed-form denoising step: pull each vector toward its positive
        # augmented view while retaining most of the graph embedding.
        return l2_normalize(0.8 * x + 0.2 * view, axis=1)

    def _diagnostic_losses(
        self,
        records: list[CrowdTextRecord],
        task_embeddings: np.ndarray,
        content_vectors: np.ndarray,
        style_vectors: np.ndarray,
        worker_ids: list[str],
        worker_embeddings: np.ndarray,
    ) -> dict[str, float]:
        recon = l2_normalize(content_vectors @ self._decoder, axis=1)
        content_reconstruction = float(np.mean((recon - task_embeddings) ** 2))
        orthogonal = float(np.mean(np.sum(content_vectors * style_vectors, axis=1) ** 2))
        consistency_values = []
        for worker in worker_ids:
            idx = [i for i, rec in enumerate(records) if rec.worker_id == worker]
            if len(idx) > 1:
                styles = style_vectors[idx]
                center = styles.mean(axis=0)
                consistency_values.append(float(np.mean((styles - center) ** 2)))
        style_consistency = float(np.mean(consistency_values)) if consistency_values else 0.0
        contrastive = self._info_nce(worker_embeddings)
        total = (
            self.config.content_weight * content_reconstruction
            + self.config.orthogonal_weight * orthogonal
            + self.config.consistency_weight * style_consistency
            + contrastive
        )
        return {
            "content_reconstruction": content_reconstruction,
            "orthogonal": orthogonal,
            "style_consistency": style_consistency,
            "info_nce": contrastive,
            "total": float(total),
        }

    def _info_nce(self, embeddings: np.ndarray) -> float:
        if len(embeddings) < 2:
            return 0.0
        rng = set_seed(self.config.random_seed + 29)
        x = l2_normalize(embeddings, axis=1)
        view = l2_normalize(x + rng.normal(0.0, self.config.noise_std, size=x.shape), axis=1)
        sim = (x @ view.T) / max(self.config.temperature, 1e-12)
        sim = sim - sim.max(axis=1, keepdims=True)
        exp = np.exp(sim)
        positive = np.diag(exp)
        denom = exp.sum(axis=1)
        return float(-np.log(np.maximum(positive / np.maximum(denom, 1e-12), 1e-12)).mean())

    def _task_contributions(
        self,
        records: list[CrowdTextRecord],
        style_vectors: np.ndarray,
        worker_ids: list[str],
        worker_clusters: np.ndarray,
        top_k: int = 25,
    ) -> list[dict[str, object]]:
        worker_cluster = dict(zip(worker_ids, worker_clusters.tolist()))
        by_cluster: dict[int, list[int]] = {}
        for idx, rec in enumerate(records):
            by_cluster.setdefault(int(worker_cluster[rec.worker_id]), []).append(idx)

        rows: list[dict[str, object]] = []
        for cluster, indices in by_cluster.items():
            center = l2_normalize(style_vectors[indices].mean(axis=0))
            by_task: dict[str, list[int]] = {}
            for idx in indices:
                by_task.setdefault(records[idx].task_id, []).append(idx)
            for task_id, task_indices in by_task.items():
                contribution = float(
                    np.mean([cosine(style_vectors[i], center) for i in task_indices])
                )
                rows.append(
                    {
                        "cluster": cluster,
                        "task_id": task_id,
                        "contribution": contribution,
                        "num_answers": len(task_indices),
                    }
                )
        rows.sort(key=lambda row: float(row["contribution"]), reverse=True)
        return rows[:top_k]

    def _text_suspicion_scores(
        self,
        records: list[CrowdTextRecord],
        worker_ids: list[str],
        embeddings: np.ndarray,
        clusters: np.ndarray,
        style_vectors: np.ndarray,
        adjacency: np.ndarray,
    ) -> np.ndarray:
        base = cluster_suspicion_scores(embeddings, clusters)
        consistency = np.zeros(len(worker_ids), dtype=np.float64)
        graph_density = np.zeros(len(worker_ids), dtype=np.float64)
        x = l2_normalize(np.asarray(embeddings), axis=1)
        for worker_idx, worker in enumerate(worker_ids):
            rec_idx = [i for i, rec in enumerate(records) if rec.worker_id == worker]
            if rec_idx:
                local = l2_normalize(style_vectors[rec_idx], axis=1)
                center = l2_normalize(local.mean(axis=0))
                consistency[worker_idx] = float(np.mean(np.maximum(0.0, local @ center)))
            same_cluster = np.where(clusters == clusters[worker_idx])[0]
            same_cluster = same_cluster[same_cluster != worker_idx]
            if len(same_cluster):
                style_density = np.maximum(0.0, x[same_cluster] @ x[worker_idx]).mean()
                graph_density[worker_idx] = 0.5 * style_density + 0.5 * adjacency[worker_idx, same_cluster].mean()

        consistency = _minmax(consistency)
        graph_density = _minmax(graph_density)
        scores = 0.35 * base + 0.40 * consistency + 0.25 * graph_density
        return _minmax(scores)


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or values.max() <= values.min():
        return np.zeros_like(values)
    return (values - values.min()) / (values.max() - values.min())
