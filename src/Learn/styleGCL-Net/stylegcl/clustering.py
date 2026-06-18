from __future__ import annotations

import math

import numpy as np

from .utils import l2_normalize, set_seed


def pairwise_cosine(x: np.ndarray) -> np.ndarray:
    z = l2_normalize(np.asarray(x, dtype=np.float64), axis=1)
    sim = z @ z.T
    return np.clip(sim, -1.0, 1.0)


def rbf_similarity(x: np.ndarray, sigma: float | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    diff = x[:, None, :] - x[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    if sigma is None:
        positive = dist2[dist2 > 1e-12]
        sigma = float(np.sqrt(np.median(positive))) if positive.size else 1.0
    gamma = 1.0 / max(2.0 * sigma * sigma, 1e-12)
    sim = np.exp(-gamma * dist2)
    np.fill_diagonal(sim, 1.0)
    return sim


def kmeans(
    x: np.ndarray,
    k: int,
    seed: int = 13,
    max_iter: int = 100,
    n_init: int = 10,
) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.asarray([], dtype=int)
    if k >= n:
        return np.arange(n, dtype=int)

    best_labels: np.ndarray | None = None
    best_inertia = float("inf")
    rng = set_seed(seed)
    for _ in range(n_init):
        centers = _kmeans_pp_init(x, k, rng)
        labels = np.zeros(n, dtype=int)
        for _iter in range(max_iter):
            distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(new_labels, labels) and _iter > 0:
                break
            labels = new_labels
            for cluster in range(k):
                members = x[labels == cluster]
                if len(members):
                    centers[cluster] = members.mean(axis=0)
                else:
                    centers[cluster] = x[rng.integers(0, n)]
        inertia = float(((x - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    return best_labels if best_labels is not None else np.zeros(n, dtype=int)


def _kmeans_pp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    centers = [x[rng.integers(0, n)].copy()]
    while len(centers) < k:
        current = np.vstack(centers)
        dist2 = ((x[:, None, :] - current[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        total = float(dist2.sum())
        if total <= 1e-12:
            centers.append(x[rng.integers(0, n)].copy())
            continue
        probs = dist2 / total
        centers.append(x[rng.choice(n, p=probs)].copy())
    return np.vstack(centers)


def spectral_cluster(
    embeddings: np.ndarray,
    n_clusters: int | None = None,
    min_clusters: int = 2,
    max_clusters: int = 8,
    seed: int = 13,
    affinity: str = "cosine",
) -> tuple[np.ndarray, dict[str, object]]:
    x = np.asarray(embeddings, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.asarray([], dtype=int), {"n_clusters": 0, "eigenvalues": []}
    if n == 1:
        return np.asarray([0], dtype=int), {"n_clusters": 1, "eigenvalues": [0.0]}

    if affinity == "rbf":
        sim = rbf_similarity(x)
    else:
        sim = (pairwise_cosine(x) + 1.0) / 2.0
    np.fill_diagonal(sim, 0.0)
    degrees = sim.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
    laplacian = np.eye(n) - (inv_sqrt[:, None] * sim * inv_sqrt[None, :])
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if n_clusters is None:
        upper = min(max_clusters, n - 1)
        lower = min(min_clusters, upper)
        if upper <= 1:
            n_clusters = 1
        else:
            gaps = np.diff(eigenvalues[: upper + 1])
            search = gaps[lower - 1 : upper]
            n_clusters = int(np.argmax(search) + lower) if search.size else lower
            n_clusters = max(2, min(n_clusters, upper))

    features = l2_normalize(eigenvectors[:, :n_clusters], axis=1)
    labels = kmeans(features, n_clusters, seed=seed)
    info = {
        "n_clusters": int(n_clusters),
        "eigenvalues": [float(v) for v in eigenvalues[: min(20, len(eigenvalues))]],
        "affinity": affinity,
    }
    return labels, info


def cluster_suspicion_scores(embeddings: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Score workers by cluster compactness and size.

    The paper uses clustering to expose collusive groups. In unsupervised mode
    there is no class name for a cluster, so this score marks dense, compact
    clusters as more suspicious and leaves final thresholding to the caller.
    """
    x = l2_normalize(np.asarray(embeddings, dtype=np.float64), axis=1)
    scores = np.zeros(len(x), dtype=np.float64)
    for cluster in sorted(set(int(c) for c in clusters)):
        idx = np.where(clusters == cluster)[0]
        if len(idx) == 0:
            continue
        centroid = l2_normalize(x[idx].mean(axis=0))
        cohesion = np.maximum(0.0, x[idx] @ centroid)
        size_bonus = math.log1p(len(idx)) / math.log1p(max(len(x), 2))
        scores[idx] = 0.75 * cohesion + 0.25 * size_bonus
    if len(scores) and scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    return scores
