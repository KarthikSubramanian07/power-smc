"""Small numpy helpers shared across the core sampler.

The core of Power-SMC (target, proposal, SMC loop) depends only on numpy so the
correctness tests run on any machine without torch or a GPU. Anything that touches
real models lives in ``hf_model`` and ``kv_cache``.
"""

from __future__ import annotations

import numpy as np


def logsumexp(x: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    """Numerically stable log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return log-probabilities from unnormalized logits."""
    logits = np.asarray(logits, dtype=np.float64)
    return logits - logsumexp(logits, axis=axis, keepdims=True)


def normalize_log_weights(log_w: np.ndarray) -> np.ndarray:
    """Turn unnormalized log-weights into a probability vector."""
    return np.exp(log_softmax(log_w, axis=-1))


def effective_sample_size(weights: np.ndarray) -> float:
    """ESS_t = 1 / sum_i W_i^2 for normalized weights W (paper Eq. 6)."""
    w = np.asarray(weights, dtype=np.float64)
    s = float(np.sum(w))
    if s <= 0:
        return 0.0
    w = w / s
    denom = float(np.sum(w * w))
    return 0.0 if denom == 0 else 1.0 / denom


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two distributions over the same support."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return 0.5 * float(np.sum(np.abs(p - q)))


def as_rng(seed) -> np.random.Generator:
    """Accept an int seed, an existing Generator, or None and return a Generator."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)
