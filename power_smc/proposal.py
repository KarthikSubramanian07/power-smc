"""Prefix-only proposals and the incremental importance weight.

At each step a particle proposes its next token from a distribution that depends only
on the prefix seen so far (a "prefix-only" proposal). The paper proves (Theorem 1 /
Corollary 1) that among all prefix-only proposals the unique minimizer of the
incremental-weight variance is

    q*_t(v) = p_t(v)^alpha / sum_u p_t(u)^alpha = softmax(logits / tau),   tau = 1/alpha.

So the default proposal is the model distribution raised to alpha and renormalized,
i.e. temperature 1/alpha. The incremental weight (Eq. 8) is

    w_t = p_theta(y_t | x, y_<t)^alpha / q_t(y_t | x, y_<t).

With the optimal proposal this reduces to sum_u p_t(u)^alpha, which does not depend on
the sampled token, so its conditional variance given the prefix is exactly zero. That
identity is what :func:`conditional_weight_variance` checks.
"""

from __future__ import annotations

import numpy as np

from .utils import as_rng, log_softmax


class Proposal:
    """Base class: map model next-token log-probs to proposal log-probs."""

    def log_q(self, model_logprobs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


class TemperatureProposal(Proposal):
    """q ∝ p^(1/tau): temper the model distribution, then renormalize.

    ``tau = 1/alpha`` recovers the variance-optimal proposal q ∝ p^alpha.
    ``tau = 1`` samples straight from the model. ``tau -> 0`` approaches greedy.
    """

    def __init__(self, tau: float):
        if tau <= 0:
            raise ValueError("temperature tau must be > 0")
        self.tau = float(tau)

    def log_q(self, model_logprobs: np.ndarray) -> np.ndarray:
        return log_softmax(np.asarray(model_logprobs, dtype=np.float64) / self.tau, axis=-1)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"TemperatureProposal(tau={self.tau:.4g})"


def optimal_proposal(alpha: float) -> TemperatureProposal:
    """The variance-minimizing prefix-only proposal for a given alpha (tau = 1/alpha)."""
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    return TemperatureProposal(tau=1.0 / alpha)


def incremental_log_weight(model_logprobs: np.ndarray, proposal_logprobs: np.ndarray,
                           tokens: np.ndarray, alpha: float) -> np.ndarray:
    """log w_t = alpha * log p_t(y_t) - log q_t(y_t), evaluated per particle (Eq. 8)."""
    model_logprobs = np.asarray(model_logprobs, dtype=np.float64)
    proposal_logprobs = np.asarray(proposal_logprobs, dtype=np.float64)
    tokens = np.asarray(tokens, dtype=np.int64)
    rows = np.arange(tokens.shape[0])
    return alpha * model_logprobs[rows, tokens] - proposal_logprobs[rows, tokens]


def sample_tokens(proposal_logprobs: np.ndarray, rng) -> np.ndarray:
    """Draw one token per particle from the proposal via the Gumbel-max trick."""
    rng = as_rng(rng)
    logq = np.asarray(proposal_logprobs, dtype=np.float64)
    # Gumbel-max: argmax(logq + Gumbel noise) is an exact categorical draw and handles
    # -inf entries (forbidden tokens) cleanly.
    u = rng.random(logq.shape)
    gumbel = -np.log(-np.log(np.clip(u, 1e-300, 1.0)))
    return np.argmax(logq + gumbel, axis=-1).astype(np.int64)


def conditional_weight_variance(model_logprob_row: np.ndarray, proposal: Proposal,
                                alpha: float) -> float:
    """Exact Var_{v~q}[ p(v)^alpha / q(v) ] for a single prefix.

    Used to demonstrate Theorem 1: this is minimized (and equals 0) at tau = 1/alpha.
    Computed in closed form rather than by sampling:
        E_q[w]   = sum_v p(v)^alpha
        E_q[w^2] = sum_v p(v)^(2 alpha) / q(v)
        Var      = E_q[w^2] - E_q[w]^2
    """
    logp = np.asarray(model_logprob_row, dtype=np.float64)
    logq = proposal.log_q(logp[None, :])[0]
    support = np.isfinite(logp)
    logp = logp[support]
    logq = logq[support]

    log_w = alpha * logp - logq  # log of p^alpha / q per token
    log_mean = _logsumexp_1d(logp * alpha)  # log E_q[w] = log sum_v p^alpha
    # log E_q[w^2] = log sum_v q(v) * w(v)^2 = log sum_v exp(logq + 2 log_w)
    log_second = _logsumexp_1d(logq + 2.0 * log_w)
    second = np.exp(log_second)
    mean = np.exp(log_mean)
    return float(max(second - mean * mean, 0.0))


def _logsumexp_1d(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return -np.inf
    m = finite.max()
    return float(m + np.log(np.sum(np.exp(finite - m))))
