"""Algorithm 1: Power-SMC, a batch-parallel sampler for the power distribution.

N particles are decoded together in a single batched pass. At every step each particle
samples a token from a prefix-only proposal and multiplies its weight by the incremental
correction ``p_theta(y_t)^alpha / q_t(y_t)`` (Eq. 8). When the effective sample size
drops below ``kappa * N`` the particles are resampled (systematic resampling), the model
state / KV cache is reindexed by ancestor, and the weights are reset. EOS is an absorbing
state. At the end an output is drawn from the final weighted particle set.

The loop is model-agnostic: it talks to any object implementing the :class:`SMCModel`
protocol. The toy model in :mod:`power_smc.power_target` and the Transformer adapter in
:mod:`power_smc.hf_model` both satisfy it, so the identical code path validates on a toy
and runs on a real model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from .proposal import (
    Proposal,
    TemperatureProposal,
    incremental_log_weight,
    optimal_proposal,
    sample_tokens,
)
from .utils import as_rng, effective_sample_size


@runtime_checkable
class SMCModel(Protocol):
    """Minimal interface the SMC loop needs from a model.

    ``prefill`` processes the prompt and returns the next-token log-probs for the first
    generated position. ``decode`` feeds one chosen token per particle and returns the
    next-token log-probs. ``reorder`` reindexes the internal state (e.g. KV cache) by
    ancestor index during resampling.
    """

    eos_id: int
    vocab_size: int

    def prefill(self, prompt_ids: Any, n: int) -> tuple[Any, np.ndarray]:
        ...

    def decode(self, state: Any, tokens: np.ndarray) -> tuple[Any, np.ndarray]:
        ...

    def reorder(self, state: Any, ancestors: np.ndarray) -> Any:
        ...


class ConstantSchedule:
    """A trivial exponent schedule: alpha is constant for every step."""

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.ramping = False

    def exponent(self, step: int) -> float:
        return self.alpha


def systematic_resample(weights: np.ndarray, rng) -> np.ndarray:
    """Systematic resampling: return ancestor indices A_1..A_N.

    Draw u0 ~ U(0,1), set positions p_i = (u0 + i - 1)/N, and take
    A_i = min{ j : sum_{k<=j} w_k >= p_i }. Lower variance than multinomial resampling
    and the standard choice inside SMC.
    """
    rng = as_rng(rng)
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if total <= 0:
        # Degenerate weights: fall back to a uniform ancestry.
        return np.arange(len(w), dtype=np.int64)
    w = w / total
    n = len(w)
    positions = (rng.random() + np.arange(n)) / n
    cumsum = np.cumsum(w)
    cumsum[-1] = 1.0  # guard against floating-point drift
    return np.searchsorted(cumsum, positions, side="left").astype(np.int64)


@dataclass
class PowerSMCResult:
    """Everything a caller might want from a run."""

    sequences: list  # list of N sequences (tuples of token ids, including EOS)
    weights: np.ndarray  # final normalized particle weights, shape [N]
    output: tuple  # the single drawn output sequence (tuple of token ids)
    output_index: int  # which particle was drawn
    num_steps: int  # decoding steps actually taken
    ess_history: np.ndarray  # ESS after each step
    resample_steps: list  # step indices at which resampling fired
    alpha: float

    def weighted_distribution(self, support: Sequence[tuple]) -> np.ndarray:
        """Aggregate particle weights onto a fixed list of sequences.

        Used by the correctness test to compare the particle set against the exact power
        distribution. Sequences not in ``support`` are ignored (should not happen once
        every particle has terminated).
        """
        index = {seq: i for i, seq in enumerate(support)}
        out = np.zeros(len(support), dtype=np.float64)
        for seq, w in zip(self.sequences, self.weights):
            key = tuple(seq)
            if key in index:
                out[index[key]] += w
        s = out.sum()
        return out / s if s > 0 else out


def power_smc(
    model: SMCModel,
    prompt_ids: Any,
    alpha: float,
    n_particles: int = 16,
    kappa: float = 0.5,
    max_tokens: int = 2048,
    proposal: Optional[Proposal] = None,
    alpha_schedule: Optional[Any] = None,
    seed: Optional[int] = None,
    stop_when_all_done: bool = True,
    record_state: bool = False,
) -> PowerSMCResult:
    """Run Power-SMC (Algorithm 1).

    Parameters
    ----------
    model : SMCModel
        Any model implementing :class:`SMCModel`.
    prompt_ids : Any
        Passed straight to ``model.prefill`` (token ids for a real model, ignored by the
        toy model).
    alpha : float
        Power exponent. Ignored if ``alpha_schedule`` is given.
    n_particles : int
        Number of particles N.
    kappa : float
        Resample when ESS < kappa * N.
    max_tokens : int
        Hard cap on decoding steps (T_max).
    proposal : Proposal, optional
        Prefix-only proposal. Defaults to the variance-optimal temperature-1/alpha
        proposal, recomputed per step when the exponent is ramped. Pass an explicit
        proposal to run ablations.
    alpha_schedule : object, optional
        An exponent schedule exposing ``exponent(step)`` and a ``ramping`` flag
        (see :mod:`power_smc.ramping`). Enables exact exponent-bridging (alpha-ramping).
    seed : int, optional
        RNG seed.
    """
    rng = as_rng(seed)
    n = int(n_particles)

    schedule = alpha_schedule if alpha_schedule is not None else ConstantSchedule(alpha)
    ramping = bool(getattr(schedule, "ramping", False))
    target_alpha = float(getattr(schedule, "alpha", alpha))
    use_fixed_proposal = proposal is not None

    state, logprobs = model.prefill(prompt_ids, n)
    eos = model.eos_id

    seqs: list = [[] for _ in range(n)]
    log_w = np.zeros(n, dtype=np.float64)  # unnormalized log-weights (weights = 1)
    seq_logprob = np.zeros(n, dtype=np.float64)  # cumulative log p_theta of each prefix
    done = np.zeros(n, dtype=bool)

    ess_history: list = []
    resample_steps: list = []
    steps_taken = 0

    for t in range(max_tokens):
        steps_taken = t + 1
        a_curr = schedule.exponent(t)
        a_prev = schedule.exponent(t - 1) if t > 0 else schedule.exponent(-1) if ramping else a_curr

        # Exponent-bridging: when the exponent increases, reweight the existing prefix so
        # the running target moves from pi_{a_prev} to pi_{a_curr} without bias.
        if ramping and a_curr != a_prev:
            log_w += (a_curr - a_prev) * seq_logprob

        prop = proposal if use_fixed_proposal else optimal_proposal(a_curr)
        logq = prop.log_q(logprobs)

        tokens = sample_tokens(logq, rng)
        if done.any():
            tokens[done] = eos  # absorbing: finished particles keep emitting EOS

        inc = incremental_log_weight(logprobs, logq, tokens, a_curr)
        inc[done] = 0.0  # no-op transition for finished particles
        log_w += inc

        rows = np.arange(n)
        step_logprob = logprobs[rows, tokens]
        step_logprob[done] = 0.0
        seq_logprob += step_logprob

        newly_emitted_eos = (~done) & (tokens == eos)
        for i in range(n):
            if not done[i]:
                seqs[i].append(int(tokens[i]))
        done = done | newly_emitted_eos

        weights = np.exp(log_w - log_w.max())
        ess = effective_sample_size(weights)
        ess_history.append(ess)

        if ess < kappa * n:
            ancestors = systematic_resample(weights, rng)
            seqs = [list(seqs[a]) for a in ancestors]
            done = done[ancestors].copy()
            seq_logprob = seq_logprob[ancestors].copy()
            tokens = tokens[ancestors].copy()
            log_w = np.zeros(n, dtype=np.float64)  # reset weights to 1
            state = model.reorder(state, ancestors)
            resample_steps.append(t)

        if stop_when_all_done and done.all():
            break

        state, logprobs = model.decode(state, tokens)

    weights = np.exp(log_w - log_w.max())
    weights = weights / weights.sum()
    output_index = int(rng.choice(n, p=weights))

    result = PowerSMCResult(
        sequences=[tuple(s) for s in seqs],
        weights=weights,
        output=tuple(seqs[output_index]),
        output_index=output_index,
        num_steps=steps_taken,
        ess_history=np.asarray(ess_history, dtype=np.float64),
        resample_steps=resample_steps,
        alpha=target_alpha,
    )
    if record_state:
        result.__dict__["state"] = state
    return result
