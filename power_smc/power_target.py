"""The sequence-level power distribution and a toy model we can enumerate exactly.

The power target is

    pi_alpha(y | x) = p_theta(y | x)^alpha / Z_alpha(x),
    Z_alpha(x)      = sum_y p_theta(y | x)^alpha,      alpha >= 1.                (Eq. 1)

Raising the *joint* sequence probability to a power and renormalizing is not the same
as lowering the temperature of each token independently, because exponentiating a
product of conditionals and renormalizing globally does not factorize. That gap is the
whole reason a sampler is needed, and it is what the toy model here lets us check by
brute force.

``ToyMarkovModel`` is a tiny autoregressive model with a closed-form joint. Because the
set of terminated sequences is finite, we can enumerate every one of them, compute the
exact power distribution, and later confirm that the SMC particle set converges to it.
The model is a first-order Markov chain, but note that its power distribution is *not*
Markov, so importance weighting and resampling are genuinely exercised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .utils import as_rng, log_softmax


def power_distribution(base_probs: np.ndarray, alpha: float) -> np.ndarray:
    """Return pi_alpha given the base probabilities of each sequence (Eq. 1)."""
    base = np.asarray(base_probs, dtype=np.float64)
    # Work in log-space for stability with long sequences.
    with np.errstate(divide="ignore"):
        log_base = np.log(base)
    log_power = alpha * log_base
    log_power -= np.max(log_power[np.isfinite(log_power)])
    power = np.exp(log_power)
    total = power.sum()
    return power / total if total > 0 else power


@dataclass
class ToyState:
    """Per-particle decoding state for the toy model (what ``reorder`` reindexes)."""

    last_token: np.ndarray  # int array [N]; -1 means "start / BOS"
    position: np.ndarray  # int array [N]; number of real tokens emitted so far
    done: np.ndarray  # bool array [N]; True once EOS has been emitted


class ToyMarkovModel:
    """A small, exactly enumerable autoregressive model.

    Real tokens are ``0 .. n_real - 1`` and EOS is ``n_real``. Sampling forces EOS once
    ``max_len`` real tokens have been emitted, so every sequence terminates and the set
    of full sequences is finite. Conditional logits are a fixed random function of the
    previous token (with a dedicated BOS row for the first step).

    The class implements the ``SMCModel`` protocol used by :mod:`power_smc.smc`, so the
    exact same Algorithm 1 code path drives both this toy and a real Transformer.
    """

    def __init__(self, n_real: int = 3, max_len: int = 4, seed: int = 0,
                 spread: float = 2.0):
        if n_real < 1:
            raise ValueError("n_real must be >= 1")
        rng = as_rng(seed)
        self.n_real = int(n_real)
        self.eos_id = int(n_real)
        self.vocab_size = int(n_real) + 1
        self.max_len = int(max_len)
        # spread scales the logits: larger spread => sharper, more skewed conditionals,
        # which makes the power distribution differ more from the base distribution.
        self._bos_logits = rng.normal(scale=spread, size=self.vocab_size)
        self._trans_logits = rng.normal(scale=spread, size=(self.n_real, self.vocab_size))

    # -- raw conditional -----------------------------------------------------------
    def _cond_logprobs(self, prev_token: int, position: int) -> np.ndarray:
        """log p(. | prev_token) at the given position, over the full vocab."""
        if position >= self.max_len:
            forced = np.full(self.vocab_size, -np.inf)
            forced[self.eos_id] = 0.0
            return forced
        base = self._bos_logits if prev_token < 0 else self._trans_logits[prev_token]
        return log_softmax(base)

    # -- SMCModel protocol ---------------------------------------------------------
    def prefill(self, prompt_ids, n: int):
        """Process the (ignored) prompt and return log-probs for the first token."""
        state = ToyState(
            last_token=np.full(n, -1, dtype=np.int64),
            position=np.zeros(n, dtype=np.int64),
            done=np.zeros(n, dtype=bool),
        )
        logprobs = self._batch_logprobs(state)
        return state, logprobs

    def decode(self, state: ToyState, tokens: np.ndarray):
        """Feed the chosen tokens and return log-probs for the next position."""
        tokens = np.asarray(tokens, dtype=np.int64)
        for i in range(len(tokens)):
            if state.done[i]:
                continue
            if tokens[i] == self.eos_id:
                state.done[i] = True
            else:
                state.last_token[i] = tokens[i]
                state.position[i] += 1
        return state, self._batch_logprobs(state)

    def reorder(self, state: ToyState, ancestors: np.ndarray) -> ToyState:
        """Reindex per-particle state by ancestor index (cache-safe resampling)."""
        a = np.asarray(ancestors, dtype=np.int64)
        return ToyState(
            last_token=state.last_token[a].copy(),
            position=state.position[a].copy(),
            done=state.done[a].copy(),
        )

    def _batch_logprobs(self, state: ToyState) -> np.ndarray:
        out = np.empty((len(state.last_token), self.vocab_size), dtype=np.float64)
        for i in range(len(state.last_token)):
            if state.done[i]:
                forced = np.full(self.vocab_size, -np.inf)
                forced[self.eos_id] = 0.0
                out[i] = forced
            else:
                out[i] = self._cond_logprobs(int(state.last_token[i]),
                                             int(state.position[i]))
        return out


@dataclass
class ExactEnumeration:
    """The exact base and power distributions over every terminated sequence."""

    sequences: list  # list of tuples, each ending in eos_id
    base_probs: np.ndarray  # p_theta(y) for each sequence
    power_probs: np.ndarray  # pi_alpha(y) for each sequence
    alpha: float
    index: dict = field(default_factory=dict)  # sequence tuple -> row

    def prob_of(self, seq) -> float:
        """pi_alpha of a given sequence tuple (0 if unreachable)."""
        return float(self.power_probs[self.index[tuple(seq)]]) if tuple(seq) in self.index else 0.0


def enumerate_exact(model: ToyMarkovModel, alpha: float) -> ExactEnumeration:
    """Brute-force every terminated sequence and compute base + power distributions.

    This is the ground truth for the correctness test. It exists only for the toy
    model, where the sequence set is finite.
    """
    sequences: list = []
    log_base: list = []

    # Depth-first walk over the generation tree. Each stack entry is
    # (prefix tuple of real tokens, previous token, position, accumulated log-prob).
    stack = [((), -1, 0, 0.0)]
    while stack:
        prefix, prev, pos, acc = stack.pop()
        logp = model._cond_logprobs(prev, pos)
        for tok in range(model.vocab_size):
            lp = logp[tok]
            if not np.isfinite(lp):
                continue
            if tok == model.eos_id:
                seq = prefix + (model.eos_id,)
                sequences.append(seq)
                log_base.append(acc + lp)
            else:
                stack.append((prefix + (tok,), tok, pos + 1, acc + lp))

    log_base_arr = np.asarray(log_base, dtype=np.float64)
    base_probs = np.exp(log_base_arr)
    # Guard: the base distribution over terminated sequences should sum to 1.
    base_probs = base_probs / base_probs.sum()
    power_probs = power_distribution(base_probs, alpha)
    index = {seq: i for i, seq in enumerate(sequences)}
    return ExactEnumeration(
        sequences=sequences,
        base_probs=base_probs,
        power_probs=power_probs,
        alpha=alpha,
        index=index,
    )
