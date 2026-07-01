import numpy as np

from power_smc import (
    TemperatureProposal,
    conditional_weight_variance,
    incremental_log_weight,
    optimal_proposal,
    sample_tokens,
)
from power_smc.utils import log_softmax


def _row(seed=0, v=5):
    rng = np.random.default_rng(seed)
    return log_softmax(rng.normal(size=v))


def test_optimal_proposal_is_temperature_one_over_alpha():
    prop = optimal_proposal(4.0)
    assert isinstance(prop, TemperatureProposal)
    assert abs(prop.tau - 0.25) < 1e-12


def test_optimal_proposal_has_zero_conditional_variance():
    row = _row(seed=1)
    for alpha in (1.5, 2.0, 4.0, 8.0):
        var = conditional_weight_variance(row, optimal_proposal(alpha), alpha)
        assert var < 1e-10


def test_variance_is_minimized_at_one_over_alpha():
    row = _row(seed=2)
    alpha = 4.0
    taus = np.linspace(0.1, 2.0, 40)
    variances = [conditional_weight_variance(row, TemperatureProposal(t), alpha) for t in taus]
    best_tau = taus[int(np.argmin(variances))]
    assert abs(best_tau - 1.0 / alpha) < 0.06


def test_incremental_weight_formula():
    logp = np.array([log_softmax(np.array([0.2, 0.5, 0.3]))])
    alpha = 3.0
    prop = optimal_proposal(alpha)
    logq = prop.log_q(logp)
    tokens = np.array([1])
    got = incremental_log_weight(logp, logq, tokens, alpha)
    expected = alpha * logp[0, 1] - logq[0, 1]
    assert abs(got[0] - expected) < 1e-12


def test_sample_tokens_respects_forbidden_entries():
    logq = np.full((100, 4), -np.inf)
    logq[:, 2] = 0.0  # only token 2 is allowed
    tokens = sample_tokens(logq, rng=np.random.default_rng(0))
    assert np.all(tokens == 2)


def test_sample_tokens_matches_target_frequencies():
    probs = np.array([0.1, 0.6, 0.3])
    logq = np.log(probs)[None, :].repeat(20000, axis=0)
    tokens = sample_tokens(logq, rng=np.random.default_rng(7))
    freqs = np.bincount(tokens, minlength=3) / len(tokens)
    np.testing.assert_allclose(freqs, probs, atol=0.02)
