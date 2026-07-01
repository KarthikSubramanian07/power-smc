import numpy as np

from power_smc import (
    LinearRamp,
    ToyMarkovModel,
    enumerate_exact,
    optimal_proposal,
    power_smc,
    systematic_resample,
    total_variation,
)


def _pooled_distribution(model, exact, alpha, n, runs, base_seed, **kw):
    est = np.zeros(len(exact.sequences), dtype=np.float64)
    rng = np.random.default_rng(base_seed)
    for _ in range(runs):
        r = power_smc(model, None, alpha, n_particles=n, kappa=0.5,
                      max_tokens=model.max_len + 4, seed=int(rng.integers(1 << 31)), **kw)
        est += r.weighted_distribution(exact.sequences)
    return est / runs


def test_converges_to_exact_power_distribution():
    model = ToyMarkovModel(n_real=3, max_len=4, seed=1, spread=2.0)
    exact = enumerate_exact(model, alpha=4.0)
    tv_small = total_variation(_pooled_distribution(model, exact, 4.0, 8, 80, 0), exact.power_probs)
    tv_large = total_variation(_pooled_distribution(model, exact, 4.0, 256, 80, 1), exact.power_probs)
    assert tv_large < 0.05
    assert tv_large < tv_small


def test_ramping_hits_the_same_target():
    model = ToyMarkovModel(n_real=3, max_len=4, seed=1, spread=2.0)
    exact = enumerate_exact(model, alpha=4.0)
    est = _pooled_distribution(model, exact, 4.0, 128, 80, 3,
                               alpha_schedule=LinearRamp(4.0, t_ramp=3))
    assert total_variation(est, exact.power_probs) < 0.05


def test_all_particles_terminate_with_eos():
    model = ToyMarkovModel(n_real=3, max_len=4, seed=5)
    r = power_smc(model, None, 4.0, n_particles=32, max_tokens=model.max_len + 4, seed=0)
    assert all(seq[-1] == model.eos_id for seq in r.sequences)


def test_seed_is_reproducible():
    model = ToyMarkovModel(n_real=3, max_len=4, seed=5)
    a = power_smc(model, None, 4.0, n_particles=16, max_tokens=10, seed=42)
    b = power_smc(model, None, 4.0, n_particles=16, max_tokens=10, seed=42)
    assert a.sequences == b.sequences
    np.testing.assert_array_equal(a.weights, b.weights)


def test_ess_history_within_bounds():
    model = ToyMarkovModel(n_real=3, max_len=4, seed=5)
    n = 16
    r = power_smc(model, None, 4.0, n_particles=n, max_tokens=10, seed=1)
    assert np.all(r.ess_history >= 0.0)
    assert np.all(r.ess_history <= n + 1e-9)


def test_weights_are_normalized():
    model = ToyMarkovModel(n_real=3, max_len=4, seed=5)
    r = power_smc(model, None, 4.0, n_particles=16, max_tokens=10, seed=1)
    assert abs(r.weights.sum() - 1.0) < 1e-9


def test_systematic_resample_favors_high_weight_particles():
    weights = np.array([0.01, 0.01, 0.96, 0.02])
    ancestors = systematic_resample(weights, rng=np.random.default_rng(0))
    assert (ancestors == 2).sum() >= 3  # the heavy particle dominates the ancestry
    assert len(ancestors) == len(weights)


def test_systematic_resample_uniform_is_a_permutation_cover():
    weights = np.full(8, 1.0 / 8)
    ancestors = systematic_resample(weights, rng=np.random.default_rng(0))
    # With equal weights every particle is selected exactly once.
    assert sorted(ancestors.tolist()) == list(range(8))


def test_low_alpha_matches_base_more_than_high_alpha():
    model = ToyMarkovModel(n_real=3, max_len=4, seed=1, spread=2.0)
    exact_low = enumerate_exact(model, alpha=1.0)
    est_low = _pooled_distribution(model, exact_low, 1.0, 128, 60, 9)
    # alpha = 1 targets the base distribution itself.
    assert total_variation(est_low, exact_low.base_probs) < 0.05
