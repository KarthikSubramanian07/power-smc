import numpy as np

from power_smc import ToyMarkovModel, enumerate_exact, power_distribution


def test_base_distribution_is_normalized():
    model = ToyMarkovModel(n_real=3, max_len=4, seed=1)
    exact = enumerate_exact(model, alpha=4.0)
    assert abs(exact.base_probs.sum() - 1.0) < 1e-9
    assert abs(exact.power_probs.sum() - 1.0) < 1e-9


def test_every_sequence_terminates_with_eos():
    model = ToyMarkovModel(n_real=3, max_len=4, seed=2)
    exact = enumerate_exact(model, alpha=2.0)
    assert all(seq[-1] == model.eos_id for seq in exact.sequences)
    # No real tokens beyond max_len.
    assert all(len(seq) - 1 <= model.max_len for seq in exact.sequences)


def test_power_distribution_matches_manual():
    base = np.array([0.5, 0.3, 0.2])
    alpha = 2.0
    expected = base ** alpha
    expected /= expected.sum()
    np.testing.assert_allclose(power_distribution(base, alpha), expected, rtol=1e-12)


def test_alpha_one_is_identity():
    base = np.array([0.1, 0.4, 0.5])
    np.testing.assert_allclose(power_distribution(base, 1.0), base, rtol=1e-12)


def test_power_concentrates_mass_on_the_mode():
    base = np.array([0.6, 0.3, 0.1])
    sharp = power_distribution(base, 8.0)
    # The most likely sequence gets even more mass under the power distribution.
    assert sharp[0] > base[0]
    assert sharp.argmax() == base.argmax()


def test_conditional_logprobs_are_valid():
    model = ToyMarkovModel(n_real=4, max_len=3, seed=3)
    _, logp = model.prefill(None, 5)
    probs = np.exp(logp)
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-9)
