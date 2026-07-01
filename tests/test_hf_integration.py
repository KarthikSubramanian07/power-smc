"""Integration tests for the real Hugging Face path on a tiny in-memory model.

These exercise the code that the toy tests cannot: HFModel.prefill/decode driving the SMC
loop, and reordering an actual transformers cache object (a DynamicCache) during
resampling. A tiny randomly initialized GPT-2 is built from a config, so nothing is
downloaded. The whole module is skipped if transformers is not installed.
"""

import numpy as np
import pytest

pytest.importorskip("transformers")

import torch  # noqa: E402
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from power_smc import power_smc  # noqa: E402
from power_smc.baselines import baseline_decode  # noqa: E402
from power_smc.hf_model import HFModel  # noqa: E402

EOS = 63


def build_model():
    config = GPT2Config(vocab_size=64, n_positions=128, n_embd=32, n_layer=2, n_head=2)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(config)
    return HFModel.from_model(model, tokenizer=None, eos_id=EOS)


def test_smc_loop_runs_and_reorders_a_real_cache():
    model = build_model()
    prompt = [1, 2, 3, 4]
    # kappa close to 1 forces resampling once particle prefixes diverge, so the real
    # DynamicCache reorder path actually runs.
    result = power_smc(model, prompt, alpha=3.0, n_particles=6, kappa=0.99,
                       max_tokens=20, seed=0)
    assert abs(result.weights.sum() - 1.0) < 1e-9
    assert len(result.resample_steps) >= 1
    assert 1 <= len(result.output) <= 20


def test_reorder_then_decode_is_consistent():
    model = build_model()
    state, logprobs = model.prefill([1, 2, 3, 4], 4)
    assert logprobs.shape == (4, model.vocab_size)
    state = model.reorder(state, np.array([3, 3, 0, 1]))
    state, logprobs = model.decode(state, np.array([5, 6, 7, 8]))
    assert logprobs.shape == (4, model.vocab_size)
    np.testing.assert_allclose(np.exp(logprobs).sum(axis=-1), 1.0, atol=1e-4)


def test_baseline_decode_runs_on_real_model():
    model = build_model()
    seqs = baseline_decode(model, [1, 2, 3, 4], n_samples=3, temperature=1.0,
                           max_tokens=8, seed=1)
    assert len(seqs) == 3
    assert all(isinstance(s, tuple) for s in seqs)


def test_from_model_requires_an_eos_id():
    config = GPT2Config(vocab_size=64, n_positions=32, n_embd=16, n_layer=1, n_head=2)
    model = GPT2LMHeadModel(config)
    model.config.eos_token_id = None
    with pytest.raises(ValueError):
        HFModel.from_model(model, tokenizer=None, eos_id=None)
