"""Baselines: plain decoding, and a wrapper around the MH power-sampling reference.

Two baselines matter for the headline accuracy-vs-latency plot:

* **Baseline decoding** - ordinary sampling from the model at a chosen temperature,
  run through the *same* decoding stack as Power-SMC so the latency comparison is fair.
  Note that decoding at temperature 1/alpha is *not* power sampling: tempering each
  token independently does not exponentiate the joint sequence probability. This
  function is offered precisely so that contrast can be measured.

* **Metropolis-Hastings power sampling** - the reference implementation from Karan & Du,
  "Reasoning with Sampling" (github.com/aakaran/reasoning-with-sampling, arXiv:2510.14901).
  We do not reimplement MH; we wrap their code and validate Power-SMC against it.
  :class:`MHReference` clones the repo, runs their MATH500 power-sampling script with the
  right flags, and locates the output CSV so it can be graded and plotted alongside ours.
"""

from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from .proposal import TemperatureProposal, sample_tokens
from .utils import as_rng

MH_REPO_URL = "https://github.com/aakaran/reasoning-with-sampling"
MH_PAPER_ARXIV = "2510.14901"

# The reference script exposes models by short name; these are the HF ids it loads. The
# first three match the models the Power-SMC paper reports on.
MH_MODEL_IDS = {
    "qwen": "Qwen/Qwen2.5-7B",
    "qwen_math": "Qwen/Qwen2.5-Math-7B",
    "phi": "microsoft/Phi-3.5-mini-instruct",
    "tulu": "allenai/Llama-3.1-Tulu-3-8B-DPO",
}


def baseline_decode(
    model: Any,
    prompt_ids: Any,
    n_samples: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    greedy: bool = False,
    seed: Optional[int] = None,
) -> list:
    """Ordinary autoregressive decoding through the SMC model interface.

    Runs ``n_samples`` independent continuations in one batch, sampling each token from
    the model distribution tempered by ``temperature`` (or greedily). EOS is absorbing.
    Returns a list of token-id tuples. Using the same ``prefill``/``decode`` path as
    Power-SMC keeps the wall-clock comparison honest.
    """
    rng = as_rng(seed)
    n = int(n_samples)
    proposal = TemperatureProposal(temperature)

    state, logprobs = model.prefill(prompt_ids, n)
    eos = model.eos_id
    seqs: list = [[] for _ in range(n)]
    done = np.zeros(n, dtype=bool)

    for _ in range(max_tokens):
        if greedy:
            tokens = np.argmax(logprobs, axis=-1).astype(np.int64)
        else:
            tokens = sample_tokens(proposal.log_q(logprobs), rng)
        if done.any():
            tokens[done] = eos

        for i in range(n):
            if not done[i]:
                seqs[i].append(int(tokens[i]))
        done = done | (tokens == eos)
        if done.all():
            break
        state, logprobs = model.decode(state, tokens)

    return [tuple(s) for s in seqs]


@dataclass
class MHReference:
    """Wrapper around Karan & Du's MH power-sampling repo, used as the reference.

    This is a *reference*, not a reimplementation. The typical flow, all handled by
    :func:`experiments.run_mh`, is: clone the repo, run ``power_samp_math.py`` for one or
    more 100-problem shards, then grade the ``mcmc_answer`` column of the output CSV.
    """

    root: str = "third_party/reasoning-with-sampling"

    def is_available(self) -> bool:
        return os.path.isdir(os.path.join(self.root, "llm_experiments"))

    def clone(self) -> None:
        """Clone the reference repo if it is not already present."""
        if self.is_available():
            return
        parent = os.path.dirname(self.root)
        if parent:
            os.makedirs(parent, exist_ok=True)
        subprocess.run(["git", "clone", MH_REPO_URL, self.root], check=True)

    def output_csv(self, model: str, mcmc_steps: int, temperature: float, batch_idx: int,
                   seed: int, save_str: str = "results") -> str:
        """Return the CSV path the reference script writes for a given configuration.

        The reference builds the name from ``str()`` of each value, so we match that
        formatting exactly (in particular for the float temperature).
        """
        fname = (f"{model}_math_base_power_samp_results_{mcmc_steps}_"
                 f"{float(temperature)}_{batch_idx}_{seed}.csv")
        return os.path.join(self.root, save_str, model, fname)

    def run_math(
        self,
        model: str = "qwen",
        mcmc_steps: int = 10,
        temperature: float = 0.25,
        batch_idx: int = 0,
        seed: int = 0,
        save_str: str = "results",
        device: str = "cuda",
        python_bin: str = "python",
        extra_args: Optional[Sequence[str]] = None,
    ) -> subprocess.CompletedProcess:
        """Run one 100-problem shard of the reference MATH500 power-sampling script.

        ``temperature`` is 1/alpha for the target exponent (0.25 corresponds to alpha=4).
        ``batch_idx`` selects problems ``100*batch_idx : 100*(batch_idx+1)``. Raises if the
        script exits nonzero.
        """
        if not self.is_available():
            raise FileNotFoundError(
                f"reference repo not found at {self.root!r}; call clone() first"
            )
        script = os.path.join("llm_experiments", "power_samp_math.py")
        cmd = [
            python_bin, script,
            "--model", model,
            "--mcmc_steps", str(mcmc_steps),
            "--temperature", str(float(temperature)),
            "--batch_idx", str(batch_idx),
            "--seed", str(seed),
            "--save_str", save_str,
            "--device", device,
            *(extra_args or []),
        ]
        return subprocess.run(cmd, check=True, cwd=self.root)

    @staticmethod
    def load_results(csv_path: str) -> list:
        """Load a reference output CSV into a list of row dicts.

        Columns include ``question``, ``correct_answer``, ``std_answer`` (standard
        decoding), ``naive_answer`` (naive temperature), and ``mcmc_answer`` (MH power
        sampling, the column to grade for the MH point).
        """
        with open(csv_path, newline="") as fh:
            return list(csv.DictReader(fh))
