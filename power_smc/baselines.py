"""Baselines: plain decoding, and a wrapper around the MH power-sampling reference.

Two baselines matter for the headline accuracy-vs-latency plot:

* **Baseline decoding** - ordinary sampling from the model at a chosen temperature,
  run through the *same* decoding stack as Power-SMC so the latency comparison is fair.
  Note that decoding at temperature 1/alpha is *not* power sampling: tempering each
  token independently does not exponentiate the joint sequence probability. This
  function is offered precisely so that contrast can be measured.

* **Metropolis-Hastings power sampling** - the reference implementation from Karan & Du,
  "Reasoning with Sampling" (github.com/aakaran/reasoning-with-sampling). We do not
  reimplement MH; we wrap their code and validate Power-SMC against it. :class:`MHReference`
  handles cloning and invoking their MATH500 scripts and parsing the resulting CSVs.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from .proposal import TemperatureProposal, sample_tokens
from .utils import as_rng

MH_REPO_URL = "https://github.com/aakaran/reasoning-with-sampling"


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

    steps = 0
    for _ in range(max_tokens):
        steps += 1
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
    """Thin wrapper around Karan & Du's MH power-sampling repo.

    This is a *reference*, not a reimplementation. Typical use::

        mh = MHReference(root="third_party/reasoning-with-sampling")
        mh.clone()                 # git clone if missing
        mh.run_math(...)           # invoke their MATH500 power-sampling script
        df = mh.load_results(csv)  # parse their output for comparison

    The scripts are SLURM-oriented; see the README for running them directly on Colab or
    Kaggle. Methods here shell out and parse CSVs rather than importing their internals,
    which keeps this repo decoupled from their exact module layout.
    """

    root: str = "third_party/reasoning-with-sampling"

    def is_available(self) -> bool:
        return os.path.isdir(os.path.join(self.root, ".git")) or os.path.isdir(self.root)

    def clone(self) -> None:
        """Clone the reference repo if it is not already present."""
        if self.is_available():
            return
        os.makedirs(os.path.dirname(self.root) or ".", exist_ok=True)
        subprocess.run(["git", "clone", MH_REPO_URL, self.root], check=True)

    def run_math(
        self,
        script: str = "llm_experiments/power_samp_math.py",
        extra_args: Optional[Sequence[str]] = None,
        python_bin: str = "python",
    ) -> subprocess.CompletedProcess:
        """Invoke the reference MATH500 power-sampling script as a subprocess.

        ``extra_args`` are passed straight through (model id, subset size, alpha, output
        path, ...). Consult the reference repo for its exact flags; they are echoed in the
        README. Raises if the script exits nonzero.
        """
        if not self.is_available():
            raise FileNotFoundError(
                f"reference repo not found at {self.root!r}; call clone() first"
            )
        cmd = [python_bin, os.path.join(self.root, script), *(extra_args or [])]
        return subprocess.run(cmd, check=True)

    @staticmethod
    def load_results(csv_path: str):
        """Load a reference output CSV (response / correct answer / prompt columns)."""
        import csv

        rows = []
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
        return rows
