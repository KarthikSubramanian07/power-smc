# Contributing

Thanks for taking an interest. This is an open reproduction of Power-SMC
(arXiv:2602.10273), so the bar for changes is simple: the code stays correct, readable,
and honest about what it does and does not reproduce.

## Setup

```bash
git clone https://github.com/KarthikSubramanian07/power-smc.git
cd power-smc
pip install -r requirements.txt
```

The core sampler and its correctness checks need only numpy, scipy, and matplotlib. torch,
transformers, datasets, and bitsandbytes are for the real-model experiments and are only
imported when you run those.

## Before you open a pull request

Run the same checks CI runs:

```bash
ruff check .        # lint
pytest -q           # full test suite (toy + real-model integration)
python validate.py  # correctness smoke: convergence + variance
```

All three should pass. CI runs them on Python 3.9 and 3.11.

## What makes a good change

- **Keep the sampler model-agnostic.** `smc.py` should not learn about a specific model.
  New models plug in by implementing the `prefill` / `decode` / `reorder` interface, the
  way `ToyMarkovModel` and `HFModel` do.
- **Add a test with a claim.** New behavior should come with a test that pins it down. For
  anything touching the target distribution, prefer a check against the exact toy
  distribution or against a closed-form quantity, not just a "does not crash" test.
- **Match the paper, or say where you diverge.** If a change departs from the paper,
  document it in the README's matches and divergences section rather than quietly
  shipping it.
- **Do not commit heavy artifacts.** Model weights, caches, and large raw outputs are
  gitignored. Committed results should be small CSVs and plots.

## Good things to work on

- Run the full MATH500 benchmark on the three models the paper uses and contribute the
  accuracy-vs-latency numbers and plot.
- Keep next-token log-probs on the GPU through the loop instead of moving to numpy each
  step, and measure the latency difference.
- Add the sibling Scalable Power Sampling method (arXiv:2601.21590) as a fourth point for
  a power-sampling family comparison.

## Reporting issues

Open an issue with the model, dataset subset, particle count, alpha, and the command you
ran. For a correctness problem, the most useful report is a small case where the toy
sampler diverges from the exact power distribution.

By contributing you agree that your contributions are licensed under the MIT License.
