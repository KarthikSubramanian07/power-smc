# Results

Small, committed artifacts that show the sampler is correct and reproduce the central
tradeoff. Large raw dumps (`results/raw/`, `*.jsonl`) are gitignored.

## Validation (CPU, produced by `python validate.py`)

- `toy_convergence.csv`, `plots/toy_convergence.png` - total variation between the
  Power-SMC particle set and the exact power distribution on the toy model, as the
  particle count grows. It should trend toward zero.
- `toy_variance.csv`, `plots/toy_variance.png` - incremental-weight variance as a
  function of proposal temperature. The minimum sits at `tau = 1/alpha`, matching
  Theorem 1.

## MATH500 (GPU, produced by `experiments.py`)

- `math500_<method>.csv` - per-problem correctness and wall-clock latency for each of
  baseline decoding, MH power sampling (reference), and Power-SMC.
- `plots/accuracy_latency.png` - the headline plot: accuracy against latency overhead
  relative to baseline decoding.

These CSVs are generated on Colab / Kaggle (see the top-level README) and are not checked
in by default, since they depend on the model and hardware used.
