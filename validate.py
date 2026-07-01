"""Correctness validation for Power-SMC, runnable on a CPU in seconds.

This reproduces the two claims that must hold before any benchmark number is trusted:

  1. On a tiny model whose power distribution can be computed exactly, the SMC particle
     set converges to it as the number of particles grows.
  2. The temperature-1/alpha proposal minimizes the incremental-weight variance among
     prefix-only proposals (Theorem 1).

It writes CSVs and plots to ``results/`` and prints a short summary. No GPU or model
download required.

Usage:
    python validate.py
"""

from __future__ import annotations

import csv
import os

import numpy as np

from power_smc import (
    ToyMarkovModel,
    TemperatureProposal,
    conditional_weight_variance,
    enumerate_exact,
    power_smc,
    total_variation,
)
from power_smc.plotting import plot_convergence, plot_variance

RESULTS = "results"
PLOTS = os.path.join(RESULTS, "plots")


def run_convergence(alpha: float = 4.0, particle_counts=(4, 8, 16, 32, 64, 128, 256, 512),
                    runs_per_setting: int = 200, seed: int = 0):
    """Pool many independent runs per N and measure TV to the exact power distribution."""
    model = ToyMarkovModel(n_real=3, max_len=4, seed=1, spread=2.0)
    exact = enumerate_exact(model, alpha)
    rng = np.random.default_rng(seed)

    counts, tvs = [], []
    for n in particle_counts:
        est = np.zeros(len(exact.sequences), dtype=np.float64)
        for _ in range(runs_per_setting):
            r = power_smc(model, None, alpha, n_particles=n, kappa=0.5,
                          max_tokens=model.max_len + 4, seed=int(rng.integers(1 << 31)))
            est += r.weighted_distribution(exact.sequences)
        est /= runs_per_setting
        tv = total_variation(est, exact.power_probs)
        counts.append(n)
        tvs.append(tv)
        print(f"  N={n:4d}   TV={tv:.4f}")
    return exact, counts, tvs


def run_variance(alpha: float = 4.0,
                 temperatures=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0),
                 seed: int = 2):
    """Exact incremental-weight variance at a representative prefix, per temperature."""
    model = ToyMarkovModel(n_real=4, max_len=3, seed=seed, spread=2.5)
    _, logp = model.prefill(None, 1)
    row = logp[0]
    taus, variances = [], []
    for tau in temperatures:
        v = conditional_weight_variance(row, TemperatureProposal(tau), alpha)
        taus.append(tau)
        variances.append(v)
        marker = "  <- 1/alpha" if abs(tau - 1.0 / alpha) < 1e-9 else ""
        print(f"  tau={tau:.3f}   Var={v:.3e}{marker}")
    return taus, variances


def main():
    os.makedirs(PLOTS, exist_ok=True)
    alpha = 4.0

    print("[1/2] Convergence to the exact power distribution")
    exact, counts, tvs = run_convergence(alpha=alpha)
    with open(os.path.join(RESULTS, "toy_convergence.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n_particles", "tv_to_exact_power"])
        w.writerows(zip(counts, tvs))
    plot_convergence(counts, tvs, os.path.join(PLOTS, "toy_convergence.png"))

    print("\n[2/2] Incremental-weight variance vs proposal temperature")
    taus, variances = run_variance(alpha=alpha)
    with open(os.path.join(RESULTS, "toy_variance.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["temperature", "incremental_weight_variance"])
        w.writerows(zip(taus, variances))
    plot_variance(taus, variances, alpha, os.path.join(PLOTS, "toy_variance.png"))

    best_tau = taus[int(np.argmin(variances))]
    print("\nSummary")
    print(f"  exact power support size : {len(exact.sequences)}")
    print(f"  TV at N={counts[-1]:<4d}          : {tvs[-1]:.4f}  (should be small)")
    print(f"  variance-minimizing tau  : {best_tau:.3f}  (theory: 1/alpha = {1/alpha:.3f})")
    print("  wrote results/toy_convergence.csv, results/toy_variance.csv, results/plots/*.png")


if __name__ == "__main__":
    main()
