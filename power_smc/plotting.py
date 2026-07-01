"""Plots for the validation checks and the headline accuracy-vs-latency tradeoff.

matplotlib is imported lazily and a non-interactive backend is used so these run on a
headless Colab / Kaggle worker.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_convergence(particle_counts: Sequence[int], tv_distances: Sequence[float],
                     out_path: str, title: str = "Toy convergence to the power distribution"):
    """TV distance from the exact power distribution vs particle count (log-log)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(particle_counts, tv_distances, "o-", color="#2c6fbb")
    ax.set_xlabel("particles N")
    ax.set_ylabel("total variation to exact power distribution")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_variance(temperatures: Sequence[float], variances: Sequence[float], alpha: float,
                  out_path: str):
    """Incremental-weight variance vs proposal temperature, marking tau = 1/alpha."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(temperatures, np.maximum(variances, 1e-16), "o-", color="#b5651d")
    ax.axvline(1.0 / alpha, color="#2c6fbb", ls="--",
               label=f"tau = 1/alpha = {1.0 / alpha:.3g}")
    ax.set_xlabel("proposal temperature tau")
    ax.set_ylabel("Var[incremental weight | prefix]")
    ax.set_title(f"Optimal proposal minimizes weight variance (alpha={alpha:g})")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_accuracy_latency(points: Sequence[dict], out_path: str,
                          title: str = "MATH500: accuracy vs latency"):
    """Scatter of methods on (latency overhead, accuracy).

    ``points`` is a list of dicts with keys ``method``, ``latency`` (x, e.g. overhead
    multiple or seconds/problem) and ``accuracy`` (y, in [0, 1]).
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"baseline": "#777777", "mh": "#b5651d", "power-smc": "#2c6fbb"}
    for p in points:
        method = str(p["method"]).lower()
        ax.scatter(p["latency"], p["accuracy"], s=90,
                   color=colors.get(method, "#444444"), zorder=3)
        ax.annotate(p["method"], (p["latency"], p["accuracy"]),
                    textcoords="offset points", xytext=(8, 4))
    ax.set_xlabel("latency overhead vs baseline decoding (x)")
    ax.set_ylabel("MATH500 accuracy")
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
