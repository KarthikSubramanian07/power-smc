"""Power-SMC: low-latency sequence-level power sampling for training-free reasoning.

An open reproduction of Power-SMC (arXiv 2602.10273). The public API below is enough to
sample from the power distribution with a toy model or a real Transformer.
"""

from __future__ import annotations

from .power_target import (
    ExactEnumeration,
    ToyMarkovModel,
    enumerate_exact,
    power_distribution,
)
from .proposal import (
    Proposal,
    TemperatureProposal,
    conditional_weight_variance,
    incremental_log_weight,
    optimal_proposal,
    sample_tokens,
)
from .ramping import AlphaSchedule, ConstantAlpha, LinearRamp
from .smc import (
    PowerSMCResult,
    SMCModel,
    power_smc,
    systematic_resample,
)
from .utils import effective_sample_size, total_variation

__all__ = [
    "ToyMarkovModel",
    "ExactEnumeration",
    "enumerate_exact",
    "power_distribution",
    "Proposal",
    "TemperatureProposal",
    "optimal_proposal",
    "incremental_log_weight",
    "sample_tokens",
    "conditional_weight_variance",
    "AlphaSchedule",
    "ConstantAlpha",
    "LinearRamp",
    "power_smc",
    "PowerSMCResult",
    "SMCModel",
    "systematic_resample",
    "effective_sample_size",
    "total_variation",
]

__version__ = "0.1.0"
