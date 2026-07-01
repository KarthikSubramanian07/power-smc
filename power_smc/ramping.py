"""Alpha-ramping (exponent bridging).

Instead of targeting the full exponent ``alpha`` from the first token, the exponent is
annealed along a schedule ``1 = a^0 < a^1 < ... < a^L = alpha``. When the exponent steps
up, the existing prefix is reweighted by ``(a^l - a^{l-1}) * log p_theta(y_1:t | x)``
(applied inside the SMC loop). This is an exact bridge: it improves particle stability
early in decoding without changing the final target, which is still pi_alpha.

The schedule objects here expose:
  * ``exponent(step)`` -> the exponent used at a given (0-based) decoding step, with
    ``exponent(-1)`` returning the starting exponent a^0 = 1;
  * ``alpha`` -> the final target exponent;
  * ``ramping`` -> whether bridging reweights should be applied.
"""

from __future__ import annotations


class AlphaSchedule:
    """Base schedule interface."""

    alpha: float
    ramping: bool

    def exponent(self, step: int) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class ConstantAlpha(AlphaSchedule):
    """No ramping: the exponent is ``alpha`` at every step."""

    def __init__(self, alpha: float):
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = float(alpha)
        self.ramping = False

    def exponent(self, step: int) -> float:
        return self.alpha


class LinearRamp(AlphaSchedule):
    """Linear exponent bridge over the first ``t_ramp`` tokens.

    a(t) = 1 + (alpha - 1) * min(t + 1, t_ramp) / t_ramp, and a(-1) = 1. The exponent
    reaches the target ``alpha`` at step ``t_ramp - 1`` and stays there afterwards.
    """

    def __init__(self, alpha: float, t_ramp: int = 100):
        if alpha < 1:
            raise ValueError("alpha must be >= 1 for ramping")
        if t_ramp < 1:
            raise ValueError("t_ramp must be >= 1")
        self.alpha = float(alpha)
        self.t_ramp = int(t_ramp)
        self.ramping = True

    def exponent(self, step: int) -> float:
        if step < 0:
            return 1.0
        frac = min((step + 1) / self.t_ramp, 1.0)
        return 1.0 + (self.alpha - 1.0) * frac
