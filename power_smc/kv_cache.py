"""Cache-safe resampling: reindex a Transformer KV cache by ancestor.

When SMC resamples, particle ``i`` inherits the full prefix of ancestor ``A_i``, so its
cached keys and values must become ancestor ``A_i``'s cached keys and values. Getting
this wrong silently corrupts decoding: the logits would be conditioned on the wrong
history. This module reorders the cache along the batch (particle) dimension.

Following Appendix C, it uses a three-tier strategy so it works across transformers
versions and cache formats:

  1. a model-provided cache-reordering hook (``model._reorder_cache``), if present;
  2. the cache object's own reorder method (``reorder_cache`` / ``batch_select_indices``);
  3. a recursive tensor reindexer that treats the cache as a nested container and calls
     ``index_select(0, beam_idx)`` on every tensor whose leading dim is the batch size.

``beam_idx`` is the standard name for the ancestor index tensor in the transformers beam
-search API, and reusing it lets tiers 1 and 2 work unchanged.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch


def to_beam_idx(ancestors: Sequence[int], device: "torch.device | str" = "cpu") -> torch.Tensor:
    """Build the LongTensor of ancestor indices used to reindex the cache."""
    if isinstance(ancestors, torch.Tensor):
        return ancestors.to(device=device, dtype=torch.long)
    return torch.as_tensor(np.asarray(ancestors, dtype=np.int64), dtype=torch.long, device=device)


def _first_tensor_device(obj: Any) -> "torch.device":
    """Find the device of the first tensor inside a nested cache structure."""
    if isinstance(obj, torch.Tensor):
        return obj.device
    if isinstance(obj, (tuple, list)):
        for item in obj:
            dev = _first_tensor_device(item)
            if dev is not None:
                return dev
    if isinstance(obj, dict):
        for item in obj.values():
            dev = _first_tensor_device(item)
            if dev is not None:
                return dev
    for attr in ("key_cache", "value_cache", "layers"):
        if hasattr(obj, attr):
            dev = _first_tensor_device(getattr(obj, attr))
            if dev is not None:
                return dev
    return None  # type: ignore[return-value]


def _recursive_reorder(obj: Any, beam_idx: torch.Tensor, n: int) -> Any:
    """Tier 3: reindex every batch-leading tensor in a nested container."""
    if isinstance(obj, torch.Tensor):
        if obj.dim() >= 1 and obj.shape[0] == n:
            return obj.index_select(0, beam_idx.to(obj.device))
        return obj
    if isinstance(obj, tuple):
        return tuple(_recursive_reorder(o, beam_idx, n) for o in obj)
    if isinstance(obj, list):
        return [_recursive_reorder(o, beam_idx, n) for o in obj]
    if isinstance(obj, dict):
        return {k: _recursive_reorder(v, beam_idx, n) for k, v in obj.items()}

    # transformers DynamicCache (<= mid-2025): parallel key_cache / value_cache lists.
    if hasattr(obj, "key_cache") and hasattr(obj, "value_cache"):
        obj.key_cache = [_recursive_reorder(t, beam_idx, n) for t in obj.key_cache]
        obj.value_cache = [_recursive_reorder(t, beam_idx, n) for t in obj.value_cache]
        return obj

    # transformers Cache with a `.layers` list of per-layer key/value holders.
    if hasattr(obj, "layers"):
        for layer in obj.layers:
            for attr in ("keys", "values", "key_cache", "value_cache"):
                if hasattr(layer, attr) and getattr(layer, attr) is not None:
                    setattr(layer, attr, _recursive_reorder(getattr(layer, attr), beam_idx, n))
        return obj

    return obj


def reorder_cache(past_key_values: Any, ancestors: Sequence[int], model: Any = None) -> Any:
    """Reorder a KV cache so particle ``i`` carries ancestor ``ancestors[i]``'s state.

    Parameters
    ----------
    past_key_values : Any
        A transformers cache: a legacy tuple-of-tuples of tensors, a ``DynamicCache``, a
        newer ``Cache`` object, or any nested container of tensors.
    ancestors : sequence of int
        Ancestor index per particle (length N), from ``systematic_resample``.
    model : optional
        The model, used only to try its ``_reorder_cache`` hook first.
    """
    if past_key_values is None:
        return None

    device = _first_tensor_device(past_key_values) or torch.device("cpu")
    beam_idx = to_beam_idx(ancestors, device)
    n = int(beam_idx.shape[0])

    # Tier 1: model-provided hook (the beam-search reordering API).
    if model is not None and hasattr(model, "_reorder_cache"):
        try:
            return model._reorder_cache(past_key_values, beam_idx)
        except (NotImplementedError, AttributeError, TypeError):
            pass

    # Tier 2: cache object's own reorder method.
    for method_name in ("reorder_cache", "batch_select_indices"):
        method = getattr(past_key_values, method_name, None)
        if callable(method):
            try:
                result = method(beam_idx)
                return result if result is not None else past_key_values
            except (NotImplementedError, AttributeError, TypeError):
                pass

    # Tier 3: recursive reindex.
    return _recursive_reorder(past_key_values, beam_idx, n)
