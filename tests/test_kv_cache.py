import numpy as np
import torch

from power_smc.kv_cache import reorder_cache, to_beam_idx


ANCESTORS = [3, 3, 0, 1]
N = 4


def _batched(fill):
    # shape [batch, heads, seq, dim] with each batch row filled by its index
    return fill.view(N, 1, 1, 1).repeat(1, 2, 5, 3)


def test_legacy_tuple_of_tuples():
    fill = torch.arange(N).float()
    legacy = tuple((_batched(fill), _batched(fill)) for _ in range(2))
    out = reorder_cache(legacy, ANCESTORS)
    assert out[0][0][:, 0, 0, 0].tolist() == [3, 3, 0, 1]
    assert out[1][1][:, 0, 0, 0].tolist() == [3, 3, 0, 1]


def test_dynamic_cache_like_object():
    fill = torch.arange(N).float()

    class DynCache:
        def __init__(self):
            self.key_cache = [_batched(fill) for _ in range(2)]
            self.value_cache = [_batched(fill) for _ in range(2)]

    out = reorder_cache(DynCache(), ANCESTORS)
    assert out.key_cache[0][:, 0, 0, 0].tolist() == [3, 3, 0, 1]
    assert out.value_cache[1][:, 0, 0, 0].tolist() == [3, 3, 0, 1]


def test_cache_with_reorder_method_is_used():
    class HookCache:
        def __init__(self):
            self.t = torch.arange(N).float()
            self.called = False

        def reorder_cache(self, beam_idx):
            self.t = self.t.index_select(0, beam_idx)
            self.called = True

    cache = HookCache()
    out = reorder_cache(cache, ANCESTORS)
    assert out.called
    assert out.t.tolist() == [3, 3, 0, 1]


def test_layers_style_cache():
    fill = torch.arange(N).float()

    class Layer:
        def __init__(self):
            self.keys = fill.view(N, 1).clone()
            self.values = fill.view(N, 1).clone()

    class LayerCache:
        def __init__(self):
            self.layers = [Layer(), Layer()]

    out = reorder_cache(LayerCache(), ANCESTORS)
    assert out.layers[0].keys[:, 0].tolist() == [3, 3, 0, 1]


def test_model_hook_is_preferred():
    fill = torch.arange(N).float()
    legacy = tuple((_batched(fill), _batched(fill)) for _ in range(2))

    class Model:
        def __init__(self):
            self.called = False

        def _reorder_cache(self, past, beam_idx):
            self.called = True
            return tuple(tuple(t.index_select(0, beam_idx) for t in layer) for layer in past)

    model = Model()
    out = reorder_cache(legacy, ANCESTORS, model=model)
    assert model.called
    assert out[0][0][:, 0, 0, 0].tolist() == [3, 3, 0, 1]


def test_none_cache_is_passthrough():
    assert reorder_cache(None, ANCESTORS) is None


def test_to_beam_idx_accepts_numpy_and_tensor():
    a = to_beam_idx(np.array([2, 0, 1]))
    b = to_beam_idx(torch.tensor([2, 0, 1]))
    assert a.dtype == torch.long and b.dtype == torch.long
    assert a.tolist() == [2, 0, 1]
