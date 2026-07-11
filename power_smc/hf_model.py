"""A Hugging Face Transformer wrapped as an :class:`~power_smc.smc.SMCModel`.

This is the real-model side of the sampler. It exposes the same three-method interface
the toy model does (``prefill`` / ``decode`` / ``reorder``), so :func:`power_smc.power_smc`
runs unchanged on a 1.5B-4B reasoning model.

Design notes
------------
* Next-token log-probs are returned as numpy arrays so the SMC loop stays framework
  independent. For small particle counts (N ~ 4-16) the per-step host transfer is cheap
  relative to the forward pass.
* Finished particles keep being fed EOS; their forward pass is wasted but harmless. Keep
  N modest on a T4 since memory scales with N x model size (see the README limitations).
* transformers / torch are imported lazily so the toy correctness tests need neither.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .kv_cache import reorder_cache
from .utils import log_softmax


class HFModel:
    """Adapter around a causal LM for Power-SMC.

    Parameters
    ----------
    model_name : str
        Hugging Face model id (e.g. ``"Qwen/Qwen2.5-1.5B-Instruct"``).
    device : str
        ``"cuda"`` or ``"cpu"``.
    dtype : str
        Torch dtype name for the weights (``"float16"``, ``"bfloat16"``, ``"float32"``).
    load_in_4bit : bool
        Load with bitsandbytes 4-bit quantization (recommended on a free T4).
    eos_id : int, optional
        Override the terminal token id. Defaults to the tokenizer's EOS.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "float16",
        load_in_4bit: bool = True,
        eos_id: Optional[int] = None,
        trust_remote_code: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

        model_kwargs = {"trust_remote_code": trust_remote_code}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, "bfloat16"),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = {"": 0} if device == "cuda" else None
        else:
            model_kwargs["torch_dtype"] = getattr(torch, dtype)
            if device == "cuda":
                model_kwargs["device_map"] = {"": 0}

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        if not load_in_4bit and device == "cpu":
            self.model.to("cpu")

        resolved_eos = eos_id if eos_id is not None else self.tokenizer.eos_token_id
        if resolved_eos is None:
            raise ValueError("tokenizer has no eos_token_id; pass eos_id explicitly")
        self.eos_id = int(resolved_eos)
        self.vocab_size = int(self.model.config.vocab_size)

    @classmethod
    def from_model(cls, model, tokenizer=None, eos_id: Optional[int] = None) -> "HFModel":
        """Wrap an already-loaded model and tokenizer without hitting the hub.

        Useful for tests (a tiny in-memory model) and for callers that manage loading
        themselves. ``tokenizer`` may be ``None`` if only the sampling loop is used
        (``prefill``/``decode``/``reorder``), since those need token ids, not text.
        """
        import torch

        self = cls.__new__(cls)
        self.torch = torch
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = str(next(model.parameters()).device)
        self.model_name = getattr(model.config, "_name_or_path", "in-memory")
        resolved_eos = eos_id if eos_id is not None else getattr(tokenizer, "eos_token_id", None)
        if resolved_eos is None:
            raise ValueError("no eos id available; pass eos_id explicitly")
        self.eos_id = int(resolved_eos)
        self.vocab_size = int(model.config.vocab_size)
        return self

    # -- prompt helpers ------------------------------------------------------------
    def encode_chat(self, question: str, system: Optional[str] = None) -> list:
        """Apply the model's chat template and return prompt token ids."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": question})
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer(text, return_tensors=None)["input_ids"]

    def decode_text(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        ids = [int(t) for t in token_ids if int(t) != self.eos_id]
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    # -- SMCModel protocol ---------------------------------------------------------
    def prefill(self, prompt_ids: Sequence[int], n: int):
        torch = self.torch
        ids = torch.as_tensor(prompt_ids, dtype=torch.long, device=self._device())
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        ids = ids.expand(n, -1).contiguous()
        with torch.no_grad():
            out = self.model(input_ids=ids, use_cache=True)
        logits = out.logits[:, -1, :]
        return out.past_key_values, self._to_logprobs(logits)

    def decode(self, state, tokens: np.ndarray):
        torch = self.torch
        toks = torch.as_tensor(np.asarray(tokens), dtype=torch.long, device=self._device())
        toks = toks.unsqueeze(-1)
        with torch.no_grad():
            out = self.model(input_ids=toks, past_key_values=state, use_cache=True)
        logits = out.logits[:, -1, :]
        return out.past_key_values, self._to_logprobs(logits)

    def reorder(self, state, ancestors: np.ndarray):
        return reorder_cache(state, ancestors, model=self.model)

    # -- internals -----------------------------------------------------------------
    def _device(self):
        return next(self.model.parameters()).device

    def _to_logprobs(self, logits) -> np.ndarray:
        arr = logits.float().detach().cpu().numpy()
        return log_softmax(arr, axis=-1)
