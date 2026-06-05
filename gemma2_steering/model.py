from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn

from .steerers import Steerer


class SteeredGemma2(nn.Module):
    """Gemma-2 wrapper that adds a steerer-produced vector at a decoder hook.

    By default the hook is attached to the full decoder layer output, so the
    steerer sees the last token's post-layer residual stream and returns a
    batched vector that is added back to that same last-token position.
    """

    HOOK_POINTS = {"layer_output", "post_attention", "post_mlp"}

    def __init__(
        self,
        steerer: Steerer,
        *,
        model: nn.Module | None = None,
        tokenizer: Any | None = None,
        model_name: str = "google/gemma-2-2b",
        layer: int = 0,
        hook_point: str = "layer_output",
        enabled: bool = True,
        **from_pretrained_kwargs: Any,
    ) -> None:
        super().__init__()
        if not isinstance(steerer, Steerer):
            raise TypeError("steerer must inherit from Steerer.")
        self.steerer, self.enabled = steerer, enabled
        self.model, self.tokenizer = model, tokenizer
        if self.model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.model = AutoModelForCausalLM.from_pretrained(model_name, **from_pretrained_kwargs)
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.layer, self.hook_point, self._hook_handle = layer, hook_point, None
        self.attach_hook(layer=layer, hook_point=hook_point)

    @property
    def config(self) -> Any:
        return self.model.config

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.generate(*args, **kwargs)

    def attach_hook(self, *, layer: int | None = None, hook_point: str | None = None) -> None:
        """Move the steering hook to a layer/hook point."""
        self.remove_hook()
        self.layer = self.layer if layer is None else layer
        self.hook_point = self.hook_point if hook_point is None else hook_point
        if self.hook_point not in self.HOOK_POINTS:
            raise ValueError(f"hook_point must be one of {sorted(self.HOOK_POINTS)}.")
        self._hook_handle = self._hook_module().register_forward_hook(self._steer_hook)

    def remove_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    @contextmanager
    def steering_enabled(self, enabled: bool = True) -> Iterator[None]:
        old = self.enabled
        self.enabled = enabled
        try:
            yield
        finally:
            self.enabled = old

    def without_steering(self) -> Iterator[None]:
        return self.steering_enabled(False)

    def _layers(self) -> Any:
        for root in (self.model, getattr(self.model, "base_model", None)):
            if root is not None and hasattr(root, "model") and hasattr(root.model, "layers"):
                return root.model.layers
        raise AttributeError("Could not find decoder layers at model.model.layers.")

    def _hook_module(self) -> nn.Module:
        block = self._layers()[self.layer]
        if self.hook_point == "layer_output":
            return block
        name = "self_attn" if self.hook_point == "post_attention" else "mlp"
        if not hasattr(block, name):
            raise AttributeError(f"Layer {self.layer} has no {name!r} module.")
        return getattr(block, name)

    def _steer_hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        if not self.enabled:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise TypeError("Steering hook expected hidden states shaped [batch, seq, hidden].")
        activation = hidden[:, -1, :].clone()
        steering = self.steerer.get_steering_vector(activation)
        if steering.shape != activation.shape:
            raise ValueError(
                f"Steering vector must have shape {tuple(activation.shape)}, got {tuple(steering.shape)}."
            )
        steered = hidden.clone()
        steered[:, -1, :] = steered[:, -1, :] + steering.to(device=hidden.device, dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered
