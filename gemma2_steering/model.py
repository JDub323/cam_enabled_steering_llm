from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn

from .steerers import Steerer


class SteeredGemma2(nn.Module):
    """Gemma-2 wrapper with separate classify and steer hooks.

    By default classification and steering happen at the same decoder-layer
    hook, preserving the original behavior: the steerer sees the last token's
    post-layer residual stream and the returned batched vector is added back to
    that same last-token position.

    For experiments where the best deception classifier layer differs from the
    best intervention layer, set ``classify_layer``/``classify_hook_point`` and
    ``steer_layer``/``steer_hook_point`` separately. The ``Steerer`` interface
    remains unchanged; the steering vector returned by ``get_steering_vector``
    is cached by the classify hook and consumed by the steer hook later in the
    same forward pass.
    """

    HOOK_POINTS = {"layer_output", "post_attention", "post_mlp"}
    _HOOK_ORDER = {"post_attention": 0, "post_mlp": 1, "layer_output": 2}

    def __init__(
        self,
        steerer: Steerer,
        *,
        model: nn.Module | None = None,
        tokenizer: Any | None = None,
        model_name: str = "google/gemma-2-2b",
        layer: int = 0,
        hook_point: str = "layer_output",
        classify_layer: int | None = None,
        steer_layer: int | None = None,
        classify_hook_point: str | None = None,
        steer_hook_point: str | None = None,
        enabled: bool = True,
        classify_enabled: bool | None = None,
        steer_enabled: bool | None = None,
        **from_pretrained_kwargs: Any,
    ) -> None:
        super().__init__()
        if not isinstance(steerer, Steerer):
            raise TypeError("steerer must inherit from Steerer.")
        self.steerer = steerer
        self._classify_enabled = bool(enabled if classify_enabled is None else classify_enabled)
        self._steer_enabled = bool(enabled if steer_enabled is None else steer_enabled)
        self._cached_steering_vector: torch.Tensor | None = None
        self.model, self.tokenizer = model, tokenizer
        if self.model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.model = AutoModelForCausalLM.from_pretrained(model_name, **from_pretrained_kwargs)
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)

        self.classify_layer = layer if classify_layer is None else classify_layer
        self.steer_layer = layer if steer_layer is None else steer_layer
        self.classify_hook_point = hook_point if classify_hook_point is None else classify_hook_point
        self.steer_hook_point = hook_point if steer_hook_point is None else steer_hook_point
        self.layer = self.steer_layer
        self.hook_point = self.steer_hook_point
        self._hook_handles: list[Any] = []
        self._hook_handle = None  # Backward-compatible alias for same-hook use.
        self.attach_hook(
            classify_layer=self.classify_layer,
            steer_layer=self.steer_layer,
            classify_hook_point=self.classify_hook_point,
            steer_hook_point=self.steer_hook_point,
        )

    @property
    def config(self) -> Any:
        return self.model.config

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def enabled(self) -> bool:
        """Backward-compatible combined enabled state."""
        return self.classify_enabled and self.steer_enabled

    @enabled.setter
    def enabled(self, enabled: bool) -> None:
        self.set_enabled(enabled)

    @property
    def classify_enabled(self) -> bool:
        return self._classify_enabled

    @classify_enabled.setter
    def classify_enabled(self, enabled: bool) -> None:
        self._classify_enabled = bool(enabled)
        if not self._classify_enabled:
            self._cached_steering_vector = None

    @property
    def steer_enabled(self) -> bool:
        return self._steer_enabled

    @steer_enabled.setter
    def steer_enabled(self, enabled: bool) -> None:
        self._steer_enabled = bool(enabled)
        if not self._steer_enabled:
            self._cached_steering_vector = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self._cached_steering_vector = None
        try:
            return self.model(*args, **kwargs)
        finally:
            self._cached_steering_vector = None

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        self._cached_steering_vector = None
        try:
            return self.model.generate(*args, **kwargs)
        finally:
            self._cached_steering_vector = None

    def attach_hook(
        self,
        *,
        layer: int | None = None,
        hook_point: str | None = None,
        classify_layer: int | None = None,
        steer_layer: int | None = None,
        classify_hook_point: str | None = None,
        steer_hook_point: str | None = None,
    ) -> None:
        """Move the classification and steering hooks.

        ``layer`` and ``hook_point`` are retained as backward-compatible aliases
        that move both hooks together. Use ``classify_*`` and ``steer_*`` to
        place the two phases at different layers or hook points.
        """
        new_classify_layer = self.classify_layer if classify_layer is None else classify_layer
        new_steer_layer = self.steer_layer if steer_layer is None else steer_layer
        new_classify_hook_point = self.classify_hook_point if classify_hook_point is None else classify_hook_point
        new_steer_hook_point = self.steer_hook_point if steer_hook_point is None else steer_hook_point

        if layer is not None:
            if classify_layer is None:
                new_classify_layer = layer
            if steer_layer is None:
                new_steer_layer = layer
        if hook_point is not None:
            if classify_hook_point is None:
                new_classify_hook_point = hook_point
            if steer_hook_point is None:
                new_steer_hook_point = hook_point

        self._validate_hook_config(
            classify_layer=new_classify_layer,
            steer_layer=new_steer_layer,
            classify_hook_point=new_classify_hook_point,
            steer_hook_point=new_steer_hook_point,
        )

        self.remove_hook()
        self.classify_layer = new_classify_layer
        self.steer_layer = new_steer_layer
        self.classify_hook_point = new_classify_hook_point
        self.steer_hook_point = new_steer_hook_point
        self.layer = self.steer_layer
        self.hook_point = self.steer_hook_point
        self._cached_steering_vector = None

        if self._same_hook:
            handle = self._hook_module(self.classify_layer, self.classify_hook_point).register_forward_hook(
                self._classify_and_steer_hook
            )
            self._hook_handles = [handle]
            self._hook_handle = handle
        else:
            classify_handle = self._hook_module(
                self.classify_layer, self.classify_hook_point
            ).register_forward_hook(self._classify_hook)
            steer_handle = self._hook_module(self.steer_layer, self.steer_hook_point).register_forward_hook(
                self._steer_hook
            )
            self._hook_handles = [classify_handle, steer_handle]
            self._hook_handle = steer_handle

    def remove_hook(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._hook_handle = None
        self._cached_steering_vector = None

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable both classification and steering."""
        self.classify_enabled = enabled
        self.steer_enabled = enabled

    def set_classification_enabled(self, enabled: bool) -> None:
        self.classify_enabled = enabled

    def set_steering_enabled(self, enabled: bool) -> None:
        self.steer_enabled = enabled

    @contextmanager
    def classification_enabled(self, enabled: bool = True) -> Iterator[None]:
        old = self.classify_enabled
        self.classify_enabled = enabled
        try:
            yield
        finally:
            self.classify_enabled = old

    @contextmanager
    def steering_enabled(self, enabled: bool = True) -> Iterator[None]:
        old = self.steer_enabled
        self.steer_enabled = enabled
        try:
            yield
        finally:
            self.steer_enabled = old

    def without_classification(self) -> Iterator[None]:
        return self.classification_enabled(False)

    def without_steering(self) -> Iterator[None]:
        return self.steering_enabled(False)

    @property
    def _same_hook(self) -> bool:
        return (
            self.classify_layer == self.steer_layer
            and self.classify_hook_point == self.steer_hook_point
        )

    def _validate_hook_config(
        self,
        *,
        classify_layer: int,
        steer_layer: int,
        classify_hook_point: str,
        steer_hook_point: str,
    ) -> None:
        for name, hook_point in (
            ("classify_hook_point", classify_hook_point),
            ("steer_hook_point", steer_hook_point),
        ):
            if hook_point not in self.HOOK_POINTS:
                raise ValueError(f"{name} must be one of {sorted(self.HOOK_POINTS)}.")
        if classify_layer > steer_layer:
            raise ValueError("classify_layer must be less than or equal to steer_layer.")
        if classify_layer == steer_layer and self._HOOK_ORDER[classify_hook_point] > self._HOOK_ORDER[steer_hook_point]:
            raise ValueError(
                "classify_hook_point must run before steer_hook_point when both hooks are on the same layer."
            )
        # Force any layer-index/module errors to happen while attaching hooks,
        # rather than later during a forward pass.
        self._hook_module(classify_layer, classify_hook_point)
        self._hook_module(steer_layer, steer_hook_point)

    def _layers(self) -> Any:
        for root in (self.model, getattr(self.model, "base_model", None)):
            if root is not None and hasattr(root, "model") and hasattr(root.model, "layers"):
                return root.model.layers
        raise AttributeError("Could not find decoder layers at model.model.layers.")

    def _hook_module(self, layer: int, hook_point: str) -> nn.Module:
        block = self._layers()[layer]
        if hook_point == "layer_output":
            return block
        name = "self_attn" if hook_point == "post_attention" else "mlp"
        if not hasattr(block, name):
            raise AttributeError(f"Layer {layer} has no {name!r} module.")
        return getattr(block, name)

    def _classify_hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        if self.classify_enabled:
            self._cached_steering_vector = self._make_steering_vector(output)
        return output

    def _steer_hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        if not self.steer_enabled:
            return output
        steering = self._pop_cached_steering_vector()
        return self._apply_steering_vector(output, steering)

    def _classify_and_steer_hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        if self.classify_enabled:
            self._cached_steering_vector = self._make_steering_vector(output)
        if not self.steer_enabled:
            return output
        steering = self._pop_cached_steering_vector()
        return self._apply_steering_vector(output, steering)

    def _make_steering_vector(self, output: Any) -> torch.Tensor:
        hidden = self._extract_hidden(output)
        activation = hidden[:, -1, :].clone()
        steering = self.steerer.get_steering_vector(activation)
        if not torch.is_tensor(steering):
            raise TypeError("Steerer.get_steering_vector must return a torch.Tensor.")
        if steering.shape != activation.shape:
            raise ValueError(
                f"Steering vector must have shape {tuple(activation.shape)}, got {tuple(steering.shape)}."
            )
        return steering

    def _pop_cached_steering_vector(self) -> torch.Tensor:
        steering = self._cached_steering_vector
        self._cached_steering_vector = None
        if steering is None:
            raise RuntimeError(
                "Steering hook fired before classification produced a steering vector. "
                "Ensure classify_layer/classify_hook_point run before steer_layer/steer_hook_point "
                "and do not disable classification while steering is enabled."
            )
        return steering

    def _apply_steering_vector(self, output: Any, steering: torch.Tensor) -> Any:
        hidden = self._extract_hidden(output)
        target = hidden[:, -1, :]
        if steering.shape != target.shape:
            raise ValueError(
                f"Cached steering vector must have shape {tuple(target.shape)}, got {tuple(steering.shape)}."
            )
        steered = hidden.clone()
        steered[:, -1, :] = target + steering.to(device=hidden.device, dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    @staticmethod
    def _extract_hidden(output: Any) -> torch.Tensor:
        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise TypeError("Steering hook expected hidden states shaped [batch, seq, hidden].")
        return hidden
