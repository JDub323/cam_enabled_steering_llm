from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Steerer(ABC):
    """Interface for residual-stream steering.

    Implementations receive only the batched last-token activation with shape
    [batch, hidden_size] and must return a batched steering vector of the same
    shape. No layer index, tokens, or metadata are passed by design.
    """

    @abstractmethod
    def get_steering_vector(self, activation: torch.Tensor) -> torch.Tensor:
        """Return a batched steering vector with activation.shape."""
        raise NotImplementedError


class ZeroSteerer(Steerer):
    """No-op steerer useful for baselines and tests."""

    def get_steering_vector(self, activation: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(activation)


class ConstantSteerer(Steerer):
    """Returns the same batched vector for every call.

    vector must be 2-D: [1, hidden_size] or [batch, hidden_size]. A [hidden_size]
    vector is intentionally rejected so all steerers obey the batched contract.
    """

    def __init__(self, vector: torch.Tensor):
        if vector.ndim != 2:
            raise ValueError("ConstantSteerer vector must be batched: [batch, hidden_size].")
        self.vector = vector

    def get_steering_vector(self, activation: torch.Tensor) -> torch.Tensor:
        if self.vector.shape[1] != activation.shape[1]:
            raise ValueError(
                f"Steering width {self.vector.shape[1]} != activation width {activation.shape[1]}."
            )
        if self.vector.shape[0] not in (1, activation.shape[0]):
            raise ValueError(
                f"Steering batch {self.vector.shape[0]} must be 1 or {activation.shape[0]}."
            )
        return self.vector.to(device=activation.device, dtype=activation.dtype).expand_as(activation)
