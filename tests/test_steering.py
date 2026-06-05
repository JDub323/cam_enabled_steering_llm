from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gemma2_steering import ConstantSteerer, SteeredGemma2, Steerer, ZeroSteerer


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Identity()
        self.mlp = nn.Identity()

    def forward(self, x):
        return (x + 1.0,)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(layers=nn.ModuleList([ToyLayer()]))
        self.config = SimpleNamespace(hidden_size=3)
        self.p = nn.Parameter(torch.zeros(()))

    def forward(self, inputs_embeds, **kwargs):
        (hidden,) = self.model.layers[0](inputs_embeds)
        return SimpleNamespace(last_hidden_state=hidden)

    def generate(self, *args, **kwargs):
        return torch.tensor([[1, 2, 3]])


class BadShapeSteerer(Steerer):
    def get_steering_vector(self, activation):
        return torch.zeros(activation.shape[-1])


def test_zero_steerer_is_noop_after_layer_output():
    toy = ToyModel()
    wrapped = SteeredGemma2(ZeroSteerer(), model=toy, layer=0)
    x = torch.zeros(2, 4, 3)
    out = wrapped(inputs_embeds=x).last_hidden_state
    assert torch.allclose(out, torch.ones_like(x))


def test_constant_steerer_changes_only_last_token():
    toy = ToyModel()
    wrapped = SteeredGemma2(ConstantSteerer(torch.tensor([[10.0, 20.0, 30.0]])), model=toy)
    out = wrapped(inputs_embeds=torch.zeros(2, 4, 3)).last_hidden_state
    assert torch.allclose(out[:, :-1, :], torch.ones(2, 3, 3))
    assert torch.allclose(out[:, -1, :], torch.tensor([[11.0, 21.0, 31.0]]).expand(2, 3))


def test_without_steering_temporarily_disables_hook():
    wrapped = SteeredGemma2(ConstantSteerer(torch.tensor([[2.0, 2.0, 2.0]])), model=ToyModel())
    x = torch.zeros(1, 2, 3)
    assert torch.allclose(wrapped(inputs_embeds=x).last_hidden_state[:, -1, :], torch.tensor([[3.0, 3.0, 3.0]]))
    with wrapped.without_steering():
        assert torch.allclose(wrapped(inputs_embeds=x).last_hidden_state[:, -1, :], torch.tensor([[1.0, 1.0, 1.0]]))
    assert wrapped.enabled is True


def test_invalid_steerer_rejected():
    with pytest.raises(TypeError):
        SteeredGemma2(object(), model=ToyModel())


def test_unbatched_constant_vector_rejected():
    with pytest.raises(ValueError):
        ConstantSteerer(torch.zeros(3))


def test_steerer_must_return_batched_shape():
    wrapped = SteeredGemma2(BadShapeSteerer(), model=ToyModel())
    with pytest.raises(ValueError, match="Steering vector must have shape"):
        wrapped(inputs_embeds=torch.zeros(2, 4, 3))


def test_can_move_hook_to_post_mlp():
    wrapped = SteeredGemma2(ConstantSteerer(torch.ones(1, 3)), model=ToyModel())
    wrapped.attach_hook(layer=0, hook_point="post_mlp")
    assert wrapped.hook_point == "post_mlp"
