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
        x = self.self_attn(x)
        x = self.mlp(x)
        return (x + 1.0,)


class ToyModel(nn.Module):
    def __init__(self, num_layers=1):
        super().__init__()
        self.model = SimpleNamespace(layers=nn.ModuleList([ToyLayer() for _ in range(num_layers)]))
        self.config = SimpleNamespace(hidden_size=3)
        self.p = nn.Parameter(torch.zeros(()))

    def forward(self, inputs_embeds, **kwargs):
        hidden = inputs_embeds
        for layer in self.model.layers:
            (hidden,) = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)

    def generate(self, *args, **kwargs):
        return torch.tensor([[1, 2, 3]])


class BadShapeSteerer(Steerer):
    def get_steering_vector(self, activation):
        return torch.zeros(activation.shape[-1])


class RecordingSteerer(Steerer):
    def __init__(self):
        self.activations = []

    def get_steering_vector(self, activation):
        self.activations.append(activation.detach().clone())
        return 2.0 * activation


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


def test_can_classify_and_steer_at_different_layers():
    steerer = RecordingSteerer()
    wrapped = SteeredGemma2(
        steerer,
        model=ToyModel(num_layers=2),
        classify_layer=0,
        steer_layer=1,
    )
    out = wrapped(inputs_embeds=torch.zeros(1, 2, 3)).last_hidden_state
    assert len(steerer.activations) == 1
    assert torch.allclose(steerer.activations[0], torch.ones(1, 3))
    assert torch.allclose(out[:, :-1, :], torch.full((1, 1, 3), 2.0))
    assert torch.allclose(out[:, -1, :], torch.full((1, 3), 4.0))


def test_same_classify_and_steer_hook_uses_one_handle():
    wrapped = SteeredGemma2(
        ConstantSteerer(torch.tensor([[10.0, 20.0, 30.0]])),
        model=ToyModel(),
        classify_layer=0,
        steer_layer=0,
        classify_hook_point="layer_output",
        steer_hook_point="layer_output",
    )
    out = wrapped(inputs_embeds=torch.zeros(2, 4, 3)).last_hidden_state
    assert len(wrapped._hook_handles) == 1
    assert torch.allclose(out[:, -1, :], torch.tensor([[11.0, 21.0, 31.0]]).expand(2, 3))


def test_classification_can_be_enabled_without_steering():
    steerer = RecordingSteerer()
    wrapped = SteeredGemma2(
        steerer,
        model=ToyModel(num_layers=2),
        classify_layer=0,
        steer_layer=1,
        steer_enabled=False,
    )
    out = wrapped(inputs_embeds=torch.zeros(1, 2, 3)).last_hidden_state
    assert len(steerer.activations) == 1
    assert torch.allclose(out, torch.full((1, 2, 3), 2.0))
    assert wrapped.classify_enabled is True
    assert wrapped.steer_enabled is False
    assert wrapped.enabled is False


def test_steering_without_classification_raises():
    wrapped = SteeredGemma2(
        ConstantSteerer(torch.ones(1, 3)),
        model=ToyModel(num_layers=2),
        classify_layer=0,
        steer_layer=1,
        classify_enabled=False,
        steer_enabled=True,
    )
    with pytest.raises(RuntimeError, match="before classification produced"):
        wrapped(inputs_embeds=torch.zeros(1, 2, 3))


def test_classify_layer_must_not_follow_steer_layer():
    with pytest.raises(ValueError, match="classify_layer"):
        SteeredGemma2(
            ZeroSteerer(),
            model=ToyModel(num_layers=2),
            classify_layer=1,
            steer_layer=0,
        )


def test_classify_hook_point_must_not_follow_steer_hook_point_on_same_layer():
    with pytest.raises(ValueError, match="classify_hook_point"):
        SteeredGemma2(
            ZeroSteerer(),
            model=ToyModel(),
            classify_layer=0,
            steer_layer=0,
            classify_hook_point="layer_output",
            steer_hook_point="post_attention",
        )


def test_without_steering_temporarily_disables_only_steering_hook():
    steerer = RecordingSteerer()
    wrapped = SteeredGemma2(steerer, model=ToyModel(), layer=0)
    x = torch.zeros(1, 2, 3)
    with wrapped.without_steering():
        assert torch.allclose(wrapped(inputs_embeds=x).last_hidden_state[:, -1, :], torch.tensor([[1.0, 1.0, 1.0]]))
    assert len(steerer.activations) == 1
    assert wrapped.classify_enabled is True
    assert wrapped.steer_enabled is True
    assert wrapped.enabled is True


def test_without_classification_errors_if_steering_remains_enabled():
    wrapped = SteeredGemma2(ConstantSteerer(torch.ones(1, 3)), model=ToyModel(), layer=0)
    with wrapped.without_classification():
        with pytest.raises(RuntimeError, match="before classification produced"):
            wrapped(inputs_embeds=torch.zeros(1, 2, 3))


def test_set_enabled_remains_backward_compatible():
    wrapped = SteeredGemma2(ConstantSteerer(torch.tensor([[2.0, 2.0, 2.0]])), model=ToyModel())
    x = torch.zeros(1, 2, 3)
    assert torch.allclose(wrapped(inputs_embeds=x).last_hidden_state[:, -1, :], torch.tensor([[3.0, 3.0, 3.0]]))
    wrapped.set_enabled(False)
    assert torch.allclose(wrapped(inputs_embeds=x).last_hidden_state[:, -1, :], torch.tensor([[1.0, 1.0, 1.0]]))
    assert wrapped.classify_enabled is False
    assert wrapped.steer_enabled is False
    wrapped.set_enabled(True)
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


def test_can_move_hooks_together_with_backward_compatible_aliases():
    wrapped = SteeredGemma2(ConstantSteerer(torch.ones(1, 3)), model=ToyModel())
    wrapped.attach_hook(layer=0, hook_point="post_mlp")
    assert wrapped.classify_layer == 0
    assert wrapped.steer_layer == 0
    assert wrapped.classify_hook_point == "post_mlp"
    assert wrapped.steer_hook_point == "post_mlp"
    assert wrapped.layer == 0
    assert wrapped.hook_point == "post_mlp"
