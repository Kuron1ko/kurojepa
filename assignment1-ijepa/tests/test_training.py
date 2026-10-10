import copy

import torch
from torch import nn

from .adapters import encode_for_probe, load_checkpoint, pretrain_step, save_checkpoint


class TinyEncoder(nn.Module):
    """A small test double; an implementation must not depend on a particular ViT class."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.last_indices = "unset"

    def forward(self, images, indices=None):
        self.last_indices = indices
        tokens = images.flatten(2).transpose(1, 2) * self.weight
        return tokens if indices is None else tokens[:, indices]


class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, context_tokens, context_indices, target_indices):
        return self.weight * context_tokens.mean(dim=1, keepdim=True).expand(-1, len(target_indices), -1)


def _components():
    context = TinyEncoder()
    target = copy.deepcopy(context)
    target.requires_grad_(False)
    predictor = TinyPredictor()
    optimizer = torch.optim.SGD(list(context.parameters()) + list(predictor.parameters()), lr=0.1)
    return context, target, predictor, optimizer


def test_pretrain_step_freezes_target_and_ema_uses_updated_context():
    context, target, predictor, optimizer = _components()
    images = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    before_context = context.weight.detach().clone()
    before_target = target.weight.detach().clone()
    before_predictor = predictor.weight.detach().clone()
    loss = pretrain_step(
        images, torch.tensor([0, 1]), [torch.tensor([2]), torch.tensor([3])],
        context, target, predictor, optimizer, 0.9,
    )
    assert isinstance(loss, float) and loss > 0
    assert target.last_indices is None  # target saw all patches
    assert torch.equal(context.last_indices, torch.tensor([0, 1]))
    assert context.weight.grad is not None and predictor.weight.grad is not None
    assert target.weight.grad is None
    assert not torch.equal(context.weight, before_context)
    assert not torch.equal(predictor.weight, before_predictor)
    torch.testing.assert_close(target.weight, 0.9 * before_target + 0.1 * context.weight.detach())


def test_probe_uses_full_target_and_mean_pools_without_gradient():
    target = TinyEncoder()
    images = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]], [[[2.0, 4.0], [6.0, 8.0]]]])
    actual = encode_for_probe(images, target)
    assert actual.shape == (2, 1)
    torch.testing.assert_close(actual, torch.tensor([[2.5], [5.0]]))
    assert target.last_indices is None
    assert not actual.requires_grad


def test_checkpoint_restores_all_modules_optimizer_and_step(tmp_path):
    context, target, predictor, optimizer = _components()
    images = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    pretrain_step(images, torch.tensor([0, 1]), [torch.tensor([2])], context, target, predictor, optimizer, 0.8)
    expected = [copy.deepcopy(x.state_dict()) for x in (context, target, predictor, optimizer)]
    path = tmp_path / "step.pt"
    save_checkpoint(path, context, target, predictor, optimizer, 7)
    with torch.no_grad():
        context.weight.add_(10)
        target.weight.add_(10)
        predictor.weight.add_(10)
    restored = load_checkpoint(path, context, target, predictor, optimizer)
    assert restored == 7
    for obj, state in zip((context, target, predictor, optimizer), expected):
        current = obj.state_dict()
        assert current.keys() == state.keys()
        for key, value in current.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, state[key])
            else:
                assert value == state[key]
