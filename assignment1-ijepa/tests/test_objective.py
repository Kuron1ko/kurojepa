import copy

import torch
from torch import nn

from .adapters import ema_update, jepa_loss, momentum_at


def test_loss_sum_within_block_mean_over_blocks_and_batch():
    full_target = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]]])
    blocks = [torch.zeros(2, 1, 2, requires_grad=True), torch.zeros(2, 2, 2, requires_grad=True)]
    indices = [torch.tensor([1]), torch.tensor([1, 2])]
    actual = jepa_loss(blocks, full_target, indices)
    expected = torch.stack(
        [
            ((full_target[b, [1]] ** 2).sum() + (full_target[b, [1, 2]] ** 2).sum()) / 2
            for b in range(2)
        ]
    ).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(x.grad is not None and x.grad.abs().sum() > 0 for x in blocks)


def test_loss_does_not_backpropagate_into_target():
    target = torch.randn(2, 3, 4, requires_grad=True)
    predictions = [torch.randn(2, 2, 4, requires_grad=True)]
    loss = jepa_loss(predictions, target, [torch.tensor([0, 2])])
    loss.backward()
    assert predictions[0].grad is not None
    assert target.grad is None


def test_ema_updates_only_target_parameters():
    student = nn.Sequential(nn.Linear(2, 3), nn.LayerNorm(3))
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        for p in student.parameters():
            p.add_(2.0)
    before_student = [p.detach().clone() for p in student.parameters()]
    before_teacher = [p.detach().clone() for p in teacher.parameters()]
    ema_update(student, teacher, 0.75)
    for s, t, old_s, old_t in zip(student.parameters(), teacher.parameters(), before_student, before_teacher):
        torch.testing.assert_close(s, old_s)
        torch.testing.assert_close(t, 0.75 * old_t + 0.25 * old_s)


def test_momentum_schedule_endpoints_and_midpoint():
    assert abs(momentum_at(0, 100, 0.9) - 0.9) < 1e-12
    assert abs(momentum_at(50, 100, 0.9) - 0.95) < 1e-12
    assert abs(momentum_at(100, 100, 0.9) - 1.0) < 1e-12
