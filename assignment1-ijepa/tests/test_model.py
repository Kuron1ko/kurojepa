import math

import torch

from .adapters import build_encoder, build_predictor, scaled_dot_product_attention


def test_scaled_dot_product_attention_exact_and_noncausal():
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    v = torch.tensor([[[2.0], [8.0]]])
    weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(2), dim=-1)
    actual = scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(actual, weights @ v)
    assert actual[0, 0, 0] > 2.0  # first token attends to the later token


def test_scaled_dot_product_attention_leading_dims_and_gradient():
    torch.manual_seed(1)
    q = torch.randn(2, 3, 4, 5, requires_grad=True)
    k = torch.randn(2, 3, 6, 5, requires_grad=True)
    v = torch.randn(2, 3, 6, 7, requires_grad=True)
    actual = scaled_dot_product_attention(q, k, v)
    expected = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(5), dim=-1) @ v
    assert actual.shape == (2, 3, 4, 7)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))


def _encoder():
    torch.manual_seed(12)
    model = build_encoder((16, 24), 4, 3, 32, 2, 4, 2.0)
    return model.eval()


def test_encoder_full_and_visible_shapes_and_gradients():
    model = _encoder()
    images = torch.randn(2, 3, 16, 24)
    full = model(images)
    visible = model(images, torch.tensor([23, 0, 6, 11]))
    assert full.shape == (2, 24, 32)
    assert visible.shape == (2, 4, 32)
    visible.square().sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_hidden_pixels_cannot_reach_context_encoder():
    model = _encoder()
    indices = torch.tensor([0, 1, 6, 7, 12, 13])
    a = torch.randn(1, 3, 16, 24)
    b = a.clone()
    visible = {(int(j) // 6, int(j) % 6) for j in indices}
    for r in range(4):
        for c in range(6):
            if (r, c) not in visible:
                b[:, :, r * 4 : (r + 1) * 4, c * 4 : (c + 1) * 4] += 100.0
    with torch.no_grad():
        torch.testing.assert_close(model(a, indices), model(b, indices), atol=1e-5, rtol=1e-5)


def test_encoder_uses_patch_positions():
    model = _encoder()
    with torch.no_grad():
        tokens = model(torch.zeros(1, 3, 16, 24))
    assert not torch.allclose(tokens[:, 0], tokens[:, 1])


def test_predictor_shape_order_and_gradient():
    torch.manual_seed(11)
    model = build_predictor(32, 16, 2, 4, 32, 24).eval()
    context = torch.randn(2, 5, 32, requires_grad=True)
    context_indices = torch.tensor([0, 2, 6, 12, 18])
    target_indices = torch.tensor([7, 13, 19])
    predicted = model(context, context_indices, target_indices)
    permuted = model(context, context_indices, target_indices[[2, 0, 1]])
    assert predicted.shape == (2, 3, 32)
    torch.testing.assert_close(permuted, predicted[:, [2, 0, 1]], atol=1e-5, rtol=1e-5)
    predicted.square().sum().backward()
    assert context.grad is not None and context.grad.abs().sum() > 0
    assert any(p.grad is not None for p in model.parameters())


def test_predictor_conditions_on_target_position():
    torch.manual_seed(11)
    model = build_predictor(32, 16, 2, 4, 32, 24).eval()
    context = torch.zeros(1, 5, 32)
    with torch.no_grad():
        output = model(context, torch.tensor([0, 2, 6, 12, 18]), torch.tensor([7, 13]))
    assert not torch.allclose(output[:, 0], output[:, 1])
