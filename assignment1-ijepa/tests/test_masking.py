import torch

from .adapters import build_context_indices, gather_tokens, patchify, sample_masks


def test_patchify_row_major_and_channel_order():
    x = torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(1, 2, 4, 6)
    actual = patchify(x, 2)
    expected = torch.stack(
        [x[0, :, r : r + 2, c : c + 2].reshape(-1) for r in (0, 2) for c in (0, 2, 4)]
    ).unsqueeze(0)
    assert actual.shape == (1, 6, 8)
    torch.testing.assert_close(actual, expected)


def test_gather_tokens_shared_and_per_image_indices():
    tokens = torch.arange(2 * 6 * 3).reshape(2, 6, 3)
    torch.testing.assert_close(gather_tokens(tokens, torch.tensor([5, 0, 3])), tokens[:, [5, 0, 3]])
    indices = torch.tensor([[5, 1], [0, 4]])
    expected = torch.stack((tokens[0, [5, 1]], tokens[1, [0, 4]]))
    torch.testing.assert_close(gather_tokens(tokens, indices), expected)


def test_context_geometry_removes_union_not_only_first_target():
    targets = [torch.tensor([1, 5, 6]), torch.tensor([6, 9, 10])]
    actual = build_context_indices((4, 4), (0, 0, 3, 4), targets)
    torch.testing.assert_close(actual, torch.tensor([0, 2, 3, 4, 7, 8, 11]))


def _is_rectangle(indices: torch.Tensor, width: int) -> bool:
    rows = indices // width
    cols = indices % width
    unique_rows = torch.unique(rows)
    unique_cols = torch.unique(cols)
    expected = torch.tensor(
        [r * width + c for r in unique_rows.tolist() for c in unique_cols.tolist()],
        dtype=torch.long,
    )
    return bool(torch.equal(indices.cpu(), expected))


def test_sample_masks_invariants_and_reproducibility():
    kwargs = dict(grid_size=(16, 16), num_targets=4)
    context, targets = sample_masks(generator=torch.Generator().manual_seed(101), **kwargs)
    again_context, again_targets = sample_masks(generator=torch.Generator().manual_seed(101), **kwargs)
    torch.testing.assert_close(context, again_context)
    assert len(targets) == len(again_targets) == 4
    assert context.ndim == 1 and context.dtype == torch.long and context.numel() > 0
    assert torch.equal(context, torch.unique(context, sorted=True))
    all_target_indices = []
    for target, again in zip(targets, again_targets):
        torch.testing.assert_close(target, again)
        assert target.ndim == 1 and target.dtype == torch.long and target.numel() > 0
        assert torch.equal(target, torch.unique(target, sorted=True))
        assert 0 <= int(target.min()) and int(target.max()) < 256
        assert _is_rectangle(target, 16)
        assert 30 <= target.numel() <= 64  # tolerance for discrete grid rounding
        all_target_indices.extend(target.tolist())
    assert 0 <= int(context.min()) and int(context.max()) < 256
    assert set(context.tolist()).isdisjoint(all_target_indices)
    other_context, other_targets = sample_masks(generator=torch.Generator().manual_seed(102), **kwargs)
    assert not (torch.equal(context, other_context) and all(torch.equal(a, b) for a, b in zip(targets, other_targets)))
