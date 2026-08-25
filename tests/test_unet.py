"""Shape and gradient-flow tests for the multi-task U-Net.

These are architecture sanity tests, not accuracy tests -- there is no real
data yet. The goal is to catch shape mismatches, broken skip connections, and
heads that don't receive gradient, all of which are silent failures that would
otherwise only surface after burning real GPU-hours in P1.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from filament.models.unet import FilamentUNet, ModelOutput  # noqa: E402


@pytest.mark.parametrize("size", [64, 96, 128])
def test_output_shapes_match_input_resolution(size):
    model = FilamentUNet(in_channels=3, base_channels=8)
    x = torch.randn(2, 3, size, size)
    out = model(x)
    assert isinstance(out, ModelOutput)
    assert out.mask_logits.shape == (2, 1, size, size)
    assert out.spine_logits.shape == (2, 1, size, size)
    assert out.offsets.shape == (2, 2, size, size)


def test_handles_non_power_of_two_input():
    """Real GONG frames are square (2048x2048) but downstream tiling or
    resizing may not always hit a clean power of two -- the Up block's
    crop-to-match guard must handle that without raising."""
    model = FilamentUNet(in_channels=1, base_channels=8)
    x = torch.randn(1, 1, 100, 84)
    out = model(x)
    assert out.mask_logits.shape == (1, 1, 100, 84)
    assert out.offsets.shape == (1, 2, 100, 84)


def test_single_channel_input():
    """Grayscale-only path (no geometry channels yet) must work standalone."""
    model = FilamentUNet(in_channels=1, base_channels=8)
    out = model(torch.randn(1, 1, 64, 64))
    assert out.mask_logits.shape == (1, 1, 64, 64)


def test_extra_geometry_channels():
    """The Edge 3 use case: intensity + r/R_sun + longitude = 3 channels."""
    model = FilamentUNet(in_channels=3, base_channels=8)
    out = model(torch.randn(1, 3, 64, 64))
    assert out.mask_logits.shape == (1, 1, 64, 64)


def test_gradient_reaches_all_three_heads():
    """Each head must receive gradient from its own loss -- a broken skip or
    a detached tensor would silently starve one head while training proceeds
    normally on the others."""
    model = FilamentUNet(in_channels=1, base_channels=8)
    x = torch.randn(1, 1, 64, 64, requires_grad=True)
    out = model(x)

    loss = out.mask_logits.sum() + out.spine_logits.sum() + out.offsets.sum()
    loss.backward()

    for name, head in [
        ("mask_head", model.mask_head),
        ("spine_head", model.spine_head),
        ("offset_head", model.offset_head),
    ]:
        grad = head.weight.grad
        assert grad is not None, f"{name} received no gradient"
        assert torch.any(grad != 0), f"{name} gradient is all zero"


def test_gradient_reaches_stem():
    """A broken bottleneck or skip connection could isolate the stem from the
    loss entirely; confirm gradient actually flows the full depth."""
    model = FilamentUNet(in_channels=1, base_channels=8)
    out = model(torch.randn(1, 1, 64, 64))
    out.mask_logits.sum().backward()
    grad = model.stem.net[0].weight.grad
    assert grad is not None
    assert torch.any(grad != 0)


def test_deterministic_given_fixed_seed():
    torch.manual_seed(0)
    model1 = FilamentUNet(in_channels=1, base_channels=8)
    torch.manual_seed(0)
    model2 = FilamentUNet(in_channels=1, base_channels=8)

    x = torch.randn(1, 1, 64, 64)
    model1.eval()
    model2.eval()
    out1 = model1(x)
    out2 = model2(x)
    assert torch.allclose(out1.mask_logits, out2.mask_logits)


def test_batch_independence():
    """Processing a batch of 2 must give identical results to processing each
    item alone -- catches accidental cross-batch leakage (e.g. via BatchNorm,
    which this model deliberately avoids in favour of GroupNorm)."""
    model = FilamentUNet(in_channels=1, base_channels=8)
    model.eval()
    x1 = torch.randn(1, 1, 64, 64)
    x2 = torch.randn(1, 1, 64, 64)
    batched = torch.cat([x1, x2], dim=0)

    with torch.no_grad():
        out_batched = model(batched)
        out1 = model(x1)
        out2 = model(x2)

    assert torch.allclose(out_batched.mask_logits[0], out1.mask_logits[0], atol=1e-5)
    assert torch.allclose(out_batched.mask_logits[1], out2.mask_logits[0], atol=1e-5)


def test_offset_head_is_unbounded():
    """Offsets must not be squashed through a bounding nonlinearity -- verify
    the raw output can exceed [-1, 1], since the loss expects raw displacement
    components (documented, deliberate design choice in unet.py)."""
    model = FilamentUNet(in_channels=1, base_channels=8)
    # Push the offset head's bias far outside a typical bounded range.
    with torch.no_grad():
        model.offset_head.bias.fill_(5.0)
    out = model(torch.randn(1, 1, 32, 32))
    assert out.offsets.max().item() > 1.0
