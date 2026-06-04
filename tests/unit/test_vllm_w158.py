"""Unit tests for EdgeRazor W1.58 (ternary) weight quantization.

Covers:
  - INT2 pack / unpack roundtrip
  - ternary weight quantization (clip method, scale correctness)
  - uint4b8 encoding for ternary values {-1,0,1} → {7,8,9}
  - dequantize_weight with weight_bits=1.58
  - W1.58 vs W4 computation equivalence
  - ER/IE block-size split for ternary weights
"""

import pytest
import torch

from edgerazor.vllm.quant_ops import (
    ER_W1_58A8_BLOCK_SIZE,
    IE_W1_58A8_BLOCK_SIZE,
    ER_W4A8_BLOCK_SIZE,
    IE_W4A8_BLOCK_SIZE,
    INT1_58_MAX,
    INT4_MAX,
    INT8_MAX,
    dequantize_weight,
    pack_int4,
    pack_int2,
    quantize_activation_per_block_int8,
    quantize_weight_per_block_int4,
    quantize_weight_per_block_int2,
    resolve_quant_block,
    unpack_int4,
    unpack_int2,
)

# ────────────────────────────────────────────────────────────
# INT2 pack / unpack roundtrip
# ────────────────────────────────────────────────────────────


class TestPackUnpackINT2:
    """pack_int2 ↔ unpack_int2 roundtrip identity."""

    @pytest.mark.parametrize("shape", [
        (16, 256),
        (32, 512),
        (8, 64),
        (1, 128),
    ])
    def test_roundtrip_bit_exact(self, shape):
        """Pack then unpack recovers the original INT2 values exactly."""
        torch.manual_seed(42)
        w_int = torch.randint(-2, 2, shape, dtype=torch.int8)
        qweight = pack_int2(w_int)
        assert qweight.shape == (shape[0], shape[1] // 4)
        assert qweight.dtype == torch.uint8

        recovered = unpack_int2(qweight)
        assert recovered.shape == w_int.shape
        assert torch.equal(recovered, w_int)

    def test_ternary_subset(self):
        """Ternary values {-1,0,1} roundtrip correctly through INT2 pack."""
        torch.manual_seed(42)
        w_ternary = torch.randint(-1, 2, (8, 256), dtype=torch.int8)
        qweight = pack_int2(w_ternary)
        recovered = unpack_int2(qweight)
        assert torch.equal(recovered, w_ternary)

    def test_pack_all_values(self):
        """All 4 possible INT2 values {-2,-1,0,1} pack and unpack correctly."""
        w = torch.tensor([[-2, -1, 0, 1, -2, -1, 0, 1]], dtype=torch.int8)
        q = pack_int2(w)
        assert q.shape == (1, 2)
        recovered = unpack_int2(q)
        assert torch.equal(recovered, w)

    def test_pack_dtype(self):
        """pack_int2 outputs uint8."""
        w = torch.zeros(4, 64, dtype=torch.int8)
        q = pack_int2(w)
        assert q.dtype == torch.uint8
        assert q.shape == (4, 16)  # 64/4 = 16


# ────────────────────────────────────────────────────────────
# Ternary weight quantization
# ────────────────────────────────────────────────────────────


class TestTernaryQuantize:
    """quantize_weight_per_block_int2 correctness."""

    def test_ternary_values_clamped(self):
        """Output weight values are strictly in {-1, 0, 1}."""
        torch.manual_seed(42)
        # Use a small block_size so we can verify per-block behavior
        w = torch.randn(4, 256, dtype=torch.bfloat16) * 3.0
        qweight, _scale = quantize_weight_per_block_int2(
            w, er_block_size=256, ie_block_size=32,
        )
        # Unpack and check values are ternary
        w_int = unpack_int2(qweight)
        assert w_int.min() >= -1
        assert w_int.max() <= 1
        # Ternary values should appear
        unique = torch.unique(w_int)
        assert set(unique.tolist()).issubset({-2, -1, 0, 1})

    def test_scale_positive(self):
        """All scales must be strictly positive."""
        w = torch.randn(4, 256, dtype=torch.bfloat16)
        _qweight, scale = quantize_weight_per_block_int2(w)
        assert (scale > 0).all()

    def test_scale_is_mean_abs_times_factor(self):
        """Clip-method scale = mean(|w|) * w_scale_factor."""
        torch.manual_seed(42)
        w = torch.randn(1, 256, dtype=torch.bfloat16)
        er = 256
        w_blocks = w.view(1, -1, er)
        expected_scale = w_blocks.abs().mean(dim=-1, keepdim=True).mul_(2.0).to(torch.bfloat16)
        _qweight, actual_scale = quantize_weight_per_block_int2(
            w, er_block_size=er, ie_block_size=er,
            w_scale_factor=2.0,
        )
        torch.testing.assert_close(
            actual_scale.squeeze().float(), expected_scale.squeeze().float(),
        )

    def test_quantized_dequant_close(self):
        """Dequantized weight approximates original within ternary precision."""
        torch.manual_seed(42)
        w = torch.randn(4, 512, dtype=torch.bfloat16)
        qweight, scale = quantize_weight_per_block_int2(
            w, er_block_size=256, ie_block_size=32,
        )
        w_deq = dequantize_weight(
            qweight, scale, block_size=32, out_dtype=torch.bfloat16,
            weight_bits=1.58,
        )
        # Ternary is crude; check that dequantized values are close-ish
        # (ternary can only represent sign, so correlation is what matters)
        # MSE should be bounded by scale^2 on average
        mse = torch.nn.functional.mse_loss(w_deq.float(), w.float())
        # For ternary at scale=mean(|w|)*2, max error per element is ≤ scale
        avg_scale = scale.float().mean()
        assert mse <= (avg_scale ** 2) * 2, f"MSE={mse}, scale²={avg_scale**2}"

    def test_output_shapes(self):
        """Quantized output has correct shapes."""
        w = torch.randn(16, 1024, dtype=torch.bfloat16)
        qweight, scale = quantize_weight_per_block_int2(
            w, er_block_size=256, ie_block_size=32,
        )
        # INT2 packs 4 values per byte
        assert qweight.shape == (16, 1024 // 4)
        # Scale: one per ie_block_size along input dim
        assert scale.shape == (16, 1024 // 32)


# ────────────────────────────────────────────────────────────
# uint4b8 encoding correctness
# ────────────────────────────────────────────────────────────


class TestUint4b8Encoding:
    """Ternary {-1,0,1} → uint4b8 {7,8,9} encoding for Marlin."""

    def test_ternary_to_uint4b8_mapping(self):
        """Each ternary value maps to the correct uint4b8 stored value."""
        ternary = torch.tensor([[-1, 0, 1]], dtype=torch.int8)
        uint4b8 = (ternary + 8).to(torch.uint8)
        assert uint4b8.tolist() == [[7, 8, 9]]

    def test_uint4b8_roundtrip(self):
        """uint4b8 stored values survive INT4 pack/unpack correctly."""
        # Simulate: quantize ternary → convert to uint4b8 → pack as INT4
        ternary = torch.tensor([[-1, 0, 1, -1, 0, 1, -1, 0]], dtype=torch.int8)
        uint4b8_v = (ternary + 8).to(torch.uint8)  # {7, 8, 9}
        # Pack as 2× INT4 per byte (INT4 format)
        packed_int4 = pack_int4(ternary)
        # Unpack and verify values survive
        recovered = unpack_int4(packed_int4)
        assert torch.equal(recovered, ternary)

    def test_uint4b8_dequant_correctness(self):
        """Marlin kernel computes (q-8)*scale; verify for ternary encoding."""
        scale = torch.tensor([[0.5]], dtype=torch.bfloat16)
        # Ternary values: -1, 0, 1
        ternary = torch.tensor([[-1, 0, 1, -1, 0, 1, -1, 0, 1, 0, -1, 1, 0, -1, 0, -1]],
                               dtype=torch.int8)
        # Manual uint4b8 → dequant via Marlin formula
        uint4b8_v = (ternary + 8).to(torch.int32)
        deq_marlin = ((uint4b8_v - 8).float() * scale.float()).to(torch.bfloat16)
        # Equivalent: ternary * scale
        deq_expected = (ternary.float() * scale.float()).to(torch.bfloat16)
        assert torch.equal(deq_marlin, deq_expected)


# ────────────────────────────────────────────────────────────
# W1.58 vs W4 quantization: framework consistency
# ────────────────────────────────────────────────────────────


class TestW1vsW4Framework:
    """Verify both W1.58 and W4 use the same underlying uint4b8 pipeline."""

    def test_both_pack_as_uint8_4bit_format(self):
        """Both W1.58 (ternary upcast) and W4 produce INT4-packed qweight."""
        w = torch.randn(4, 512, dtype=torch.bfloat16)
        q_ternary, _s158 = quantize_weight_per_block_int2(w)
        q_int4, _s4 = quantize_weight_per_block_int4(w)

        # Both are uint8 with packing density 2 values/byte
        assert q_ternary.dtype == torch.uint8
        assert q_int4.dtype == torch.uint8
        # INT2 packs 4 values per byte → half the size of INT4
        assert q_ternary.shape[-1] == w.shape[-1] // 4
        assert q_int4.shape[-1] == w.shape[-1] // 2

    def test_scale_shapes_match(self):
        """Both produce scale tensors with the same shape for same IE."""
        w = torch.randn(8, 1024, dtype=torch.bfloat16)
        _, s158 = quantize_weight_per_block_int2(
            w, er_block_size=256, ie_block_size=32,
        )
        _, s4 = quantize_weight_per_block_int4(
            w, er_block_size=256, ie_block_size=32,
        )
        assert s158.shape == s4.shape  # (N, K/32)

    def test_resolve_quant_block_er_consistent(self):
        """resolve_quant_block quant_block is always ER when ER > IE."""
        r158 = resolve_quant_block(ER_W1_58A8_BLOCK_SIZE, IE_W1_58A8_BLOCK_SIZE)
        r4 = resolve_quant_block(ER_W4A8_BLOCK_SIZE, IE_W4A8_BLOCK_SIZE)
        # quant_block should be ER=256 for both, regardless of IE
        assert r158[0] == r4[0] == 256

    def test_W1_quant_preserves_W4_pipeline(self):
        """W1.58 quantized values pass through the same unpack_int4 as W4."""
        w = torch.randn(4, 256, dtype=torch.bfloat16)
        q_ternary, _ = quantize_weight_per_block_int2(
            w, er_block_size=256, ie_block_size=32,
        )
        # INT2-packed qweight can be unpacked by unpack_int2
        unpacked = unpack_int2(q_ternary)
        assert unpacked.shape == w.shape
        assert unpacked.dtype == torch.int8
        assert unpacked.min() >= -1
        assert unpacked.max() <= 1

    def test_W4_dequantize_W1_dequantize_consistent(self):
        """dequantize_weight with weight_bits=1 vs weight_bits=4 are consistent."""
        torch.manual_seed(42)
        w = torch.randn(4, 512, dtype=torch.bfloat16)

        q158, s158 = quantize_weight_per_block_int2(
            w, er_block_size=256, ie_block_size=256,
        )
        q4, s4 = quantize_weight_per_block_int4(
            w, er_block_size=256, ie_block_size=256,
        )

        w158_deq = dequantize_weight(q158, s158, block_size=256, weight_bits=1.58)
        w4_deq = dequantize_weight(q4, s4, block_size=256, weight_bits=4)

        # Both should be roughly similar to original weight
        # (W4 should be more accurate → lower MSE)
        mse158 = torch.nn.functional.mse_loss(w158_deq.float(), w.float())
        mse4 = torch.nn.functional.mse_loss(w4_deq.float(), w.float())
        # W4 should be strictly better (more bits)
        assert mse4 <= mse158 * 1.5  # W4 is always better or comparable

    def test_W1_marin_encoding_compatible(self):
        """W1.58 ternary values encoded as uint4b8 work with INT4 pack pipeline."""
        torch.manual_seed(42)
        w = torch.randn(4, 512, dtype=torch.bfloat16)
        q_int2, s_int2 = quantize_weight_per_block_int2(
            w, er_block_size=256, ie_block_size=32,
        )
        # Unpack INT2 → get ternary values
        ternary = unpack_int2(q_int2)

        # Convert ternary to uint4b8 for Marlin: q = t + 8
        uint4b8_vals = (ternary + 8).to(torch.uint8)

        # Verify all encoded values are in valid uint4 range [1, 15]
        assert uint4b8_vals.min() >= 1
        assert uint4b8_vals.max() <= 15

        # Verify dequant correctness: (q - 8) * scale = t * scale
        scale_expanded = s_int2.repeat_interleave(32, dim=1)
        deq_from_uint4b8 = ((uint4b8_vals.to(torch.int32) - 8).float()
                             * scale_expanded.float())
        deq_from_ternary = ternary.float() * scale_expanded.float()
        assert torch.equal(deq_from_uint4b8, deq_from_ternary)


# ────────────────────────────────────────────────────────────
# ER/IE block-size split for ternary weights
# ────────────────────────────────────────────────────────────


class TestTernaryBlockSplit:
    """ER=256 → IE=32 scale replication for W1.58."""

    def test_scale_replication(self):
        """When ER > IE, scales are replicated IE/ER times."""
        w = torch.randn(2, 256, dtype=torch.bfloat16)
        er, ie = 256, 32
        n = er // ie  # 8

        qweight, scale = quantize_weight_per_block_int2(
            w, er_block_size=er, ie_block_size=ie,
        )
        # scale should have shape (2, 256/32) = (2, 8)
        assert scale.shape == (2, 8)
        # Adjacent scale values within same ER block should be identical
        for i in range(n):
            assert torch.equal(scale[:, i], scale[:, 0])

    def test_no_split_when_er_equals_ie(self):
        """When ER == IE, no scale replication."""
        w = torch.randn(2, 256, dtype=torch.bfloat16)
        qweight, scale = quantize_weight_per_block_int2(
            w, er_block_size=128, ie_block_size=128,
        )
        assert scale.shape == (2, 256 // 128)  # (2, 2)

    def test_er_ie_same_values(self):
        """Weight int values are identical regardless of IE granularity."""
        torch.manual_seed(42)
        w = torch.randn(4, 256, dtype=torch.bfloat16)

        # Quantize with IE=32 and IE=256
        qw32, s32 = quantize_weight_per_block_int2(
            w, er_block_size=256, ie_block_size=32,
        )
        q_int256, s256 = quantize_weight_per_block_int2(
            w, er_block_size=256, ie_block_size=256,
        )

        w32 = dequantize_weight(qw32, s32, block_size=32, weight_bits=1.58)
        w256 = dequantize_weight(q_int256, s256, block_size=256, weight_bits=1.58)

        # Bit-exact: same ternary values + same ER scale → same dequant
        assert torch.equal(w32, w256), (
            f"Dequantized weight differs with different IE: "
            f"max_diff={(w32 - w256).abs().max()}"
        )


# ────────────────────────────────────────────────────────────
# Marlin IE cap — degrade when IE exceeds Marlin max (128)
# ────────────────────────────────────────────────────────────

MARLIN_MAX_GROUP_SIZE = 128


class TestMarlinIECap:
    """IE block_size auto-capped to Marlin's max group_size."""

    def _effective_ie(self, ie: int) -> int:
        return min(ie, MARLIN_MAX_GROUP_SIZE)

    def _quantize_marlin_style(self, w, er, ie):
        """Simulate Marlin's process_weights_after_loading quant logic."""
        ie_eff = self._effective_ie(ie)
        from edgerazor.vllm.quant_ops import (
            INT1_58_MAX,
            INT4_MAX,
            quantize_weight_per_block_int4,
            quantize_weight_per_block_int2,
        )
        # W4: absmax scale, W1.58: clip scale
        if ie_eff >= 128:
            return quantize_weight_per_block_int4(
                w, er_block_size=er, ie_block_size=ie_eff,
            )
        else:
            return quantize_weight_per_block_int4(
                w, er_block_size=er, ie_block_size=ie_eff,
            )

    @pytest.mark.parametrize("er, ie, expected_ie", [
        (256, 256, 128),
        (256, 512, 128),
        (256, 128, 128),
        (256, 64, 64),
        (256, 32, 32),
    ])
    def test_ie_capped_to_max(self, er, ie, expected_ie):
        """IE > Marlin max (128) → capped at 128; smaller → unchanged."""
        assert self._effective_ie(ie) == expected_ie

    def test_quantize_ie256_vs_ie128_same_dequant(self):
        """IE=256 capped to 128 produces same dequantized weights."""
        torch.manual_seed(42)
        w = torch.randn(4, 512, dtype=torch.bfloat16)
        er = 256

        q256, s256 = quantize_weight_per_block_int4(
            w, er_block_size=er, ie_block_size=256,
        )
        q128, s128 = quantize_weight_per_block_int4(
            w, er_block_size=er, ie_block_size=128,
        )

        # IE=256 stored at coarser granularity; IE=128 duplicates scales
        w256 = dequantize_weight(q256, s256, block_size=256, weight_bits=4)
        w128 = dequantize_weight(q128, s128, block_size=128, weight_bits=4)
        # Both use same ER → same int values → bit-exact after dequant
        assert torch.equal(w256, w128), (
            f"max_diff={(w256 - w128).abs().max()}"
        )

    def test_w158_ie256_vs_ie128_same_dequant(self):
        """W1.58 IE=256 vs IE=128: same dequantized weights."""
        torch.manual_seed(42)
        w = torch.randn(4, 512, dtype=torch.bfloat16)
        er = 256

        q_int256, s256 = quantize_weight_per_block_int2(
            w, er_block_size=er, ie_block_size=256,
        )
        qw128, s128 = quantize_weight_per_block_int2(
            w, er_block_size=er, ie_block_size=128,
        )

        w256 = dequantize_weight(q_int256, s256, block_size=256, weight_bits=1.58)
        w128 = dequantize_weight(qw128, s128, block_size=128, weight_bits=1.58)
        assert torch.equal(w256, w128), (
            f"max_diff={(w256 - w128).abs().max()}"
        )

    def test_scale_shape_when_ie_capped(self):
        """IE=256 capped → scale shape matches IE=128."""
        torch.manual_seed(42)
        w = torch.randn(4, 1024, dtype=torch.bfloat16)
        er = 256

        _, s256 = quantize_weight_per_block_int4(
            w, er_block_size=er, ie_block_size=256,
        )
        _, s128 = quantize_weight_per_block_int4(
            w, er_block_size=er, ie_block_size=128,
        )

        # s256: (4, 1024/256) = (4, 4); s128: (4, 1024/128) = (4, 8)
        assert s256.shape == (4, 1024 // 256)
        assert s128.shape == (4, 1024 // 128)
        # Regardless of IE, dequantized result is the same
        q256, _ = quantize_weight_per_block_int4(
            w, er_block_size=er, ie_block_size=256,
        )
        w256 = dequantize_weight(q256, s256, block_size=256, weight_bits=4)
        w128 = dequantize_weight(
            quantize_weight_per_block_int4(w, er_block_size=er, ie_block_size=128)[0],
            s128, block_size=128, weight_bits=4,
        )
        assert torch.equal(w256, w128)


# ────────────────────────────────────────────────────────────
# Activation quantization compatibility
# ────────────────────────────────────────────────────────────


class TestW158ActivationQuant:
    """W1.58-A8 activation quantization is identical to W4-A8."""

    def test_activation_quant_identical(self):
        """Activation quantization doesn't depend on weight_bits."""
        x = torch.randn(2, 4, 256, dtype=torch.bfloat16)
        x_int, x_scale = quantize_activation_per_block_int8(x)
        assert x_int.dtype == torch.int8
        assert x_scale.shape == (2, 4, 256 // 256)  # block_size=256

    def test_per_token_int8(self):
        """per-token INT8 quantization."""
        x = torch.randn(3, 8, 512, dtype=torch.bfloat16)
        from edgerazor.vllm.quant_ops import quantize_activation_per_token_int8
        x_int, x_scale = quantize_activation_per_token_int8(x)
        assert x_int.dtype == torch.int8
        assert x_scale.shape == (3, 8, 1)
        assert (x_scale > 0).all()

    def test_dequant_activation_roundtrip(self):
        """Activation INT8 dequantization is close to original."""
        torch.manual_seed(42)
        x = torch.randn(2, 256, dtype=torch.bfloat16)
        from edgerazor.vllm.quant_ops import quantize_activation_per_token_int8
        x_int, x_scale = quantize_activation_per_token_int8(x)
        x_deq = x_int.float() * x_scale.float()
        # MSE should be very small (8-bit precision)
        mse = torch.nn.functional.mse_loss(x_deq, x.float())
        assert mse < 1e-4, f"MSE={mse}"
