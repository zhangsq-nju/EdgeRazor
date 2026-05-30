"""Unit tests for edgerazor.vllm plugin.

Tests pure quant_ops functions (no vLLM dependency required).
Integration tests for vLLM-specific linear methods require CUDA GPU
and are excluded from the local test suite.
"""

import pytest
import torch

from edgerazor.vllm.quant_ops import (
    ER_W4A8_BLOCK_SIZE,
    IE_W4A8_BLOCK_SIZE,
    INT4_MAX,
    INT8_MAX,
    W1_58A8_BLOCK_SIZE,
    dequantize_weight,
    pack_int4,
    quantize_activation_per_block_int8,
    quantize_activation_per_token_int8,
    quantize_weight_per_block_int4,
    quantize_weight_ternary_to_int4,
    resolve_quant_block,
    unpack_int4,
)

# ────────────────────────────────────────────────────────────
# pack / unpack INT4
# ────────────────────────────────────────────────────────────


class TestPackUnpackInt4:
    """pack_int4 ↔ unpack_int4 roundtrip identity."""

    @pytest.mark.parametrize("shape", [
        (16, 256),
        (32, 512),
        (8, 64),
        (1, 128),
    ])
    def test_roundtrip_bit_exact(self, shape):
        """Pack then unpack recovers the original INT4 values exactly."""
        torch.manual_seed(42)
        w_int = torch.randint(-INT4_MAX, INT4_MAX + 1, shape, dtype=torch.int8)
        qweight = pack_int4(w_int)
        assert qweight.shape == (shape[0], shape[1] // 2)
        assert qweight.dtype == torch.uint8

        recovered = unpack_int4(qweight)
        assert recovered.shape == w_int.shape
        assert torch.equal(w_int, recovered)

    def test_pack_correct_shape(self):
        """Packing halves the last dimension."""
        w = torch.randint(-INT4_MAX, INT4_MAX + 1, (4, 32), dtype=torch.int8)
        q = pack_int4(w)
        assert q.shape == (4, 16)
        assert q.dtype == torch.uint8

    def test_unpack_correct_shape(self):
        """Unpacking doubles the last dimension."""
        q = torch.randint(0, 255, (4, 16), dtype=torch.uint8)
        w = unpack_int4(q)
        assert w.shape == (4, 32)
        assert w.dtype == torch.int8


# ────────────────────────────────────────────────────────────
# resolve_quant_block
# ────────────────────────────────────────────────────────────


class TestResolveQuantBlock:
    """ER / IE block-size resolution."""

    @pytest.mark.parametrize("er,ie,exp_quant,exp_scale,exp_split", [
        (128, 128, 128, 128, False),   # equal
        (32, 64, 64, 64, False),       # ER < IE
        (256, 32, 256, 32, True),      # ER > IE, split
        (256, 128, 256, 128, True),    # ER > IE, split
        (512, 64, 512, 64, True),      # ER > IE, split
    ])
    def test_valid_resolutions(self, er, ie, exp_quant, exp_scale, exp_split):
        """Valid ER/IE pairs return correct resolution."""
        quant_block, scale_block, needs_split = resolve_quant_block(er, ie)
        assert quant_block == exp_quant
        assert scale_block == exp_scale
        assert needs_split == exp_split

    def test_er_not_multiple_of_ie_raises(self):
        """ER not divisible by IE raises ValueError."""
        with pytest.raises(ValueError, match="integer multiple"):
            resolve_quant_block(256, 48)

    @pytest.mark.parametrize("er,ie", [
        (64, 128),    # ER < IE
        (128, 128),   # ER = IE
        (32, 64),     # ER < IE, both small
    ])
    def test_no_split_when_er_le_ie(self, er, ie):
        """When ER <= IE, quant block equals IE and no split."""
        qb, sb, split = resolve_quant_block(er, ie)
        assert qb == ie
        assert sb == ie
        assert not split


# ────────────────────────────────────────────────────────────
# Weight quantization + dequantization
# ────────────────────────────────────────────────────────────


class TestWeightQuantize:
    """per-block INT4 weight quantization and dequantization."""

    @pytest.mark.parametrize("out_dim,in_dim,er,ie", [
        (64, 512, 256, 256),   # no split
        (64, 512, 256, 32),    # split 8x
        (32, 768, 256, 64),    # split 4x, odd dims
    ])
    def test_dequantize_cosine_similarity(self, out_dim, in_dim, er, ie):
        """Dequantized weight has cos_sim > 0.99 with original."""
        torch.manual_seed(123)
        w = torch.randn(out_dim, in_dim, dtype=torch.bfloat16)
        qw, sc = quantize_weight_per_block_int4(w, er_block_size=er, ie_block_size=ie)
        w_deq = dequantize_weight(qw, sc, block_size=ie, out_dtype=torch.bfloat16)
        cos_sim = torch.nn.functional.cosine_similarity(
            w_deq.flatten().float(), w.flatten().float(), dim=0,
        )
        assert cos_sim > 0.99

    def test_matmul_output_accuracy(self):
        """Matmul with quantized weight produces near-identical output."""
        torch.manual_seed(456)
        out_dim, in_dim, batch, seq = 128, 256, 2, 8
        w_ref = torch.randn(out_dim, in_dim, dtype=torch.bfloat16)
        x = torch.randn(batch, seq, in_dim, dtype=torch.bfloat16)

        qw, sc = quantize_weight_per_block_int4(w_ref, er_block_size=256, ie_block_size=32)
        w_deq = dequantize_weight(qw, sc, block_size=32, out_dtype=torch.bfloat16)

        out_quant = torch.nn.functional.linear(x, w_deq)
        out_ref = torch.nn.functional.linear(x, w_ref)
        cos_sim = torch.nn.functional.cosine_similarity(
            out_quant.flatten().float(), out_ref.flatten().float(), dim=0,
        )
        assert cos_sim > 0.99

    def test_clamps_to_int4_range(self):
        """Quantized int values stay within [-7, 7]."""
        torch.manual_seed(99)
        w = torch.randn(16, 512, dtype=torch.bfloat16) * 5.0  # large values
        qw, _sc = quantize_weight_per_block_int4(w, er_block_size=256, ie_block_size=32)
        w_int = unpack_int4(qw)
        assert w_int.min() >= -INT4_MAX
        assert w_int.max() <= INT4_MAX


# ────────────────────────────────────────────────────────────
# ER / IE split correctness
# ────────────────────────────────────────────────────────────


class TestErIeSplit:
    """ER > IE scale splitting preserves weight-int values."""

    @pytest.mark.parametrize("er,ie", [
        (256, 32),
        (256, 64),
        (256, 128),
    ])
    def test_split_preserves_weight_ints(self, er, ie):
        """Weight-int values identical whether split or not."""
        torch.manual_seed(111)
        out_dim, in_dim = 8, 512
        scale_true = torch.rand(out_dim, 1, dtype=torch.bfloat16) * 0.1 + 0.01
        w_int_truth = torch.randint(-7, 8, (out_dim, in_dim), dtype=torch.int8)
        w = (w_int_truth.float() * scale_true.float()).to(torch.bfloat16)

        # No-split: ER=IE
        qw_ns, sc_ns = quantize_weight_per_block_int4(
            w, er_block_size=er, ie_block_size=er,
        )
        w_int_ns = unpack_int4(qw_ns)

        # Split: ER > IE
        qw_s, sc_s = quantize_weight_per_block_int4(
            w, er_block_size=er, ie_block_size=ie,
        )
        w_int_s = unpack_int4(qw_s)

        assert torch.equal(w_int_ns, w_int_s), (
            f"ER={er}, IE={ie}: split changed weight-int values"
        )

    @pytest.mark.parametrize("er,ie,n_split", [
        (256, 32, 8),
        (256, 64, 4),
        (256, 128, 2),
    ])
    def test_scale_replication_within_group(self, er, ie, n_split):
        """Each ER scale is replicated N times across consecutive IE blocks."""
        torch.manual_seed(222)
        out_dim, in_dim = 4, er * 4
        w = torch.randn(out_dim, in_dim, dtype=torch.bfloat16)

        _qw, sc = quantize_weight_per_block_int4(
            w, er_block_size=er, ie_block_size=ie,
        )
        num_ie_blocks = in_dim // ie
        sc_reshaped = sc.view(out_dim, num_ie_blocks)

        for g in range(num_ie_blocks // n_split):
            group = sc_reshaped[:, g * n_split : (g + 1) * n_split]
            first = group[:, 0:1]
            max_diff = (group - first).abs().max()
            assert max_diff < 1e-6, (
                f"Scale group {g} not replicated: max_diff={max_diff.item():.8f}"
            )

    def test_dequantized_nearly_identical_after_split(self):
        """ER>IE split dequant result near-identical to no-split."""
        torch.manual_seed(333)
        out_dim, in_dim = 8, 1024
        w = torch.randn(out_dim, in_dim, dtype=torch.bfloat16)

        qw_a, sc_a = quantize_weight_per_block_int4(
            w, er_block_size=256, ie_block_size=256,
        )
        w_a = dequantize_weight(qw_a, sc_a, block_size=256, out_dtype=torch.bfloat16)

        qw_b, sc_b = quantize_weight_per_block_int4(
            w, er_block_size=256, ie_block_size=32,
        )
        w_b = dequantize_weight(qw_b, sc_b, block_size=32, out_dtype=torch.bfloat16)

        mse = torch.nn.functional.mse_loss(w_a.float(), w_b.float())
        assert mse < 1e-4, f"Split dequant MSE too high: {mse.item():.8f}"


# ────────────────────────────────────────────────────────────
# Activation quantization
# ────────────────────────────────────────────────────────────


class TestActivationQuant:
    """Per-token and per-block INT8 activation quantization."""

    @pytest.mark.parametrize("batch,seq,hidden", [
        (2, 16, 1024),
        (1, 8, 256),
        (4, 32, 512),
    ])
    def test_per_token_cosine_similarity(self, batch, seq, hidden):
        """Per-token INT8 quantization maintains cos_sim > 0.99."""
        torch.manual_seed(321)
        x = torch.randn(batch, seq, hidden, dtype=torch.bfloat16)
        x_int, x_scale = quantize_activation_per_token_int8(x)
        x_deq = x_int.float() * x_scale.float()
        cos_sim = torch.nn.functional.cosine_similarity(
            x_deq.reshape(-1, hidden), x.float().reshape(-1, hidden), dim=1,
        )
        assert cos_sim.mean() > 0.99

    @pytest.mark.parametrize("batch,seq,hidden,block_size", [
        (2, 16, 1024, 256),
        (1, 8, 512, 128),
    ])
    def test_per_block_cosine_similarity(self, batch, seq, hidden, block_size):
        """Per-block INT8 quantization maintains cos_sim > 0.99."""
        torch.manual_seed(654)
        x = torch.randn(batch, seq, hidden, dtype=torch.bfloat16)
        x_int, x_scale = quantize_activation_per_block_int8(x, block_size=block_size)
        nblocks = hidden // block_size
        x_deq = x_int.float().view(batch, seq, nblocks, block_size) * x_scale.float().unsqueeze(-1)
        x_deq = x_deq.view(batch, seq, hidden)
        cos_sim = torch.nn.functional.cosine_similarity(
            x_deq.reshape(-1, hidden), x.float().reshape(-1, hidden), dim=1,
        )
        assert cos_sim.mean() > 0.99

    def test_clamps_to_int8_range(self):
        """Per-token quantization clamps to [-127, 127]."""
        torch.manual_seed(77)
        x = torch.randn(2, 8, 256, dtype=torch.bfloat16) * 10.0
        x_int, _sc = quantize_activation_per_token_int8(x)
        assert x_int.min() >= -INT8_MAX
        assert x_int.max() <= INT8_MAX

    def test_scale_positive(self):
        """All per-token scales must be positive."""
        x = torch.randn(4, 16, 512, dtype=torch.bfloat16)
        _x_int, x_scale = quantize_activation_per_token_int8(x)
        assert (x_scale > 0).all()


# ────────────────────────────────────────────────────────────
# Ternary → INT4
# ────────────────────────────────────────────────────────────


class TestTernaryToInt4:
    """1.58-bit ternary weights degraded to INT4 packing."""

    def test_near_ternary_weights_preserved(self):
        """Weights that are already ternary-like survive roundtrip."""
        torch.manual_seed(789)
        out_dim, in_dim = 64, 512
        w = torch.randn(out_dim, in_dim, dtype=torch.bfloat16) * 0.5
        w = w.round()  # near-ternary

        qw, sc = quantize_weight_ternary_to_int4(w)
        w_deq = dequantize_weight(qw, sc, block_size=W1_58A8_BLOCK_SIZE, out_dtype=torch.bfloat16)
        cos_sim = torch.nn.functional.cosine_similarity(
            w_deq.flatten().float(), w.flatten().float(), dim=0,
        )
        assert cos_sim > 0.99

    def test_produces_only_ternary_values(self):
        """Quantized int values are only -1, 0, or 1."""
        torch.manual_seed(42)
        w = torch.randn(32, 256, dtype=torch.bfloat16) * 3.0
        qw, _sc = quantize_weight_ternary_to_int4(w)
        w_int = unpack_int4(qw)
        unique = w_int.unique()
        assert set(unique.tolist()).issubset({-1, 0, 1})


# ────────────────────────────────────────────────────────────
# Memory / compression ratio
# ────────────────────────────────────────────────────────────


class TestCompressionRatio:
    """Packed weight vs bf16 memory savings."""

    def test_er256_compression_lt_30pct(self):
        """ER=256: packed weight under 30% of bf16."""
        out_dim, in_dim = 1024, 4096
        qweight = torch.zeros(out_dim, in_dim // 2, dtype=torch.uint8)
        qweight_scale = torch.zeros(
            out_dim, in_dim // ER_W4A8_BLOCK_SIZE, dtype=torch.bfloat16,
        )
        packed_bytes = qweight.numel() * 1 + qweight_scale.numel() * 2
        bf16_bytes = out_dim * in_dim * 2
        ratio = packed_bytes / bf16_bytes
        assert ratio < 0.30

    def test_ie32_compression_lt_35pct(self):
        """IE=32 (split): packed weight under 35% of bf16."""
        out_dim, in_dim = 1024, 4096
        qweight = torch.zeros(out_dim, in_dim // 2, dtype=torch.uint8)
        qweight_scale = torch.zeros(
            out_dim, in_dim // IE_W4A8_BLOCK_SIZE, dtype=torch.bfloat16,
        )
        packed_bytes = qweight.numel() * 1 + qweight_scale.numel() * 2
        bf16_bytes = out_dim * in_dim * 2
        ratio = packed_bytes / bf16_bytes
        assert ratio < 0.35
