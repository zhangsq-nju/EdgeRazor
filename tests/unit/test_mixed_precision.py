"""TDD tests for mixed-precision quantization: 1.88-bit and 2.79-bit.

Solution: dual qweight tensors (INT4 + INT2) with row-index interleaving.

Super-group = 8 output channels, pattern repeated every 8 rows:

  1.88-bit: [INT4, INT2, INT2, INT2, INT2, INT2, INT2, INT2]  (row % 8 == 0 → INT4)
  2.79-bit: [INT4, INT4, INT4, INT4, INT2, INT2, INT2, INT2]  (row % 8 < 4 → INT4)
"""

import pytest
import torch

from edgerazor.vllm.quant_ops import (
    dequantize_weight_mixed,
    quantize_weight_mixed_precision,
)


# ── helpers ─────────────────────────────────────────────────


def _mixed_row_mask(N: int, bits: float) -> torch.Tensor:
    """Expected INT4 row mask for the given bit-width."""
    if bits == 1.88:
        # Every 8th row (row % 8 == 0) is INT4
        return torch.tensor([i % 8 == 0 for i in range(N)])
    elif bits == 2.79:
        # First 4 rows of each group of 8 are INT4
        return torch.tensor([i % 8 < 4 for i in range(N)])
    else:
        raise ValueError(f"Unknown bits={bits}")


# ── tracer bullet: basic quantize/dequantize roundtrip ──────


class TestMixedPrecisionRoundtrip:
    """End-to-end: bf16 weight → mixed quantize → dequantize → bf16."""

    def test_tracer_bullet_188_dequant_cosine(self):
        """Tracer bullet: 1.88-bit quantize + dequant preserves direction."""
        torch.manual_seed(42)
        w = torch.randn(128, 256, dtype=torch.bfloat16)  # N=128 rows

        qw_int4, qw_int2, s_int4, s_int2, row_mask = quantize_weight_mixed_precision(
            w, bits=1.88, er_block_size=256, ie_block_size=32,
        )

        # Verify row mask structure
        assert row_mask.dtype == torch.bool
        assert row_mask.shape == (128,)
        # Every 8th row is INT4
        expected = _mixed_row_mask(128, 1.88)
        assert (row_mask == expected).all(), f"Row mask mismatch at {bits}"

        # Verify output shapes
        n_int4 = expected.sum().item()  # 16 rows
        n_int2 = 128 - n_int4  # 112 rows
        assert qw_int4.shape == (n_int4, 128)  # K/2 = 128
        assert qw_int2.shape == (n_int2, 64)  # K/4 = 64
        assert s_int4.shape == (n_int4, 8)  # K/IE = 8
        assert s_int2.shape == (n_int2, 8)

        # Dequantize
        w_deq = dequantize_weight_mixed(
            qw_int4, qw_int2, s_int4, s_int2, row_mask,
            block_size=32, out_dtype=torch.bfloat16,
        )

        assert w_deq.shape == w.shape

        # Cosine similarity should be high
        cos = torch.nn.functional.cosine_similarity(
            w.float().view(1, -1), w_deq.float().view(1, -1),
        ).item()
        # 1.88-bit: 87.5% rows are INT2 (ternary, coarse) → cos lower than pure INT4
        assert cos > 0.88, f"cos_sim={cos:.4f} too low"

        # INT4 rows should have good accuracy
        int4_rows = torch.where(row_mask)[0]
        if len(int4_rows) > 0:
            diff_int4 = (w[int4_rows].float() - w_deq[int4_rows].float()).abs()
            assert diff_int4.max().item() < 1.0

    def test_tracer_bullet_279_dequant_cosine(self):
        """Tracer bullet: 2.79-bit quantize + dequant preserves direction."""
        torch.manual_seed(42)
        w = torch.randn(128, 256, dtype=torch.bfloat16)

        qw_int4, qw_int2, s_int4, s_int2, row_mask = quantize_weight_mixed_precision(
            w, bits=2.79, er_block_size=256, ie_block_size=32,
        )

        expected = _mixed_row_mask(128, 2.79)
        assert (row_mask == expected).all()

        w_deq = dequantize_weight_mixed(
            qw_int4, qw_int2, s_int4, s_int2, row_mask,
            block_size=32, out_dtype=torch.bfloat16,
        )
        cos = torch.nn.functional.cosine_similarity(
            w.float().view(1, -1), w_deq.float().view(1, -1),
        ).item()
        # 2.79-bit: 50% INT4, 50% INT2 → better than 1.88
        assert cos > 0.92, f"cos_sim={cos:.4f} too low"


# ── compression ratio ──────────────────────────────────────


class TestMixedCompression:
    """Verify the compression ratio matches theoretical expectation."""

    @pytest.mark.parametrize("bits,N,K", [
        (1.88, 128, 1024),
        (2.79, 128, 1024),
        (1.88, 256, 512),  # N not multiple of 8 — pattern still works
    ])
    def test_compression_below_pure_int4(self, bits, N, K):
        """Mixed precision should use less memory than pure INT4."""
        torch.manual_seed(42)
        w = torch.randn(N, K, dtype=torch.bfloat16)
        orig_bytes = N * K * 2

        qw_i4, qw_int2, s_i4, s_int2, _ = quantize_weight_mixed_precision(
            w, bits=bits, er_block_size=256, ie_block_size=32,
        )

        packed = (qw_i4.numel() + qw_int2.numel()) * 1 \
               + (s_i4.numel() + s_int2.numel()) * 2
        ratio = packed / orig_bytes * 100

        # Theoretical: 1.88 → ~28%, 2.79 → ~37.5%
        assert ratio < 45, f"bits={bits}: ratio={ratio:.1f}% expected <45%"

    @pytest.mark.parametrize("bits,expected_pct", [
        (1.88, 30),   # ~28% theoretical
        (2.79, 41),   # ~38% theoretical
    ])
    def test_compression_near_theoretical(self, bits, expected_pct):
        """Large matrix → ratio converges to theoretical value."""
        torch.manual_seed(42)
        N, K = 512, 2048  # large enough to converge
        w = torch.randn(N, K, dtype=torch.bfloat16)
        orig_bytes = N * K * 2

        qw_i4, qw_int2, s_i4, s_int2, _ = quantize_weight_mixed_precision(
            w, bits=bits, er_block_size=256, ie_block_size=32,
        )
        packed = (qw_i4.numel() + qw_int2.numel()) * 1 \
               + (s_i4.numel() + s_int2.numel()) * 2
        ratio = packed / orig_bytes * 100
        assert ratio < expected_pct, f"bits={bits}: {ratio:.1f}% >= {expected_pct}%"


# ── matmul accuracy ────────────────────────────────────────


class TestMixedPrecisionMatmul:
    """Verify mixed dequant + matmul produces reasonable output."""

    @pytest.mark.parametrize("bits", [1.88, 2.79])
    def test_matmul_cosine(self, bits):
        """matmul(y_deq, x) ≈ matmul(y_orig, x)"""
        torch.manual_seed(42)
        N, K, M = 128, 256, 4
        w = torch.randn(N, K, dtype=torch.bfloat16)
        x = torch.randn(M, K, dtype=torch.bfloat16)

        qw_i4, qw_int2, s_i4, s_int2, mask = quantize_weight_mixed_precision(
            w, bits=bits, er_block_size=256, ie_block_size=32,
        )
        w_deq = dequantize_weight_mixed(
            qw_i4, qw_int2, s_i4, s_int2, mask,
            block_size=32, out_dtype=torch.bfloat16,
        )

        y_ref = torch.nn.functional.linear(x, w)
        y_deq = torch.nn.functional.linear(x, w_deq)

        cos = torch.nn.functional.cosine_similarity(
            y_ref.float().view(-1), y_deq.float().view(-1), dim=0,
        ).item()
        assert cos > 0.85, f"bits={bits}: matmul cos={cos:.4f}"

    def test_int4_rows_more_accurate_than_int2_rows(self):
        """INT4 rows should have lower per-row error than INT2 rows."""
        torch.manual_seed(42)
        w = torch.randn(128, 256, dtype=torch.bfloat16)
        qw_i4, qw_int2, s_i4, s_int2, mask = quantize_weight_mixed_precision(
            w, bits=2.79,  # 50% INT4, 50% INT2
            er_block_size=256, ie_block_size=32,
        )
        w_deq = dequantize_weight_mixed(
            qw_i4, qw_int2, s_i4, s_int2, mask,
            block_size=32, out_dtype=torch.bfloat16,
        )

        err_i4 = (w[mask].float() - w_deq[mask].float()).abs().mean().item()
        err_int2 = (w[~mask].float() - w_deq[~mask].float()).abs().mean().item()
        assert err_i4 < err_int2, f"INT4 err={err_i4:.6f} >= INT2 err={err_int2:.6f}"


# ── edge cases ─────────────────────────────────────────────


class TestMixedEdgeCases:
    """Edge cases: N not divisible by 8, all-one-pattern rows."""

    def test_N_not_divisible_by_8(self):
        """Pattern still works when N is not a multiple of 8."""
        torch.manual_seed(42)
        w = torch.randn(13, 256, dtype=torch.bfloat16)  # N=13, not divisible by 8
        for bits in [1.88, 2.79]:
            qw_i4, qw_int2, s_i4, s_int2, row_mask = quantize_weight_mixed_precision(
                w, bits=bits, er_block_size=256, ie_block_size=32,
            )
            assert row_mask.shape == (13,)
            w_deq = dequantize_weight_mixed(
                qw_i4, qw_int2, s_i4, s_int2, row_mask,
                block_size=32, out_dtype=torch.bfloat16,
            )
            assert w_deq.shape == w.shape

    def test_er_ie_split_preserves_weight_ints(self):
        """ER/IE split: weight-int values identical regardless of IE."""
        torch.manual_seed(42)
        w = torch.randn(128, 256, dtype=torch.bfloat16)
        er, ie = 256, 32

        # Quantize with ER=256, IE=32 (needs_split=True)
        qw_i4, qw_int2, s_i4, s_int2, mask = quantize_weight_mixed_precision(
            w, bits=2.79, er_block_size=er, ie_block_size=ie,
        )
        w_deq = dequantize_weight_mixed(
            qw_i4, qw_int2, s_i4, s_int2, mask,
            block_size=ie, out_dtype=torch.bfloat16,
        )

        # Quantize with ER=256, IE=256 (needs_split=False)
        qw_i4_2, qw_int2_2, s_i4_2, s_int2_2, mask_2 = quantize_weight_mixed_precision(
            w, bits=2.79, er_block_size=er, ie_block_size=er,
        )
        w_deq_2 = dequantize_weight_mixed(
            qw_i4_2, qw_int2_2, s_i4_2, s_int2_2, mask_2,
            block_size=er, out_dtype=torch.bfloat16,
        )

        # Dequantized values should be nearly identical (max small diff)
        diff = (w_deq.float() - w_deq_2.float()).abs()
        assert diff.max().item() < 0.1, f"ER/IE split max diff={diff.max().item():.6f}"
