"""TDD tests for mixed-precision quantization: 1.88-bit and 2.79-bit.

Solution: dual qweight tensors (INT4 + INT2) with inverse-perm interleave.

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


def _expected_row_mask(N: int, bits: float) -> torch.Tensor:
    """Expected INT4 row mask for the given bit-width."""
    if bits == 1.88:
        return torch.tensor([i % 8 == 0 for i in range(N)])
    elif bits == 2.79:
        return torch.tensor([i % 8 < 4 for i in range(N)])
    else:
        raise ValueError(f"Unknown bits={bits}")


def _expected_inverse_perm(N: int, bits: float) -> torch.Tensor:
    """Expected inverse permutation: cat([INT4, INT2])[perm] → original order."""
    mask = _expected_row_mask(N, bits)
    idx_int4 = torch.where(mask)[0]
    idx_int2 = torch.where(~mask)[0]
    combined = torch.cat([idx_int4, idx_int2])
    return torch.argsort(combined)


# ── tracer bullet: basic quantize/dequantize roundtrip ──────


class TestMixedPrecisionRoundtrip:
    """End-to-end: bf16 weight → mixed quantize → dequantize → bf16."""

    def test_tracer_bullet_188_dequant_cosine(self):
        """Tracer bullet: 1.88-bit quantize + dequant preserves direction."""
        torch.manual_seed(42)
        w = torch.randn(128, 256, dtype=torch.bfloat16)  # N=128 rows

        qw_int4, qw_int2, s_int4, s_int2, inv_perm = quantize_weight_mixed_precision(
            w, bits=1.88, er_block_size=256, ie_block_size=32,
        )

        # Verify inverse_perm structure
        assert inv_perm.dtype == torch.int64
        assert inv_perm.shape == (128,)
        expected_perm = _expected_inverse_perm(128, 1.88)
        assert (inv_perm == expected_perm).all()

        # Verify output shapes
        n_int4 = _expected_row_mask(128, 1.88).sum().item()  # 16 rows
        n_int2 = 128 - n_int4  # 112 rows
        assert qw_int4.shape == (n_int4, 128)  # K/2 = 128
        assert qw_int2.shape == (n_int2, 64)  # K/4 = 64
        assert s_int4.shape == (n_int4, 8)  # K/IE = 8
        assert s_int2.shape == (n_int2, 8)

        # Dequantize
        w_deq = dequantize_weight_mixed(
            qw_int4, qw_int2, s_int4, s_int2, inv_perm,
            block_size=32, out_dtype=torch.bfloat16,
        )
        assert w_deq.shape == w.shape

        # Cosine similarity
        cos = torch.nn.functional.cosine_similarity(
            w.float().view(1, -1), w_deq.float().view(1, -1),
        ).item()
        assert cos > 0.88, f"cos_sim={cos:.4f} too low"

        # INT4 rows should have good accuracy
        mask = _expected_row_mask(128, 1.88)
        int4_rows = torch.where(mask)[0]
        if len(int4_rows) > 0:
            diff_int4 = (w[int4_rows].float() - w_deq[int4_rows].float()).abs()
            assert diff_int4.max().item() < 1.0

    def test_tracer_bullet_279_dequant_cosine(self):
        """Tracer bullet: 2.79-bit quantize + dequant preserves direction."""
        torch.manual_seed(42)
        w = torch.randn(128, 256, dtype=torch.bfloat16)

        qw_int4, qw_int2, s_int4, s_int2, inv_perm = quantize_weight_mixed_precision(
            w, bits=2.79, er_block_size=256, ie_block_size=32,
        )

        expected_perm = _expected_inverse_perm(128, 2.79)
        assert (inv_perm == expected_perm).all()

        w_deq = dequantize_weight_mixed(
            qw_int4, qw_int2, s_int4, s_int2, inv_perm,
            block_size=32, out_dtype=torch.bfloat16,
        )
        cos = torch.nn.functional.cosine_similarity(
            w.float().view(1, -1), w_deq.float().view(1, -1),
        ).item()
        assert cos > 0.92, f"cos_sim={cos:.4f} too low"

    def test_inverse_perm_reconstructs_order(self):
        """cat([INT4, INT2])[inv_perm] reproduces the original row-major order."""
        torch.manual_seed(42)
        w = torch.randn(64, 256, dtype=torch.bfloat16)

        qw_i4, qw_int2, s_i4, s_int2, inv_perm = quantize_weight_mixed_precision(
            w, bits=1.88, er_block_size=256, ie_block_size=32,
        )
        w_deq = dequantize_weight_mixed(
            qw_i4, qw_int2, s_i4, s_int2, inv_perm,
            block_size=32, out_dtype=torch.bfloat16,
        )
        # INT4 rows: tight bound; INT2 rows: looser due to ternary quantization
        mask = _expected_row_mask(64, 1.88)
        for i in range(64):
            diff = (w[i].float() - w_deq[i].float()).abs().max().item()
            limit = 0.5 if mask[i] else 2.0  # INT4 tight, INT2 permissive
            assert diff < limit, f"Row {i} max_diff={diff:.4f} > {limit}"


# ── compression ratio ──────────────────────────────────────


class TestMixedCompression:
    """Verify the compression ratio matches theoretical expectation."""

    @pytest.mark.parametrize("bits,N,K", [
        (1.88, 128, 1024),
        (2.79, 128, 1024),
        (1.88, 256, 512),
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
        assert ratio < 45, f"bits={bits}: ratio={ratio:.1f}% expected <45%"

    @pytest.mark.parametrize("bits,expected_pct", [
        (1.88, 30),
        (2.79, 41),
    ])
    def test_compression_near_theoretical(self, bits, expected_pct):
        """Large matrix → ratio converges to theoretical value."""
        torch.manual_seed(42)
        N, K = 512, 2048
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
        """matmul(w_deq, x) ≈ matmul(w_orig, x)"""
        torch.manual_seed(42)
        N, K, M = 128, 256, 4
        w = torch.randn(N, K, dtype=torch.bfloat16)
        x = torch.randn(M, K, dtype=torch.bfloat16)

        qw_i4, qw_int2, s_i4, s_int2, inv_perm = quantize_weight_mixed_precision(
            w, bits=bits, er_block_size=256, ie_block_size=32,
        )
        w_deq = dequantize_weight_mixed(
            qw_i4, qw_int2, s_i4, s_int2, inv_perm,
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
        mask = _expected_row_mask(128, 2.79)

        qw_i4, qw_int2, s_i4, s_int2, inv_perm = quantize_weight_mixed_precision(
            w, bits=2.79, er_block_size=256, ie_block_size=32,
        )
        w_deq = dequantize_weight_mixed(
            qw_i4, qw_int2, s_i4, s_int2, inv_perm,
            block_size=32, out_dtype=torch.bfloat16,
        )

        err_i4 = (w[mask].float() - w_deq[mask].float()).abs().mean().item()
        err_int2 = (w[~mask].float() - w_deq[~mask].float()).abs().mean().item()
        assert err_i4 < err_int2, f"INT4 err={err_i4:.6f} >= INT2 err={err_int2:.6f}"


# ── edge cases ─────────────────────────────────────────────


class TestMixedEdgeCases:
    """Edge cases: N not divisible by 8, ER/IE split."""

    def test_N_not_divisible_by_8(self):
        """Pattern works when N is not a multiple of 8."""
        torch.manual_seed(42)
        w = torch.randn(13, 256, dtype=torch.bfloat16)
        for bits in [1.88, 2.79]:
            qw_i4, qw_int2, s_i4, s_int2, inv_perm = quantize_weight_mixed_precision(
                w, bits=bits, er_block_size=256, ie_block_size=32,
            )
            assert inv_perm.shape == (13,)
            w_deq = dequantize_weight_mixed(
                qw_i4, qw_int2, s_i4, s_int2, inv_perm,
                block_size=32, out_dtype=torch.bfloat16,
            )
            assert w_deq.shape == w.shape

    def test_er_ie_split_preserves_weight_ints(self):
        """ER/IE split: dequantized values nearly identical."""
        torch.manual_seed(42)
        w = torch.randn(128, 256, dtype=torch.bfloat16)
        er, ie = 256, 32

        qw_i4, qw_int2, s_i4, s_int2, inv_perm = quantize_weight_mixed_precision(
            w, bits=2.79, er_block_size=er, ie_block_size=ie,
        )
        w_deq = dequantize_weight_mixed(
            qw_i4, qw_int2, s_i4, s_int2, inv_perm,
            block_size=ie, out_dtype=torch.bfloat16,
        )

        qw_i4_2, qw_int2_2, s_i4_2, s_int2_2, inv_perm_2 = \
            quantize_weight_mixed_precision(
                w, bits=2.79, er_block_size=er, ie_block_size=er,
            )
        w_deq_2 = dequantize_weight_mixed(
            qw_i4_2, qw_int2_2, s_i4_2, s_int2_2, inv_perm_2,
            block_size=er, out_dtype=torch.bfloat16,
        )

        diff = (w_deq.float() - w_deq_2.float()).abs()
        assert diff.max().item() < 0.1, f"ER/IE split max diff={diff.max().item():.6f}"
