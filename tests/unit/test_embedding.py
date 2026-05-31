"""Tests for quantized embedding: roundtrip, dequant correctness."""

import pytest
import torch

from edgerazor.vllm.quant_ops import (
    dequantize_weight,
    pack_int4,
    pack_w2,
    quantize_weight_per_block_int4,
    quantize_weight_per_block_w2,
    unpack_int4,
    unpack_w2,
)


# ── tracer bullet: quantize → index_select → dequant ──


class TestEmbeddingQuantize:
    """Simulate the embedding forward path."""

    @pytest.mark.parametrize("vocab,hidden,indices", [
        (1000, 256, [0, 5, 10]),
        (500, 512, [3, 99, 250]),
        (100, 256, [0, 1, 2, 3]),
    ])
    def test_int4_embedding_roundtrip(self, vocab, hidden, indices):
        """Quantized → index_select → dequant ≈ original."""
        torch.manual_seed(42)
        w = torch.randn(vocab, hidden, dtype=torch.bfloat16)

        qweight, scale = quantize_weight_per_block_int4(
            w, er_block_size=256, ie_block_size=32,
        )
        idx = torch.tensor(indices)

        # Sparse lookup → dequant
        sel_qw = qweight[idx]        # (len, hidden//2)
        sel_s  = scale[idx]           # (len, hidden//ie)
        deq = dequantize_weight(sel_qw, sel_s, block_size=32, weight_bits=4)

        orig = w[idx]
        cos = torch.nn.functional.cosine_similarity(
            deq.float().view(-1), orig.float().view(-1), dim=0,
        )
        assert cos > 0.99

    @pytest.mark.parametrize("vocab,hidden,indices", [
        (500, 256, [0, 7, 42]),
        (200, 512, [1, 50, 199]),
    ])
    def test_w2_embedding_roundtrip(self, vocab, hidden, indices):
        """W2 embedding: index_select → unpack_w2 → dequant."""
        torch.manual_seed(42)
        w = torch.randn(vocab, hidden, dtype=torch.bfloat16)

        qweight, scale = quantize_weight_per_block_w2(
            w, er_block_size=256, ie_block_size=256,
        )
        idx = torch.tensor(indices)
        sel_qw = qweight[idx]       # (len, hidden//4)
        sel_s  = scale[idx]          # (len, hidden//256)
        deq = dequantize_weight(sel_qw, sel_s, block_size=256, weight_bits=1.58)

        orig = w[idx]
        # Ternary is coarse — sign correlation should be strong
        cos = torch.nn.functional.cosine_similarity(
            deq.float().view(-1), orig.float().view(-1), dim=0,
        )
        assert cos > 0.8


class TestEmbeddingDequantOnly:
    """Dequant-only correctness on packed rows."""

    def test_int4_single_row_dequant(self):
        """One row: pack → unpack → dequant matches original."""
        row = torch.randn(1, 256, dtype=torch.bfloat16)
        qw, sc = quantize_weight_per_block_int4(row)
        deq = dequantize_weight(qw, sc, block_size=32, weight_bits=4)
        cos = torch.nn.functional.cosine_similarity(
            deq.float(), row.float(), dim=-1,
        )
        assert cos > 0.99

    def test_w2_single_row_dequant(self):
        """One row W2: pack → unpack → dequant matches original."""
        row = torch.randn(1, 256, dtype=torch.bfloat16)
        qw, sc = quantize_weight_per_block_w2(row)
        deq = dequantize_weight(qw, sc, block_size=256, weight_bits=1.58)
        cos = torch.nn.functional.cosine_similarity(
            deq.float(), row.float(), dim=-1,
        )
        assert cos > 0.8
