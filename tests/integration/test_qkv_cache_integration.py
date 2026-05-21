"""Integration tests for KV Cache quantization pipeline.

Tests the full KV Cache QAT pipeline:
1. Create model + quant config
2. Wrap DynamicCache with QuantizedKVState
3. Forward pass with wrapped cache
4. Backward pass (STE gradients)
5. Optimizer step
6. Save/load model with quantized KV cache
"""

import pytest
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from edgerazor.qat.block.qkv_cache import QuantizedKVState, create_quantized_kv_cache


# ──────────────────────────────────────────────
# Helper: simple attention model
# ──────────────────────────────────────────────

class _SimpleAttentionModel(nn.Module):
    """A minimal model that mimics the transformers attention + cache pattern.

    Uses the same structure as transformers models:
    - k_proj, v_proj, q_proj (nn.Linear)
    - RoPE-like position encoding
    - DynamicCache.update()
    - Scaled dot-product attention
    """

    def __init__(self, hidden_size=64, num_heads=8, head_dim=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)

    def forward(self, hidden_states, past_key_values=None, layer_idx=0):
        bsz, q_len, _ = hidden_states.shape
        hidden_shape = (bsz, q_len, self.num_heads, self.head_dim)

        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, layer_idx)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


# ──────────────────────────────────────────────
# Quantization function
# ──────────────────────────────────────────────

def _ternary_quant(x, epsilon=1e-5, block_size=64):
    """Simple ternary quantization for testing."""
    scale = x.abs().max() / 2.0
    q = torch.zeros_like(x)
    q[x > scale * 0.25] = scale
    q[x < -scale * 0.25] = -scale
    return q


# ──────────────────────────────────────────────
# Full pipeline tests
# ──────────────────────────────────────────────

class TestKVCacheQATFullPipeline:
    """End-to-end KV cache quantization pipeline."""

    def test_forward_with_wrapped_cache(self):
        model = _SimpleAttentionModel()
        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        hidden = torch.randn(2, 16, 64)
        out = model(hidden, past_key_values=wrapped, layer_idx=0)
        assert out.shape == (2, 16, 64)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_without_cache_unchanged(self):
        """Without cache, model should work normally (no crash)."""
        model = _SimpleAttentionModel()
        hidden = torch.randn(2, 16, 64)
        out = model(hidden, past_key_values=None, layer_idx=0)
        assert out.shape == (2, 16, 64)

    def test_gradients_flow_through_wrapped_cache(self):
        model = _SimpleAttentionModel()
        model.train()

        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        hidden = torch.randn(2, 16, 64)
        out = model(hidden, past_key_values=wrapped, layer_idx=0)
        loss = out.sum()
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"
            assert p.grad.abs().sum() > 0, f"{name} has zero gradient"

    def test_training_step_updates_parameters(self):
        model = _SimpleAttentionModel()
        model.train()

        params_before = {n: p.clone() for n, p in model.named_parameters()}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for _ in range(5):
            optimizer.zero_grad()
            cache = DynamicCache()
            wrapped = QuantizedKVState(
                cache, quant_fn=_ternary_quant,
                kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
            )
            hidden = torch.randn(2, 16, 64)
            out = model(hidden, past_key_values=wrapped, layer_idx=0)
            loss = out.sum()
            loss.backward()
            optimizer.step()

        changed = 0
        for n, p in model.named_parameters():
            if not torch.equal(p, params_before[n]):
                changed += 1
        # At least weight params should change; biases may have small gradients
        assert changed >= 4, f"Only {changed}/8 params changed"

    def test_multi_layer_training(self):
        """Test with multiple attention layers."""
        class _MultiLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_0 = _SimpleAttentionModel()
                self.layer_1 = _SimpleAttentionModel()

            def forward(self, hidden, past_key_values=None):
                h = self.layer_0(hidden, past_key_values=past_key_values, layer_idx=0)
                h = torch.relu(h)
                h = self.layer_1(h, past_key_values=past_key_values, layer_idx=1)
                return h

        model = _MultiLayerModel()
        model.train()

        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        hidden = torch.randn(2, 16, 64)
        out = model(hidden, past_key_values=wrapped)
        loss = out.sum()
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"


class TestKVCacheQATSaveLoad:
    """Save/load with KV cache quantization."""

    def test_save_load_after_kv_cache_training(self, temp_dir):
        model = _SimpleAttentionModel()
        model.train()

        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )
        hidden = torch.randn(2, 16, 64)
        out = model(hidden, past_key_values=wrapped, layer_idx=0)
        loss = out.sum()
        loss.backward()

        save_path = temp_dir / "kv_quant_model.pt"
        torch.save(model.state_dict(), save_path)

        new_model = _SimpleAttentionModel()
        new_model.load_state_dict(torch.load(save_path))

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            torch.testing.assert_close(p1, p2)


class TestKVCacheQATAutoRegressive:
    """Simulate autoregressive generation with KV cache quantization."""

    def test_autoregressive_generation(self):
        model = _SimpleAttentionModel()
        model.eval()

        cache = DynamicCache()
        wrapped = QuantizedKVState(
            cache, quant_fn=_ternary_quant,
            kv_kwargs={'epsilon': 1e-5, 'block_size': 64}
        )

        with torch.no_grad():
            # Prefill: full sequence
            hidden = torch.randn(1, 8, 64)
            out = model(hidden, past_key_values=wrapped, layer_idx=0)

            # Decode: one token at a time
            for _ in range(4):
                hidden = torch.randn(1, 1, 64)
                out = model(hidden, past_key_values=wrapped, layer_idx=0)

        assert out.shape == (1, 1, 64)
        # After 8 prefill + 4 decode steps, cache should have 12 tokens
        assert wrapped.get_seq_length(0) == 12


class TestCreateQuantizedKVCache:
    """Test the helper factory function."""

    def test_create_with_config(self):
        from edgerazor.qat.util.quant_config import QuantConfig

        config = QuantConfig({
            "method": "QAT",
            "select": {"target_types": ["linear"],
                       "target_names": [], "exclude_types": [], "exclude_names": []},
            "function": {
                "epsilon": 1e-5,
                "weight_function": "weight_quant_uniform_symmetric_clip_per_block_int1_58",
                "w_scale_factor": 2.0,
                "w_block_size": 64,
                "w_mixed_precision_prop": -1.0,
                "is_w_quantized": False,
                "activation_function": "",
                "a_block_size": -1,
                "a_mixed_precision_prop": -1.0,
                "kv_cache_function": "state_quant_uniform_symmetric_absmax_per_block_int8",
                "kv_block_size": 128,
                "kv_mixed_precision_prop": -1.0,
            },
            "training": "all",
        })

        kv_cache = create_quantized_kv_cache(config)
        assert isinstance(kv_cache, QuantizedKVState)
        assert isinstance(kv_cache._cache, DynamicCache)

    def test_create_without_kv_config_returns_none(self):
        from edgerazor.qat.util.quant_config import QuantConfig

        config = QuantConfig({
            "method": "QAT",
            "select": {"target_types": ["linear"],
                       "target_names": [], "exclude_types": [], "exclude_names": []},
            "function": {
                "epsilon": 1e-5,
                "weight_function": "weight_quant_uniform_symmetric_clip_per_block_int1_58",
                "w_scale_factor": 2.0,
                "w_block_size": 64,
                "w_mixed_precision_prop": -1.0,
                "is_w_quantized": False,
                "activation_function": "",
                "a_block_size": -1,
                "a_mixed_precision_prop": -1.0,
                "kv_cache_function": "",  # empty — no KV quant
                "kv_block_size": -1,
                "kv_mixed_precision_prop": -1.0,
            },
            "training": "all",
        })

        kv_cache = create_quantized_kv_cache(config)
        assert kv_cache is None
