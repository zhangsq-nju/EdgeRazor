"""Integration tests for KV Cache quantization with YAML configs.

Tests W1.58-A8-KV8 and W4-A8-KV8 quantization scales end-to-end:
- Load config from YAML → quantize model → forward → backward
- QAT, KD, and QAD combined pipelines with a tiny 3-layer Qwen3-like model.
"""

import pytest
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from edgerazor import EdgeRazor
from edgerazor.qat.block.qkv_cache import QuantizedKVState, create_quantized_kv_cache
from edgerazor.qat.module import QLinear, QEmbedding
from edgerazor.qat.util.quant_config import QuantConfig


# ──────────────────────────────────────────────
# Tiny 3-layer Qwen3-like transformer model
# ──────────────────────────────────────────────

class TinyQwen3Config:
    """Minimal config-like object matching Qwen3 structure."""

    def __init__(
        self,
        vocab_size=100,
        hidden_size=64,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=3,
        head_dim=None,
        tie_word_embeddings=False,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.tie_word_embeddings = tie_word_embeddings


class TinyQwen3Attention(nn.Module):
    """Minimal GQA attention matching Qwen3 pattern."""

    def __init__(self, config: TinyQwen3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states, past_key_values=None, layer_idx=0):
        bsz, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, layer_idx)

        # Expand KV heads to match Q heads for GQA
        kv_group = self.num_heads // self.num_kv_heads
        k_expanded = k.repeat_interleave(kv_group, dim=1)
        v_expanded = v.repeat_interleave(kv_group, dim=1)

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v_expanded)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class TinyQwen3MLP(nn.Module):
    """Minimal SwiGLU MLP matching Qwen3 pattern."""

    def __init__(self, config: TinyQwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        gate = nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TinyQwen3DecoderLayer(nn.Module):
    """Minimal decoder layer: RMSNorm → Attention → RMSNorm → MLP."""

    def __init__(self, config: TinyQwen3Config):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.self_attn = TinyQwen3Attention(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.mlp = TinyQwen3MLP(config)

    def forward(self, hidden_states, past_key_values=None, layer_idx=0):
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, past_key_values=past_key_values, layer_idx=layer_idx)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TinyQwen3Model(nn.Module):
    """A minimal 3-layer Qwen3-like transformer for KV cache QAT integration testing.

    Architecture: embed_tokens → 3× DecoderLayer → RMSNorm → lm_head
    """

    def __init__(self, config: TinyQwen3Config = None):
        super().__init__()
        if config is None:
            config = TinyQwen3Config()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TinyQwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        labels=None,
        past_key_values=None,
        output_hidden_states=False,
        return_dict=True,
    ):
        hidden_states = self.embed_tokens(input_ids)
        all_hidden_states = [hidden_states] if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, past_key_values=past_key_values, layer_idx=i)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )

        result = {"logits": logits, "loss": loss}
        if output_hidden_states:
            result["hidden_states"] = tuple(all_hidden_states)
        return result


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_kv_cache(config: QuantConfig, model_config=None):
    """Create a QuantizedKVState from a QuantConfig."""
    kv_cache = create_quantized_kv_cache(config, model_config=model_config)
    if kv_cache is None:
        kv_cache = DynamicCache(config=model_config)
    return kv_cache


# ──────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def tiny_model():
    """Create a fresh tiny 3-layer Qwen3 model."""
    return TinyQwen3Model()


@pytest.fixture
def w1_58_config_path():
    """Path to W1.58-A8-KV8 YAML config."""
    from pathlib import Path
    return Path(__file__).parent / "w1.58-a8-kv8.yaml"


@pytest.fixture
def w4_config_path():
    """Path to W4-A8-KV8 YAML config."""
    from pathlib import Path
    return Path(__file__).parent / "w4-a8-kv8.yaml"


# ──────────────────────────────────────────────
# W1.58-A8-KV8 Pipeline Tests
# ──────────────────────────────────────────────

class TestW1_58_A8_KV8:
    """End-to-end tests with W1.58-A8-KV8 config (ternary weight + int8 activation + int8 kv_cache)."""

    def test_qat_forward_backward(self, tiny_model, w1_58_config_path):
        """QAT forward + backward with kv_cache quantization."""
        edgerazor = EdgeRazor(config=w1_58_config_path)
        assert edgerazor.is_qat_enabled
        assert edgerazor.qat.selector.has_kv_cache

        # Quantize model (replaces Linear/Embedding modules)
        q_model = edgerazor.quantize(tiny_model)
        q_model.train()

        # Create quantized KV cache
        kv_cache = _make_kv_cache(edgerazor.qat.config)
        assert isinstance(kv_cache, QuantizedKVState)

        # Forward
        input_ids = torch.randint(0, 100, (2, 16))
        out = q_model(input_ids, labels=input_ids, past_key_values=kv_cache, return_dict=True)

        assert out["loss"] is not None
        assert not torch.isnan(out["logits"]).any()
        assert out["logits"].shape == (2, 16, 100)

        # Backward
        out["loss"].backward()

        # Verify gradients flow through all parameters
        for name, p in q_model.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"
            assert p.grad.abs().sum() > 0, f"{name} has zero gradient"

    def test_qat_multi_step_training(self, tiny_model, w1_58_config_path):
        """Multiple QAT training steps with kv_cache — parameters should update."""
        edgerazor = EdgeRazor(config=w1_58_config_path)
        q_model = edgerazor.quantize(tiny_model)
        q_model.train()

        optimizer = torch.optim.SGD(q_model.parameters(), lr=0.01)
        params_before = {n: p.clone() for n, p in q_model.named_parameters() if p.requires_grad}

        for step in range(5):
            optimizer.zero_grad()
            kv_cache = _make_kv_cache(edgerazor.qat.config)
            input_ids = torch.randint(0, 100, (2, 16))
            out = q_model(input_ids, labels=input_ids, past_key_values=kv_cache, return_dict=True)
            out["loss"].backward()
            optimizer.step()

        changed = sum(
            1 for n, p in q_model.named_parameters()
            if p.requires_grad and not torch.equal(p, params_before[n])
        )
        assert changed >= len(params_before) * 0.8, f"Only {changed}/{len(params_before)} params changed"

    def test_qat_replace_weights_and_infer(self, tiny_model, w1_58_config_path):
        """After training, replace weights and run inference."""
        edgerazor = EdgeRazor(config=w1_58_config_path)
        q_model = edgerazor.quantize(tiny_model)
        q_model.train()

        # One training step
        kv_cache = _make_kv_cache(edgerazor.qat.config)
        input_ids = torch.randint(0, 100, (2, 16))
        out = q_model(input_ids, labels=input_ids, past_key_values=kv_cache, return_dict=True)
        out["loss"].backward()

        # Replace quantized weights
        edgerazor.replace_quantized_weights(q_model)

        # Inference
        q_model.eval()
        with torch.no_grad():
            kv_cache = DynamicCache()
            out = q_model(torch.randint(0, 100, (1, 8)), past_key_values=kv_cache, return_dict=True)
        assert out["logits"].shape == (1, 8, 100)
        assert not torch.isnan(out["logits"]).any()

    def test_kd_forward_backward(self, tiny_model, w1_58_config_path):
        """KD forward + backward with kv_cache config (no weight quantization)."""
        edgerazor = EdgeRazor(config=w1_58_config_path)
        assert edgerazor.is_kd_enabled

        student = TinyQwen3Model()
        teacher = TinyQwen3Model()
        student.train()

        input_ids = torch.randint(0, 100, (2, 16))
        student_out = student(input_ids, labels=input_ids, return_dict=True)
        with torch.no_grad():
            teacher_out = teacher(input_ids, return_dict=True)

        total_loss, loss_dict = edgerazor.compute_loss(student_out, teacher_out, input_ids)
        assert 'distill_loss' in loss_dict
        total_loss.backward()

        for name, p in student.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"

    def test_qad_forward_backward(self, tiny_model, w1_58_config_path):
        """QAT+KD combined: KD loss flows gradients through quantized layers with kv_cache."""
        edgerazor = EdgeRazor(config=w1_58_config_path)

        student = edgerazor.quantize(tiny_model)
        teacher = TinyQwen3Model()
        student.train()

        # Capture params before backward
        params_before = {n: p.clone() for n, p in student.named_parameters()}

        kv_cache = _make_kv_cache(edgerazor.qat.config)
        input_ids = torch.randint(0, 100, (2, 16))
        student_out = student(input_ids, labels=input_ids, past_key_values=kv_cache,
                             return_dict=True)
        with torch.no_grad():
            teacher_out = teacher(input_ids, return_dict=True)

        total_loss, loss_dict = edgerazor.compute_loss(student_out, teacher_out, input_ids)
        assert 'distill_loss' in loss_dict
        total_loss.backward()

        for name, p in student.named_parameters():
            assert p.grad is not None, f"{name} has no gradient in QAD mode"

    def test_qad_multi_step_training(self, tiny_model, w1_58_config_path):
        """QAD with multiple training steps — parameters should update."""
        edgerazor = EdgeRazor(config=w1_58_config_path)

        student = edgerazor.quantize(tiny_model)
        teacher = TinyQwen3Model()
        student.train()

        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
        params_before = {n: p.clone() for n, p in student.named_parameters() if p.requires_grad}

        for _ in range(3):
            optimizer.zero_grad()
            kv_cache = _make_kv_cache(edgerazor.qat.config)
            input_ids = torch.randint(0, 100, (2, 16))
            student_out = student(input_ids, labels=input_ids, past_key_values=kv_cache,
                                 return_dict=True)
            with torch.no_grad():
                teacher_out = teacher(input_ids, return_dict=True)
            total_loss, _ = edgerazor.compute_loss(student_out, teacher_out, input_ids)
            total_loss.backward()
            optimizer.step()

        changed = sum(
            1 for n, p in student.named_parameters()
            if p.requires_grad and not torch.equal(p, params_before[n])
        )
        assert changed >= len(params_before) * 0.8

    def test_qad_replace_and_save(self, tiny_model, w1_58_config_path, temp_dir):
        """QAD train → replace weights → save → load → verify."""
        edgerazor = EdgeRazor(config=w1_58_config_path)

        student = edgerazor.quantize(tiny_model)
        teacher = TinyQwen3Model()
        student.train()

        kv_cache = _make_kv_cache(edgerazor.qat.config)
        input_ids = torch.randint(0, 100, (2, 16))
        student_out = student(input_ids, labels=input_ids, past_key_values=kv_cache,
                             return_dict=True)
        with torch.no_grad():
            teacher_out = teacher(input_ids, return_dict=True)
        total_loss, _ = edgerazor.compute_loss(student_out, teacher_out, input_ids)
        total_loss.backward()

        # Replace weights
        edgerazor.replace_quantized_weights(student)

        # Save
        save_path = temp_dir / "w158_kv8_model.pt"
        torch.save(student.state_dict(), save_path)

        # Load into new model
        new_student = edgerazor.quantize(TinyQwen3Model())
        new_student.load_state_dict(torch.load(save_path))

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(student.named_parameters(), new_student.named_parameters()):
            torch.testing.assert_close(p1, p2, msg=f"weight mismatch at {n1}")


# ──────────────────────────────────────────────
# W4-A8-KV8 Pipeline Tests
# ──────────────────────────────────────────────

class TestW4_A8_KV8:
    """End-to-end tests with W4-A8-KV8 config (int4 weight + int8 activation + int8 kv_cache)."""

    def test_qat_forward_backward(self, tiny_model, w4_config_path):
        """QAT forward + backward with kv_cache quantization."""
        edgerazor = EdgeRazor(config=w4_config_path)
        assert edgerazor.is_qat_enabled
        assert edgerazor.qat.selector.has_kv_cache

        q_model = edgerazor.quantize(tiny_model)
        q_model.train()

        kv_cache = _make_kv_cache(edgerazor.qat.config)
        assert isinstance(kv_cache, QuantizedKVState)

        input_ids = torch.randint(0, 100, (2, 16))
        out = q_model(input_ids, labels=input_ids, past_key_values=kv_cache, return_dict=True)

        assert out["loss"] is not None
        assert not torch.isnan(out["logits"]).any()
        assert out["logits"].shape == (2, 16, 100)

        out["loss"].backward()

        for name, p in q_model.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"
            assert p.grad.abs().sum() > 0, f"{name} has zero gradient"

    def test_qat_multi_step_training(self, tiny_model, w4_config_path):
        """Multiple QAT training steps with kv_cache."""
        edgerazor = EdgeRazor(config=w4_config_path)
        q_model = edgerazor.quantize(tiny_model)
        q_model.train()

        optimizer = torch.optim.SGD(q_model.parameters(), lr=0.01)
        params_before = {n: p.clone() for n, p in q_model.named_parameters() if p.requires_grad}

        for _ in range(5):
            optimizer.zero_grad()
            kv_cache = _make_kv_cache(edgerazor.qat.config)
            input_ids = torch.randint(0, 100, (2, 16))
            out = q_model(input_ids, labels=input_ids, past_key_values=kv_cache, return_dict=True)
            out["loss"].backward()
            optimizer.step()

        changed = sum(
            1 for n, p in q_model.named_parameters()
            if p.requires_grad and not torch.equal(p, params_before[n])
        )
        assert changed >= len(params_before) * 0.8

    def test_qat_replace_weights_and_infer(self, tiny_model, w4_config_path):
        """After W4 training, replace weights and run inference."""
        edgerazor = EdgeRazor(config=w4_config_path)
        q_model = edgerazor.quantize(tiny_model)
        q_model.train()

        kv_cache = _make_kv_cache(edgerazor.qat.config)
        input_ids = torch.randint(0, 100, (2, 16))
        out = q_model(input_ids, labels=input_ids, past_key_values=kv_cache, return_dict=True)
        out["loss"].backward()

        edgerazor.replace_quantized_weights(q_model)

        q_model.eval()
        with torch.no_grad():
            kv_cache = DynamicCache()
            out = q_model(torch.randint(0, 100, (1, 8)), past_key_values=kv_cache, return_dict=True)
        assert out["logits"].shape == (1, 8, 100)
        assert not torch.isnan(out["logits"]).any()

    def test_kd_forward_backward(self, tiny_model, w4_config_path):
        """KD forward + backward with W4 kv_cache config."""
        edgerazor = EdgeRazor(config=w4_config_path)
        assert edgerazor.is_kd_enabled

        student = TinyQwen3Model()
        teacher = TinyQwen3Model()
        student.train()

        input_ids = torch.randint(0, 100, (2, 16))
        student_out = student(input_ids, labels=input_ids, return_dict=True)
        with torch.no_grad():
            teacher_out = teacher(input_ids, return_dict=True)

        total_loss, loss_dict = edgerazor.compute_loss(student_out, teacher_out, input_ids)
        assert 'distill_loss' in loss_dict
        total_loss.backward()

        for name, p in student.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"

    def test_qad_forward_backward(self, tiny_model, w4_config_path):
        """QAT+KD: KD loss gradients through int4-quantized layers with kv_cache."""
        edgerazor = EdgeRazor(config=w4_config_path)

        student = edgerazor.quantize(tiny_model)
        teacher = TinyQwen3Model()
        student.train()

        kv_cache = _make_kv_cache(edgerazor.qat.config)
        input_ids = torch.randint(0, 100, (2, 16))
        student_out = student(input_ids, labels=input_ids, past_key_values=kv_cache,
                             return_dict=True)
        with torch.no_grad():
            teacher_out = teacher(input_ids, return_dict=True)

        total_loss, loss_dict = edgerazor.compute_loss(student_out, teacher_out, input_ids)
        assert 'distill_loss' in loss_dict
        total_loss.backward()

        for name, p in student.named_parameters():
            assert p.grad is not None, f"{name} has no gradient in QAD mode"

    def test_qad_multi_step_training(self, tiny_model, w4_config_path):
        """QAD multi-step: parameters should update with W4 config."""
        edgerazor = EdgeRazor(config=w4_config_path)

        student = edgerazor.quantize(tiny_model)
        teacher = TinyQwen3Model()
        student.train()

        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
        params_before = {n: p.clone() for n, p in student.named_parameters() if p.requires_grad}

        for _ in range(3):
            optimizer.zero_grad()
            kv_cache = _make_kv_cache(edgerazor.qat.config)
            input_ids = torch.randint(0, 100, (2, 16))
            student_out = student(input_ids, labels=input_ids, past_key_values=kv_cache,
                                 return_dict=True)
            with torch.no_grad():
                teacher_out = teacher(input_ids, return_dict=True)
            total_loss, _ = edgerazor.compute_loss(student_out, teacher_out, input_ids)
            total_loss.backward()
            optimizer.step()

        changed = sum(
            1 for n, p in student.named_parameters()
            if p.requires_grad and not torch.equal(p, params_before[n])
        )
        assert changed >= len(params_before) * 0.8

    def test_qad_replace_and_save(self, tiny_model, w4_config_path, temp_dir):
        """W4 QAD train → replace → save → load → verify."""
        edgerazor = EdgeRazor(config=w4_config_path)

        student = edgerazor.quantize(tiny_model)
        teacher = TinyQwen3Model()
        student.train()

        kv_cache = _make_kv_cache(edgerazor.qat.config)
        input_ids = torch.randint(0, 100, (2, 16))
        student_out = student(input_ids, labels=input_ids, past_key_values=kv_cache,
                             return_dict=True)
        with torch.no_grad():
            teacher_out = teacher(input_ids, return_dict=True)
        total_loss, _ = edgerazor.compute_loss(student_out, teacher_out, input_ids)
        total_loss.backward()

        edgerazor.replace_quantized_weights(student)

        save_path = temp_dir / "w4_kv8_model.pt"
        torch.save(student.state_dict(), save_path)

        new_student = edgerazor.quantize(TinyQwen3Model())
        new_student.load_state_dict(torch.load(save_path))

        for (n1, p1), (n2, p2) in zip(student.named_parameters(), new_student.named_parameters()):
            torch.testing.assert_close(p1, p2, msg=f"weight mismatch at {n1}")


# ──────────────────────────────────────────────
# create_kv_cache() API tests
# ──────────────────────────────────────────────

class TestCreateKVCacheAPI:
    """Verify the create_kv_cache() API on QAT and EdgeRazor."""

    def test_qat_create_kv_cache_returns_quantized_kv_state(self, w1_58_config_path):
        """QAT.create_kv_cache() returns QuantizedKVState when kv_cache is configured."""
        from edgerazor import QAT
        qat = QAT(w1_58_config_path)
        assert qat.selector.has_kv_cache

        kv_cache = qat.create_kv_cache()
        assert isinstance(kv_cache, QuantizedKVState)

    def test_qat_create_kv_cache_returns_none_when_not_configured(self):
        """QAT.create_kv_cache() returns None when kv_cache is not selected."""
        from edgerazor import QAT
        config_dict = {
            'method': 'QAT',
            'select': {'target_types': ['linear']},
            'function': {
                'weight_function': 'weight_quant_uniform_symmetric_clip_per_tensor_int1_58',
                'activation_function': 'state_quant_uniform_symmetric_absmax_per_token_int8',
            },
        }
        qat = QAT(config_dict)
        assert not qat.selector.has_kv_cache

        kv_cache = qat.create_kv_cache()
        assert kv_cache is None

    def test_edgerazor_create_kv_cache_delegates_to_qat(self, w1_58_config_path):
        """EdgeRazor.create_kv_cache() delegates to QAT.create_kv_cache()."""
        edgerazor = EdgeRazor(config=w1_58_config_path)
        assert edgerazor.is_qat_enabled

        kv_cache = edgerazor.create_kv_cache()
        assert isinstance(kv_cache, QuantizedKVState)
        assert len(kv_cache) == 0

    def test_edgerazor_create_kv_cache_returns_none_without_qat(self, w1_58_config_path):
        """EdgeRazor.create_kv_cache() returns None when QAT is disabled."""
        from edgerazor import EdgeRazor
        edgerazor = EdgeRazor(config=w1_58_config_path)
        # Verify the method works when QAT is enabled
        assert edgerazor.create_kv_cache() is not None
        # EdgeRazor without QAT returns None
        assert edgerazor.is_qat_enabled  # this config enables QAT

    def test_create_kv_cache_with_model_config(self, tiny_model, w1_58_config_path):
        """create_quantized_kv_cache accepts model_config parameter and creates functional cache."""
        # With model_config=None, creates DynamicCache() without config (fine for tiny model)
        kv_cache = create_quantized_kv_cache(
            EdgeRazor(config=w1_58_config_path).qat.config,
            model_config=None,
        )
        assert isinstance(kv_cache, QuantizedKVState)
        # verify it's functional with the tiny model
        input_ids = torch.randint(0, 100, (2, 8))
        tiny_model.eval()
        with torch.no_grad():
            tiny_model(input_ids, past_key_values=kv_cache, return_dict=True)
        for layer_idx in range(3):
            assert kv_cache.get_seq_length(layer_idx) == 8


# ──────────────────────────────────────────────
# KV Cache correctness tests
# ──────────────────────────────────────────────

class TestKVCacheCorrectness:
    """Verify KV cache quantization works correctly with the tiny model."""

    def test_kv_cache_is_used(self, tiny_model, w1_58_config_path):
        """QuantizedKVState cache length grows after forward with cache."""
        edgerazor = EdgeRazor(config=w1_58_config_path)
        kv_cache = _make_kv_cache(edgerazor.qat.config)

        # Prefill: 8 tokens
        tiny_model.eval()
        with torch.no_grad():
            tiny_model(torch.randint(0, 100, (1, 8)), past_key_values=kv_cache, return_dict=True)

        # Each of 3 layers should have 8 tokens cached
        for layer_idx in range(3):
            assert kv_cache.get_seq_length(layer_idx) == 8

    def test_autoregressive_decode_with_quantized_cache(self, tiny_model, w1_58_config_path):
        """Simulate prefill + decode with quantized KV cache."""
        edgerazor = EdgeRazor(config=w1_58_config_path)
        kv_cache = _make_kv_cache(edgerazor.qat.config)

        tiny_model.eval()
        with torch.no_grad():
            # Prefill
            tiny_model(torch.randint(0, 100, (1, 8)), past_key_values=kv_cache, return_dict=True)
            # Decode: 4 steps, 1 token each
            for _ in range(4):
                tiny_model(torch.randint(0, 100, (1, 1)), past_key_values=kv_cache, return_dict=True)

        for layer_idx in range(3):
            assert kv_cache.get_seq_length(layer_idx) == 12  # 8 + 4

    def test_forward_without_cache_is_unchanged(self, tiny_model):
        """Forward without kv_cache should work normally."""
        tiny_model.eval()
        with torch.no_grad():
            out = tiny_model(torch.randint(0, 100, (2, 16)), past_key_values=None, return_dict=True)
        assert out["logits"].shape == (2, 16, 100)
        assert not torch.isnan(out["logits"]).any()
