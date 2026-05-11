"""Shared fixtures for EdgeRazor test suite."""

import logging
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn


# ──────────────────────────────────────────────
# Logging fixtures
# ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def silence_edgerazor_logs():
    """Suppress EdgeRazor logging during tests to keep output clean."""
    for name in ("EdgeRazor", "EdgeRazor.QAT", "EdgeRazor.KD"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL)
        logger.handlers.clear()
        logger.propagate = True


@pytest.fixture
def temp_dir():
    """Create a temporary directory that cleans up after test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ──────────────────────────────────────────────
# Model fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def simple_linear_model():
    """A minimal nn.Module with a single Linear layer."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 8)

        def forward(self, x):
            return self.fc(x)

    return SimpleModel()


@pytest.fixture
def simple_multi_layer_model():
    """A small nn.Module with Linear, Embedding, and Conv2d layers."""
    class MultiLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 32)
            self.conv = nn.Conv2d(3, 16, kernel_size=3)
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    return MultiLayerModel()


@pytest.fixture
def simple_cnn_model():
    """A model with Conv1d, Conv2d, Conv3d layers."""
    class CNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1d = nn.Conv1d(8, 16, kernel_size=3)
            self.conv2d = nn.Conv2d(3, 8, kernel_size=3)
            self.conv3d = nn.Conv3d(2, 4, kernel_size=3)

        def forward(self, x):
            return x

    return CNNModel()


# ──────────────────────────────────────────────
# Configuration fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def basic_qat_config_dict():
    """Minimal QAT configuration dictionary."""
    return {
        "method": "QAT",
        "select": {
            "target_types": ["linear"],
            "target_names": [],
            "exclude_types": [],
            "exclude_names": [],
        },
        "function": {
            "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
            "w_scale_factor": 2.0,
            "w_block_size": 256,
            "w_mixed_precision_prop": -1.0,
            "is_w_quantized": True,
            "activation_function": "",
            "a_block_size": -1,
            "a_mixed_precision_prop": -1.0,
            "kv_cache_function": "",
            "kv_block_size": -1,
            "kv_mixed_precision_prop": -1.0,
        },
        "training": "all",
    }


@pytest.fixture
def basic_kd_config_dict():
    """Minimal KD configuration dictionary."""
    return {
        "method": "KD",
        "loss_task_alpha": 1.0,
        "loss_1": {
            "loss_type": "logits",
            "loss_function": "compute_kld_reverse",
            "alpha": 0.5,
            "temperature": 2.0,
        },
    }


@pytest.fixture
def unified_config_dict():
    """Unified EdgeRazorConfig dictionary with both QAT and KD."""
    return {
        "qat_configuration": {
            "method": "QAT",
            "select": {
                "target_types": ["linear"],
                "target_names": [],
                "exclude_types": [],
                "exclude_names": [],
            },
            "function": {
                "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
                "w_scale_factor": 2.0,
                "w_block_size": 256,
                "w_mixed_precision_prop": -1.0,
                "is_w_quantized": True,
                "activation_function": "",
                "a_block_size": -1,
                "a_mixed_precision_prop": -1.0,
                "kv_cache_function": "",
                "kv_block_size": -1,
                "kv_mixed_precision_prop": -1.0,
            },
            "training": "all",
        },
        "kd_configuration": {
            "method": "KD",
            "loss_1": {
                "loss_type": "logits",
                "loss_function": "compute_kld_reverse",
                "alpha": 0.5,
                "temperature": 2.0,
            },
        },
    }


# ──────────────────────────────────────────────
# Tensor fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def dummy_logits():
    """Create dummy logits tensor: (batch=2, seq=4, vocab=10)."""
    return torch.randn(2, 4, 10)


@pytest.fixture
def dummy_labels():
    """Create dummy labels tensor: (batch=2, seq=4)."""
    return torch.randint(0, 10, (2, 4))


@pytest.fixture
def dummy_student_outputs():
    """Create student outputs dict with loss and logits."""
    logits = torch.randn(2, 4, 10)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, 10), torch.randint(0, 10, (2 * 4,))
    )
    return {"loss": loss, "logits": logits}


@pytest.fixture
def dummy_teacher_outputs():
    """Create teacher outputs dict with logits."""
    return {"logits": torch.randn(2, 4, 10)}
