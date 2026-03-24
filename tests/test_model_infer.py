"""Tests for integrations.hf.model_infer — RSRLinear, load/save roundtrip, CLI."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from integrations.hf.model_infer import (
    RSRLinear,
    _bitnet_act_quant,
    _detect_device_from_dir,
    _resolve_module,
    _set_module,
    parse_args,
)
from integrations.hf.model_prep import (
    preprocess_layer_cpu,
    preprocess_layer_cuda,
    save_preprocessed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ternary_weight(n_rows, n_cols):
    return torch.randint(-1, 2, (n_rows, n_cols), dtype=torch.int8)


def _make_cpu_rsr_linear(n_rows, n_cols, k=8, weight_scale_mode="multiply",
                          weight_scale_val=None):
    """Build an RSRLinear from a random ternary matrix (CPU backend)."""
    w = _make_ternary_weight(n_rows, n_cols)
    arrays = preprocess_layer_cpu(w, k=k)
    meta = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "k": k,
        "backend": "cpu",
        "weight_scale_mode": weight_scale_mode,
    }
    ws = torch.tensor([weight_scale_val]) if weight_scale_val is not None else None
    return RSRLinear("test_layer", meta, arrays, weight_scale=ws), w


def _make_cuda_rsr_linear(n_rows, n_cols, k=8, weight_scale_mode="multiply",
                           weight_scale_val=None):
    """Build an RSRLinear from a random ternary matrix (CUDA backend)."""
    w = _make_ternary_weight(n_rows, n_cols)
    arrays = preprocess_layer_cuda(w, k=k)
    meta = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "k": k,
        "backend": "cuda",
        "n_rows_padded": ((n_rows + k - 1) // k) * k,
        "num_blocks": ((n_rows + k - 1) // k),
        "weight_scale_mode": weight_scale_mode,
    }
    ws = torch.tensor([weight_scale_val]) if weight_scale_val is not None else None
    return RSRLinear("test_layer", meta, arrays, weight_scale=ws), w


# ---------------------------------------------------------------------------
# _bitnet_act_quant
# ---------------------------------------------------------------------------

class TestBitnetActQuant:
    def test_output_shape(self):
        x = torch.randn(32)
        out = _bitnet_act_quant(x)
        assert out.shape == x.shape

    def test_preserves_dtype(self):
        x = torch.randn(32, dtype=torch.float32)
        assert _bitnet_act_quant(x).dtype == torch.float32

    def test_zero_input(self):
        x = torch.zeros(16)
        out = _bitnet_act_quant(x)
        torch.testing.assert_close(out, x)

    def test_bounded_values(self):
        """Quantized values should be close to the original."""
        x = torch.randn(128)
        out = _bitnet_act_quant(x)
        # The relative error should be bounded
        assert (out - x).abs().max() < x.abs().max() * 0.02


# ---------------------------------------------------------------------------
# RSRLinear (CPU)
# ---------------------------------------------------------------------------

class TestRSRLinearCPU:
    def test_forward_shape(self):
        layer, _ = _make_cpu_rsr_linear(32, 48)
        x = torch.randn(48)
        out = layer(x)
        assert out.shape == (32,)

    def test_forward_batch(self):
        layer, _ = _make_cpu_rsr_linear(32, 48)
        x = torch.randn(4, 48)
        out = layer(x)
        assert out.shape == (4, 32)

    def test_forward_3d(self):
        layer, _ = _make_cpu_rsr_linear(32, 48)
        x = torch.randn(2, 3, 48)
        out = layer(x)
        assert out.shape == (2, 3, 32)

    def test_wrong_input_dim_raises(self):
        layer, _ = _make_cpu_rsr_linear(32, 48)
        with pytest.raises(ValueError, match="expected hidden size"):
            layer(torch.randn(2, 64))

    def test_weight_scale_multiply(self):
        """weight_scale_mode='multiply' scales output up."""
        layer, w = _make_cpu_rsr_linear(32, 48, weight_scale_mode="multiply",
                                         weight_scale_val=2.0)
        layer_no_ws, _ = _make_cpu_rsr_linear(32, 48)
        # Ensure both use the same RSR data
        layer_no_ws.multiplier = layer.multiplier
        x = torch.randn(48)
        out_ws = layer(x)
        out_no = layer_no_ws(x)
        torch.testing.assert_close(out_ws, out_no * 2.0, atol=1e-5, rtol=1e-5)

    def test_weight_scale_divide(self):
        """weight_scale_mode='divide' scales output down."""
        layer, w = _make_cpu_rsr_linear(32, 48, weight_scale_mode="divide",
                                         weight_scale_val=2.0)
        layer_no_ws, _ = _make_cpu_rsr_linear(32, 48)
        layer_no_ws.multiplier = layer.multiplier
        x = torch.randn(48)
        out_ws = layer(x)
        out_no = layer_no_ws(x)
        torch.testing.assert_close(out_ws, out_no / 2.0, atol=1e-5, rtol=1e-5)

    def test_bias(self):
        layer, _ = _make_cpu_rsr_linear(32, 48)
        bias = torch.randn(32)
        layer.register_buffer("bias", bias)
        x = torch.randn(48)
        out_biased = layer(x)
        layer.register_buffer("bias", None, persistent=False)
        out_no_bias = layer(x)
        torch.testing.assert_close(out_biased, out_no_bias + bias, atol=1e-5, rtol=1e-5)

    def test_cpu_correctness_against_matmul(self):
        """RSR CPU output should approximate W @ act_quant(x)."""
        n_rows, n_cols = 32, 48
        layer, w = _make_cpu_rsr_linear(n_rows, n_cols)
        x = torch.randn(n_cols)
        rsr_out = layer(x)
        # Reference: W @ act_quant(x)
        x_q = _bitnet_act_quant(x)
        ref = w.float() @ x_q
        torch.testing.assert_close(rsr_out, ref, atol=1e-3, rtol=1e-3)

    def test_extra_repr(self):
        layer, _ = _make_cpu_rsr_linear(32, 48)
        s = layer.extra_repr()
        assert "in_features=48" in s
        assert "out_features=32" in s


# ---------------------------------------------------------------------------
# RSRLinear (CUDA)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestRSRLinearCUDA:
    def test_forward_shape(self):
        layer, _ = _make_cuda_rsr_linear(32, 64)
        x = torch.randn(64)
        out = layer(x)
        assert out.shape == (32,)

    def test_forward_batch(self):
        layer, _ = _make_cuda_rsr_linear(32, 64)
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 32)

    def test_weight_scale_multiply(self):
        layer, _ = _make_cuda_rsr_linear(32, 64, weight_scale_mode="multiply",
                                          weight_scale_val=2.0)
        layer_no, _ = _make_cuda_rsr_linear(32, 64)
        layer_no.multiplier = layer.multiplier
        x = torch.randn(64)
        out_ws = layer(x)
        out_no = layer_no(x)
        torch.testing.assert_close(out_ws, out_no * 2.0, atol=1e-4, rtol=1e-4)

    def test_weight_scale_divide(self):
        layer, _ = _make_cuda_rsr_linear(32, 64, weight_scale_mode="divide",
                                          weight_scale_val=100.0)
        layer_no, _ = _make_cuda_rsr_linear(32, 64)
        layer_no.multiplier = layer.multiplier
        x = torch.randn(64)
        out_ws = layer(x)
        out_no = layer_no(x)
        torch.testing.assert_close(out_ws, out_no / 100.0, atol=1e-4, rtol=1e-4)

    def test_non_square_matrix(self):
        layer, _ = _make_cuda_rsr_linear(1024, 4096)
        x = torch.randn(4096)
        out = layer(x)
        assert out.shape == (1024,)


# ---------------------------------------------------------------------------
# Module resolution helpers
# ---------------------------------------------------------------------------

class TestModuleHelpers:
    def test_resolve_module(self):
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        assert _resolve_module(model, "0") is model[0]

    def test_set_module(self):
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        new_layer = nn.Linear(10, 3)
        _set_module(model, "0", new_layer)
        assert model[0] is new_layer

    def test_resolve_nested(self):
        inner = nn.ModuleDict({"proj": nn.Linear(10, 5)})
        model = nn.ModuleDict({"layer": inner})
        assert _resolve_module(model, "layer.proj") is inner["proj"]


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

class TestDetectDevice:
    def test_cuda_suffix(self):
        assert _detect_device_from_dir("/path/to/model_cuda") == "cuda"

    def test_cpu_suffix(self):
        assert _detect_device_from_dir("/path/to/model_cpu") == "cpu"

    def test_no_suffix_defaults_cpu(self):
        assert _detect_device_from_dir("/path/to/model") == "cpu"


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestInferParseArgs:
    def test_defaults(self):
        args = parse_args(["--prompt", "hello"])
        assert args.prompt == "hello"
        assert args.backend == "rsr"
        assert args.max_new_tokens == 64

    def test_hf_backend(self):
        args = parse_args(["--prompt", "hi", "--backend", "hf"])
        assert args.backend == "hf"

    def test_missing_prompt_exits(self):
        with pytest.raises(SystemExit):
            parse_args([])


# ---------------------------------------------------------------------------
# End-to-end: preprocess → save → load RSRLinear (CPU)
# ---------------------------------------------------------------------------

class TestPrepInferRoundtripCPU:
    def test_roundtrip(self):
        """Preprocess a ternary matrix, save, load into RSRLinear, run forward."""
        from safetensors import safe_open

        n_rows, n_cols, k = 32, 48, 8
        w = _make_ternary_weight(n_rows, n_cols)
        arrays = preprocess_layer_cpu(w, k=k)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "model"
            rsr_data = {"test_layer": arrays}
            layer_meta = {
                "test_layer": {
                    "n_rows": n_rows, "n_cols": n_cols, "k": k,
                    "backend": "cpu", "weight_scale_mode": "multiply",
                }
            }
            save_preprocessed(
                rsr_data, {}, layer_meta,
                model_config={}, output_dir=out,
                k=k, version="adaptive", model_name="test",
            )

            with safe_open(out / "rsr_weights.safetensors", framework="pt") as handle:
                tensors = {}
                for key in ["perms", "group_ends", "pos_masks", "neg_masks", "block_meta"]:
                    tensors[key] = handle.get_tensor(f"test_layer.{key}")

            layer = RSRLinear("test_layer", layer_meta["test_layer"], tensors)
            x = torch.randn(n_cols)
            out = layer(x)
            assert out.shape == (n_rows,)

            # Verify against direct matmul
            ref = w.float() @ _bitnet_act_quant(x)
            torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPrepInferRoundtripCUDA:
    def test_roundtrip(self):
        """Preprocess → save → load → RSRLinear forward (CUDA)."""
        from safetensors import safe_open

        n_rows, n_cols, k = 32, 64, 8
        w = _make_ternary_weight(n_rows, n_cols)
        arrays = preprocess_layer_cuda(w, k=k)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "model"
            meta = {
                "n_rows": n_rows, "n_cols": n_cols, "k": k,
                "backend": "cuda", "weight_scale_mode": "multiply",
                "n_rows_padded": ((n_rows + k - 1) // k) * k,
                "num_blocks": ((n_rows + k - 1) // k),
            }
            rsr_data = {"test_layer": arrays}
            layer_meta = {"test_layer": meta}
            save_preprocessed(
                rsr_data, {}, layer_meta,
                model_config={}, output_dir=out,
                k=k, version="adaptive", model_name="test",
            )

            with safe_open(out / "rsr_weights.safetensors", framework="pt") as handle:
                tensors = {
                    "perms": handle.get_tensor("test_layer.perms"),
                    "group_packed": handle.get_tensor("test_layer.group_packed"),
                    "block_meta": handle.get_tensor("test_layer.block_meta"),
                }

            layer = RSRLinear("test_layer", meta, tensors)
            x = torch.randn(n_cols)
            out_rsr = layer(x)
            assert out_rsr.shape == (n_rows,)

            # CUDA RSR should approximate W @ x (within quantization error)
            ref = w.float() @ x
            assert torch.allclose(out_rsr.cpu(), ref, atol=0.5, rtol=0.1)


# ---------------------------------------------------------------------------
# Weight scale mode affects output direction
# ---------------------------------------------------------------------------

class TestWeightScaleModeEffect:
    def test_multiply_vs_divide_opposite(self):
        """Multiply and divide modes produce different magnitudes."""
        n_rows, n_cols = 32, 48
        layer_mul, w = _make_cpu_rsr_linear(n_rows, n_cols,
                                             weight_scale_mode="multiply",
                                             weight_scale_val=50.0)
        layer_div, _ = _make_cpu_rsr_linear(n_rows, n_cols,
                                             weight_scale_mode="divide",
                                             weight_scale_val=50.0)
        # Use same multiplier
        layer_div.multiplier = layer_mul.multiplier
        x = torch.randn(n_cols)
        out_mul = layer_mul(x)
        out_div = layer_div(x)
        # Multiply by 50 vs divide by 50 => ratio should be ~2500
        ratio = out_mul.norm() / out_div.norm()
        assert ratio > 100  # 50^2 = 2500, but allow margin

    def test_default_mode_is_multiply(self):
        """When weight_scale_mode is absent, default to 'multiply'."""
        n_rows, n_cols = 32, 48
        w = _make_ternary_weight(n_rows, n_cols)
        arrays = preprocess_layer_cpu(w, k=8)
        meta = {"n_rows": n_rows, "n_cols": n_cols, "k": 8, "backend": "cpu"}
        ws = torch.tensor([3.0])
        layer = RSRLinear("test", meta, arrays, weight_scale=ws)
        assert layer._weight_scale_mode == "multiply"
