"""Tests for integrations.hf.model_infer — RSRLinear, load/save roundtrip, CLI."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from integrations.hf.model_infer import (
    GreenTextStreamer,
    RSRLinear,
    _bitnet_act_quant,
    _detect_device_from_dir,
    _print_inference_stats,
    _resolve_module,
    _set_module,
    parse_args,
)
from integrations.hf.model_prep import (
    preprocess_layer_cpu,
    preprocess_layer_cuda,
    save_preprocessed,
)
from multiplier.bit_1_58.cpu.rsr_runtime import select_cpu_tensor_keys


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

    def test_preserves_bfloat16_for_downstream_cuda_linear(self):
        layer, _ = _make_cuda_rsr_linear(32, 64)
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)

        out = layer(x)

        assert out.device.type == "cuda"
        assert out.dtype == torch.bfloat16

        downstream = nn.Linear(32, 16, bias=False, device="cuda", dtype=torch.bfloat16)
        y = downstream(out)
        assert y.dtype == torch.bfloat16
        assert y.shape == (4, 16)

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
                for key in select_cpu_tensor_keys(n_cols, k):
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


# ---------------------------------------------------------------------------
# GreenTextStreamer
# ---------------------------------------------------------------------------

class TestGreenTextStreamer:
    def test_output_wrapped_in_green(self, capsys):
        """on_finalized_text prints text wrapped in ANSI green codes."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        streamer = GreenTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        streamer.on_finalized_text("hello", stream_end=False)
        captured = capsys.readouterr()
        assert "\033[32m" in captured.out
        assert "hello" in captured.out
        assert "\033[0m" in captured.out

    def test_stream_end_adds_newline(self, capsys):
        """stream_end=True terminates with a newline."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        streamer = GreenTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        streamer.on_finalized_text("done", stream_end=True)
        captured = capsys.readouterr()
        assert captured.out.endswith("\n")

    def test_green_codes_present_without_tokenizer(self, capsys):
        """GreenTextStreamer wraps any text in green regardless of content."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        streamer = GreenTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        for token in ["The", " quick", " brown", " fox"]:
            streamer.on_finalized_text(token, stream_end=False)
        captured = capsys.readouterr()
        assert captured.out.count("\033[32m") == 4
        assert captured.out.count("\033[0m") == 4


# ---------------------------------------------------------------------------
# _print_inference_stats
# ---------------------------------------------------------------------------

class TestPrintInferenceStats:
    def test_contains_required_fields(self, capsys):
        _print_inference_stats(n_tokens=42, elapsed=1.234)
        out = capsys.readouterr().out
        assert "tokens" in out
        assert "time" in out
        assert "tok/s" in out

    def test_token_count_displayed(self, capsys):
        _print_inference_stats(n_tokens=100, elapsed=2.0)
        out = capsys.readouterr().out
        assert "100" in out

    def test_elapsed_time_displayed(self, capsys):
        _print_inference_stats(n_tokens=10, elapsed=3.5)
        out = capsys.readouterr().out
        assert "3.500 s" in out

    def test_throughput_displayed(self, capsys):
        _print_inference_stats(n_tokens=50, elapsed=2.0)
        out = capsys.readouterr().out
        assert "25.0" in out  # 50 / 2.0

    def test_zero_elapsed_no_crash(self, capsys):
        _print_inference_stats(n_tokens=10, elapsed=0.0)
        out = capsys.readouterr().out
        assert "tok/s" in out
        assert "inf" in out

    def test_table_borders(self, capsys):
        _print_inference_stats(n_tokens=5, elapsed=0.5)
        out = capsys.readouterr().out
        assert "┌" in out and "┐" in out
        assert "└" in out and "┘" in out
        assert "│" in out

    def test_output_is_bold_cyan(self, capsys):
        _print_inference_stats(n_tokens=10, elapsed=1.0)
        out = capsys.readouterr().out
        assert "\033[1;36m" in out  # bold cyan
        assert "\033[0m" in out     # reset after each line


# ---------------------------------------------------------------------------
# Stream header ("▶ response")
# ---------------------------------------------------------------------------

class TestStreamHeader:
    def test_header_printed_before_tokens(self, capsys, monkeypatch):
        """generate_text prints a bold-cyan '▶ response' line before streaming."""
        import integrations.hf.model_infer as mi

        # Minimal stubs so generate_text can run without a real model/tokenizer.
        fake_ids = torch.tensor([[1, 2, 3, 4]])  # 4 tokens, no prompt

        class _FakeTokenizer:
            pad_token_id = 0
            def __call__(self, prompt, return_tensors):
                return {"input_ids": fake_ids[:, :1]}  # 1-token "prompt"
            def decode(self, ids, skip_special_tokens):
                return "ok"

        class _FakeModel(torch.nn.Module):
            def parameters(self):
                return iter([torch.empty(1)])
            def generate(self, **kwargs):
                return fake_ids

        monkeypatch.setattr(mi, "_print_inference_stats", lambda *a, **k: None)

        mi.generate_text(
            _FakeModel(), _FakeTokenizer(), "hi",
            use_chat_template=False, stream=True,
        )
        out = capsys.readouterr().out
        assert "▶ response" in out
        assert "\033[1;36m" in out

    def test_header_absent_without_stream(self, capsys, monkeypatch):
        """No header is printed when stream=False."""
        import integrations.hf.model_infer as mi

        fake_ids = torch.tensor([[1, 2]])

        class _FakeTokenizer:
            pad_token_id = 0
            def __call__(self, prompt, return_tensors):
                return {"input_ids": fake_ids[:, :1]}
            def decode(self, ids, skip_special_tokens):
                return "ok"

        class _FakeModel(torch.nn.Module):
            def parameters(self):
                return iter([torch.empty(1)])
            def generate(self, **kwargs):
                return fake_ids

        monkeypatch.setattr(mi, "_print_inference_stats", lambda *a, **k: None)

        mi.generate_text(
            _FakeModel(), _FakeTokenizer(), "hi",
            use_chat_template=False, stream=False,
        )
        out = capsys.readouterr().out
        assert "▶ response" not in out
