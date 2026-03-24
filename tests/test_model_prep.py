"""Tests for integrations.hf.model_prep — weight unpacking, RSR preprocessing, save/load."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from integrations.hf.model_prep import (
    _detect_weight_scale_mode,
    _is_ternary_linear,
    get_non_quantized_params,
    get_ternary_layers,
    pack_ternary_weights,
    parse_args,
    preprocess_layer_cpu,
    preprocess_layer_cuda,
    save_preprocessed,
    unpack_ternary_weights,
)


# ---------------------------------------------------------------------------
# Unpack / pack round-trip
# ---------------------------------------------------------------------------

class TestUnpackTernaryWeights:
    def test_known_values(self):
        weights = torch.tensor(
            [
                [-1, 0, 1, -1],
                [0, 1, -1, 0],
                [1, -1, 0, 1],
                [0, 0, 0, 0],
                [-1, -1, -1, -1],
                [1, 1, 1, 1],
                [0, 1, 0, 1],
                [-1, 0, -1, 0],
            ],
            dtype=torch.int8,
        )
        packed = pack_ternary_weights(weights)
        unpacked = unpack_ternary_weights(packed)
        assert unpacked.shape == weights.shape
        assert unpacked.dtype == torch.int8
        torch.testing.assert_close(unpacked, weights)

    def test_all_zeros(self):
        w = torch.zeros(16, 8, dtype=torch.int8)
        torch.testing.assert_close(unpack_ternary_weights(pack_ternary_weights(w)), w)

    def test_all_ones(self):
        w = torch.ones(16, 8, dtype=torch.int8)
        torch.testing.assert_close(unpack_ternary_weights(pack_ternary_weights(w)), w)

    def test_all_neg_ones(self):
        w = -torch.ones(16, 8, dtype=torch.int8)
        torch.testing.assert_close(unpack_ternary_weights(pack_ternary_weights(w)), w)

    @pytest.mark.parametrize("n_rows,n_cols", [(8, 4), (32, 16), (64, 32), (128, 64)])
    def test_random_roundtrip(self, n_rows, n_cols):
        w = torch.randint(-1, 2, (n_rows, n_cols), dtype=torch.int8)
        torch.testing.assert_close(unpack_ternary_weights(pack_ternary_weights(w)), w)

    def test_packed_shape(self):
        w = torch.randint(-1, 2, (32, 16), dtype=torch.int8)
        packed = pack_ternary_weights(w)
        assert packed.shape == (8, 16)
        assert packed.dtype == torch.uint8

    def test_large_matrix(self):
        """Dimensions close to real LLM layers."""
        w = torch.randint(-1, 2, (2560, 6912), dtype=torch.int8)
        torch.testing.assert_close(unpack_ternary_weights(pack_ternary_weights(w)), w)


class TestPackTernaryWeights:
    def test_not_divisible_by_4_raises(self):
        w = torch.randint(-1, 2, (7, 4), dtype=torch.int8)
        with pytest.raises(AssertionError):
            pack_ternary_weights(w)


# ---------------------------------------------------------------------------
# Layer discovery
# ---------------------------------------------------------------------------

class TestIsTernaryLinear:
    def test_weight_scale_detected(self):
        mod = MagicMock()
        mod.weight = torch.zeros(10, 20)
        mod.weight_scale = torch.tensor(0.5)
        assert _is_ternary_linear(mod) is True

    def test_uint8_detected(self):
        mod = MagicMock()
        mod.weight = torch.zeros(5, 20, dtype=torch.uint8)
        del mod.weight_scale
        assert _is_ternary_linear(mod) is True

    def test_online_quant_detected(self):
        mod = MagicMock()
        mod.weight = torch.randn(10, 20)
        del mod.weight_scale
        mod.online_quant = True
        type(mod).__name__ = "AutoBitLinear"
        assert _is_ternary_linear(mod) is True

    def test_no_weight_not_detected(self):
        mod = MagicMock(spec=[])
        assert _is_ternary_linear(mod) is False

    def test_plain_linear_not_detected(self):
        mod = MagicMock()
        mod.weight = torch.randn(10, 20)
        del mod.weight_scale
        mod.online_quant = False
        type(mod).__name__ = "Linear"
        assert _is_ternary_linear(mod) is False


class TestDetectWeightScaleMode:
    def test_autobitlinear_is_multiply(self):
        mod = MagicMock()
        type(mod).__name__ = "AutoBitLinear"
        assert _detect_weight_scale_mode(mod) == "multiply"

    def test_bitlinear_is_divide(self):
        mod = MagicMock()
        type(mod).__name__ = "BitLinear"
        assert _detect_weight_scale_mode(mod) == "divide"

    def test_unknown_class_defaults_to_divide(self):
        mod = MagicMock()
        type(mod).__name__ = "CustomTernaryLinear"
        assert _detect_weight_scale_mode(mod) == "divide"


class TestGetTernaryLayers:
    def test_finds_bitlinear_via_weight_scale(self):
        """Detect AutoBitLinear modules (have weight_scale attribute)."""
        model = MagicMock()
        ternary_mod = MagicMock()
        ternary_mod.weight = torch.zeros(10, 20, dtype=torch.float32)
        ternary_mod.weight_scale = torch.tensor(0.5)
        fp_mod = MagicMock()
        fp_mod.weight = torch.zeros(10, 20, dtype=torch.float32)
        del fp_mod.weight_scale  # ensure no weight_scale
        fp_mod.online_quant = False
        type(fp_mod).__name__ = "LayerNorm"
        model.named_modules.return_value = [
            ("attn.q_proj", ternary_mod),
            ("norm", fp_mod),
        ]
        result = get_ternary_layers(model)
        assert "attn.q_proj" in result
        assert "norm" not in result

    def test_finds_uint8_packed_modules(self):
        """Detect modules with packed uint8 weights."""
        model = MagicMock()
        packed_mod = MagicMock()
        packed_mod.weight = torch.zeros(5, 20, dtype=torch.uint8)
        del packed_mod.weight_scale
        model.named_modules.return_value = [("layer.linear", packed_mod)]
        result = get_ternary_layers(model)
        assert "layer.linear" in result

    def test_finds_online_quant_modules(self):
        """Detect online_quant=True modules (bf16 BitNet)."""
        model = MagicMock()
        mod = MagicMock()
        mod.weight = torch.randn(10, 20)
        del mod.weight_scale
        mod.online_quant = True
        type(mod).__name__ = "AutoBitLinear"
        model.named_modules.return_value = [("attn.q_proj", mod)]
        result = get_ternary_layers(model)
        assert "attn.q_proj" in result

    def test_empty_model(self):
        model = MagicMock()
        model.named_modules.return_value = []
        assert len(get_ternary_layers(model)) == 0


class TestGetNonQuantizedParams:
    def test_excludes_packed_weight_only(self):
        model = MagicMock()
        model.named_parameters.return_value = [
            ("attn.q_proj.weight", torch.zeros(10, dtype=torch.uint8)),
            ("attn.q_proj.weight_scale", torch.tensor(0.5)),
            ("norm.weight", torch.ones(20)),
            ("embed.weight", torch.randn(100, 64)),
        ]
        result = get_non_quantized_params(model, {"attn.q_proj"})
        assert "attn.q_proj.weight" not in result
        assert "attn.q_proj.weight_scale" in result
        assert "norm.weight" in result
        assert "embed.weight" in result


# ---------------------------------------------------------------------------
# RSR preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessLayerCpu:
    EXPECTED_KEYS = {"perms", "group_ends", "pos_masks", "neg_masks", "block_meta"}

    @pytest.mark.parametrize(
        "n_rows,n_cols", [(16, 32), (32, 16), (64, 64), (24, 48)]
    )
    def test_output_keys(self, n_rows, n_cols):
        w = torch.randint(-1, 2, (n_rows, n_cols), dtype=torch.int8)
        result = preprocess_layer_cpu(w, k=8)
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_all_tensors_int32(self):
        w = torch.randint(-1, 2, (32, 32), dtype=torch.int8)
        result = preprocess_layer_cpu(w, k=8)
        for key, tensor in result.items():
            assert tensor.dtype == torch.int32, f"{key} dtype = {tensor.dtype}"

    @pytest.mark.parametrize("k", [4, 6, 8])
    def test_different_k(self, k):
        w = torch.randint(-1, 2, (24, 48), dtype=torch.int8)
        result = preprocess_layer_cpu(w, k=k)
        assert all(isinstance(v, torch.Tensor) for v in result.values())

    def test_identity_matrix(self):
        """Preprocessing an identity matrix should succeed without error."""
        n = 32
        w = torch.eye(n, dtype=torch.int8)
        result = preprocess_layer_cpu(w, k=8)
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_all_zeros(self):
        w = torch.zeros(16, 32, dtype=torch.int8)
        result = preprocess_layer_cpu(w, k=8)
        assert set(result.keys()) == self.EXPECTED_KEYS


# ---------------------------------------------------------------------------
# CUDA preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessLayerCuda:
    EXPECTED_KEYS = {"perms", "group_packed", "block_meta"}

    @pytest.mark.parametrize(
        "n_rows,n_cols", [(16, 32), (32, 16), (64, 64), (24, 48)]
    )
    def test_output_keys(self, n_rows, n_cols):
        w = torch.randint(-1, 2, (n_rows, n_cols), dtype=torch.int8)
        result = preprocess_layer_cuda(w, k=8)
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_perms_dtype_uint16(self):
        w = torch.randint(-1, 2, (32, 64), dtype=torch.int8)
        result = preprocess_layer_cuda(w, k=8)
        assert result["perms"].dtype == torch.uint16

    def test_group_packed_dtype_int64(self):
        w = torch.randint(-1, 2, (32, 64), dtype=torch.int8)
        result = preprocess_layer_cuda(w, k=8)
        assert result["group_packed"].dtype == torch.int64

    @pytest.mark.parametrize("k", [4, 6, 8, 12])
    def test_different_k(self, k):
        w = torch.randint(-1, 2, (24, 64), dtype=torch.int8)
        result = preprocess_layer_cuda(w, k=k)
        assert "group_packed" in result

    def test_non_square(self):
        w = torch.randint(-1, 2, (1024, 4096), dtype=torch.int8)
        result = preprocess_layer_cuda(w, k=8)
        assert result["perms"].shape == ((1024 // 8) * 4096,)
        assert result["block_meta"].shape == (2 * (1024 // 8),)

    def test_no_column_padding(self):
        w = torch.randint(-1, 2, (16, 20), dtype=torch.int8)
        result = preprocess_layer_cuda(w, k=8)
        assert result["perms"].shape == ((16 // 8) * 20,)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

class TestSavePreprocessed:
    def _make_dummy(self):
        rsr_data = {
            "layer1": {
                "perms": torch.zeros(10, dtype=torch.int32),
                "group_ends": torch.zeros(5, dtype=torch.int32),
                "pos_masks": torch.zeros(5, dtype=torch.int32),
                "neg_masks": torch.zeros(5, dtype=torch.int32),
                "block_meta": torch.zeros(4, dtype=torch.int32),
            }
        }
        non_quantized = {"embed.weight": torch.randn(100, 64)}
        layer_meta = {"layer1": {"n_rows": 32, "n_cols": 64, "k": 8}}
        return rsr_data, non_quantized, layer_meta

    def test_creates_output_files(self):
        rsr_data, non_quantized, layer_meta = self._make_dummy()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "output"
            save_preprocessed(
                rsr_data, non_quantized, layer_meta,
                model_config={"model_type": "test"},
                output_dir=out, k=8, version="adaptive", model_name="test",
            )
            assert (out / "rsr_weights.safetensors").exists()
            assert (out / "non_quantized_weights.safetensors").exists()
            assert (out / "rsr_config.json").exists()
            assert (out / "config.json").exists()

    def test_rsr_config_contents(self):
        rsr_data, non_quantized, layer_meta = self._make_dummy()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "output"
            save_preprocessed(
                rsr_data, non_quantized, layer_meta,
                model_config={}, output_dir=out,
                k=8, version="adaptive", model_name="my-model",
            )
            cfg = json.loads((out / "rsr_config.json").read_text())
            assert cfg["k"] == 8
            assert cfg["version"] == "adaptive"
            assert cfg["model_name"] == "my-model"
            assert "layer1" in cfg["layers"]

    def test_safetensors_roundtrip(self):
        from safetensors.torch import load_file

        rsr_data, non_quantized, layer_meta = self._make_dummy()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "output"
            save_preprocessed(
                rsr_data, non_quantized, layer_meta,
                model_config={}, output_dir=out,
                k=8, version="adaptive", model_name="test",
            )
            loaded = load_file(str(out / "rsr_weights.safetensors"))
            for key in ["perms", "group_ends", "pos_masks", "neg_masks", "block_meta"]:
                full_key = f"layer1.{key}"
                assert full_key in loaded
                torch.testing.assert_close(loaded[full_key], rsr_data["layer1"][key])


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_required_args(self):
        args = parse_args(["--model", "test-model", "--output", "/tmp/out"])
        assert args.model == "test-model"
        assert args.output == "/tmp/out"
        assert args.k == 8
        assert args.version == "adaptive"
        assert args.device == "cpu"

    def test_custom_k(self):
        args = parse_args(["--model", "m", "--output", "o", "--k", "4"])
        assert args.k == 4

    def test_cuda_device(self):
        args = parse_args(["--model", "m", "--output", "o", "--device", "cuda"])
        assert args.device == "cuda"

    def test_short_flags(self):
        args = parse_args(["-m", "model-name", "-o", "out-dir"])
        assert args.model == "model-name"
        assert args.output == "out-dir"

    def test_trust_remote_code(self):
        args = parse_args(["-m", "m", "-o", "o", "--trust-remote-code"])
        assert args.trust_remote_code is True

    def test_missing_required_exits(self):
        with pytest.raises(SystemExit):
            parse_args([])


# ---------------------------------------------------------------------------
# End-to-end (synthetic data, no HF model)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_unpack_preprocess_save_roundtrip(self):
        """Full pipeline: pack -> unpack -> preprocess -> save -> load."""
        from safetensors.torch import load_file

        n_rows, n_cols, k = 32, 48, 8
        weights = torch.randint(-1, 2, (n_rows, n_cols), dtype=torch.int8)
        packed = pack_ternary_weights(weights)

        unpacked = unpack_ternary_weights(packed)
        torch.testing.assert_close(unpacked, weights)

        result = preprocess_layer_cpu(unpacked, k=k)
        assert len(result) == 5

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "model"
            rsr_data = {"test_layer": result}
            layer_meta = {"test_layer": {"n_rows": n_rows, "n_cols": n_cols, "k": k}}
            save_preprocessed(
                rsr_data, {}, layer_meta,
                model_config={}, output_dir=out,
                k=k, version="adaptive", model_name="test",
            )

            loaded = load_file(str(out / "rsr_weights.safetensors"))
            for key in result:
                torch.testing.assert_close(loaded[f"test_layer.{key}"], result[key])

    def test_multiple_layers(self):
        """Preprocess multiple layers of different shapes and save together."""
        from safetensors.torch import load_file

        k = 8
        shapes = [(32, 48), (64, 32), (16, 16)]
        rsr_data = {}
        layer_meta = {}

        for i, (nr, nc) in enumerate(shapes):
            w = torch.randint(-1, 2, (nr, nc), dtype=torch.int8)
            arrays = preprocess_layer_cpu(w, k=k)
            name = f"layers.{i}.linear"
            rsr_data[name] = arrays
            layer_meta[name] = {"n_rows": nr, "n_cols": nc, "k": k}

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "model"
            save_preprocessed(
                rsr_data, {}, layer_meta,
                model_config={}, output_dir=out,
                k=k, version="adaptive", model_name="test",
            )

            loaded = load_file(str(out / "rsr_weights.safetensors"))
            for name, arrays in rsr_data.items():
                for key, tensor in arrays.items():
                    torch.testing.assert_close(loaded[f"{name}.{key}"], tensor)

            cfg = json.loads((out / "rsr_config.json").read_text())
            assert len(cfg["layers"]) == 3
