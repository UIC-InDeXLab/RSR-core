"""
Run a Hugging Face causal LM on top of RSR-preprocessed ternary weights.

This keeps the user-facing flow close to normal transformers usage:

    model, tokenizer = load_preprocessed_model("./integrations/hf")
    inputs = tokenizer("Hello", return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=32)

The main difference is that ternary linear layers are replaced with an
RSR-backed runtime module initialized from the artifacts written by
``integrations.hf.model_prep``.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers.modeling_utils import no_init_weights
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from multiplier.bit_1_58.cpu._rsr_v3_common import (  # noqa: E402
    INT32_PTR,
    UINT16_PTR,
    ensure_cpu_float32_contiguous,
    tensor_float_ptr,
)

_KERNEL_DIR = _PROJECT_ROOT / "kernels" / "bit_1_58" / "cpu"
_RSR_V33_LIB = None
_RSR_TENSOR_KEYS = ("perms", "group_ends", "pos_masks", "neg_masks", "block_meta")
_RSR_CUDA_TENSOR_KEYS = ("perms", "group_packed", "block_meta")
_RSR_CUDA_MODULE = None


def _load_rsr_v33_lib():
    global _RSR_V33_LIB
    if _RSR_V33_LIB is None:
        lib = ctypes.CDLL(str(_KERNEL_DIR / "rsr_ternary_v3_3.so"))
        lib.rsr_ternary_gemv_v3_3.restype = None
        lib.rsr_ternary_gemv_v3_3.argtypes = [
            UINT16_PTR,
            UINT16_PTR,
            UINT16_PTR,
            UINT16_PTR,
            INT32_PTR,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        _RSR_V33_LIB = lib
    return _RSR_V33_LIB


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _strip_auto_map(config_dict: dict[str, Any]) -> dict[str, Any]:
    clean = dict(config_dict)
    clean.pop("auto_map", None)
    return clean


def _load_local_config(model_dir: Path):
    config_dict = _strip_auto_map(_read_json(model_dir / "config.json"))
    model_type = config_dict.get("model_type")
    if model_type not in CONFIG_MAPPING:
        raise ValueError(
            f"Unsupported local model_type={model_type!r}. "
            "Pass a base model that transformers can load directly."
        )
    return CONFIG_MAPPING[model_type].from_dict(config_dict)


def _resolve_module(root: nn.Module, dotted_name: str) -> nn.Module:
    module = root
    for part in dotted_name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _set_module(root: nn.Module, dotted_name: str, module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = module
    else:
        setattr(parent, last, module)


class _PreprocessedRSRMultiplier:
    """GEMV runtime built directly from saved RSR tensors."""

    def __init__(
        self,
        layer_name: str,
        layer_meta: dict[str, Any],
        tensors: dict[str, torch.Tensor],
    ):
        self.layer_name = layer_name
        self.n_rows = int(layer_meta["n_rows"])
        self.n_cols = int(layer_meta["n_cols"])
        self.k = int(layer_meta["k"])

        self._perms = self._to_uint16(tensors["perms"], "perms")
        self._group_ends = self._to_uint16(tensors["group_ends"], "group_ends")
        self._pos_masks = self._to_uint16(tensors["pos_masks"], "pos_masks")
        self._neg_masks = self._to_uint16(tensors["neg_masks"], "neg_masks")
        self._block_meta = self._to_int32(tensors["block_meta"], "block_meta")

        if self._block_meta.size % 2 != 0:
            raise ValueError(
                f"Layer {layer_name!r} has invalid block_meta length {self._block_meta.size}"
            )

        self._num_blocks = self._block_meta.size // 2
        self._perms_ptr = self._perms.ctypes.data_as(UINT16_PTR)
        self._group_ends_ptr = self._group_ends.ctypes.data_as(UINT16_PTR)
        self._pos_masks_ptr = self._pos_masks.ctypes.data_as(UINT16_PTR)
        self._neg_masks_ptr = self._neg_masks.ctypes.data_as(UINT16_PTR)
        self._block_meta_ptr = self._block_meta.ctypes.data_as(INT32_PTR)

    @staticmethod
    def _to_uint16(tensor: torch.Tensor, name: str) -> np.ndarray:
        cpu = tensor.detach().to(device="cpu", dtype=torch.int32).contiguous()
        if cpu.numel() and (
            cpu.min().item() < 0 or cpu.max().item() > np.iinfo(np.uint16).max
        ):
            raise ValueError(f"Tensor {name!r} contains values outside uint16 range")
        return cpu.numpy().astype(np.uint16, copy=True)

    @staticmethod
    def _to_int32(tensor: torch.Tensor, name: str) -> np.ndarray:
        cpu = tensor.detach().to(device="cpu", dtype=torch.int32).contiguous()
        if not cpu.is_contiguous():
            raise ValueError(f"Tensor {name!r} must be contiguous")
        return cpu.numpy().copy()

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        if vector.ndim != 1 or vector.shape[0] != self.n_cols:
            raise ValueError(
                f"Layer {self.layer_name!r} expected vector of shape ({self.n_cols},), "
                f"got {tuple(vector.shape)}"
            )

        v_cpu = ensure_cpu_float32_contiguous(vector)
        out_cpu = torch.empty(self.n_rows, dtype=torch.float32)
        _load_rsr_v33_lib().rsr_ternary_gemv_v3_3(
            self._perms_ptr,
            self._group_ends_ptr,
            self._pos_masks_ptr,
            self._neg_masks_ptr,
            self._block_meta_ptr,
            tensor_float_ptr(v_cpu),
            tensor_float_ptr(out_cpu),
            self.n_cols,
            self.k,
            self._num_blocks,
        )
        return out_cpu


def _load_rsr_cuda_module():
    global _RSR_CUDA_MODULE
    if _RSR_CUDA_MODULE is None:
        from multiplier.bit_1_58.cuda._jit_build import load_kernel

        _RSR_CUDA_MODULE = load_kernel("rsr_ternary_cuda_v2_0", "rsr_ternary_v2_0.cu")
    return _RSR_CUDA_MODULE


class _PreprocessedRSRCudaMultiplier:
    """CUDA GEMV runtime built from saved v2.0 compact metadata."""

    def __init__(
        self,
        layer_name: str,
        layer_meta: dict[str, Any],
        tensors: dict[str, torch.Tensor],
    ):
        self.layer_name = layer_name
        self.n_rows = int(layer_meta["n_rows"])
        self.n_cols = int(layer_meta["n_cols"])
        self.k = int(layer_meta["k"])
        self.n_rows_padded = int(
            layer_meta.get(
                "n_rows_padded",
                ((self.n_rows + self.k - 1) // self.k) * self.k,
            )
        )
        self._num_blocks = int(
            layer_meta.get("num_blocks", self.n_rows_padded // self.k)
        )
        self.device = torch.device("cuda")

        self._perms = tensors["perms"].to(dtype=torch.uint16, device=self.device)
        self._group_packed = tensors["group_packed"].to(
            dtype=torch.int64,
            device=self.device,
        )
        self._block_meta = tensors["block_meta"].to(
            dtype=torch.int32,
            device=self.device,
        )
        self._out = torch.empty(
            self.n_rows_padded, dtype=torch.float32, device=self.device
        )

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        if vector.ndim != 1 or vector.shape[0] != self.n_cols:
            raise ValueError(
                f"Layer {self.layer_name!r} expected vector of shape ({self.n_cols},), "
                f"got {tuple(vector.shape)}"
            )

        if vector.device != self.device or vector.dtype != torch.float32:
            v_gpu = vector.to(self.device, dtype=torch.float32)
        else:
            v_gpu = vector

        _load_rsr_cuda_module().rsr_ternary_gemv_v2_0(
            self._perms,
            self._group_packed,
            self._block_meta,
            v_gpu.contiguous(),
            self._out,
            self.n_cols,
            self.k,
            self._num_blocks,
        )
        return self._out[: self.n_rows]


def _bitnet_act_quant(activation: torch.Tensor) -> torch.Tensor:
    """Match transformers.integrations.bitnet.ActQuant for offline BitNet."""
    dtype = activation.dtype
    activation = activation.float()
    scale = 127 / activation.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    activation = (activation * scale).round().clamp(-128, 127) / scale
    return activation.to(dtype)


class RSRLinear(nn.Module):
    """BitNet-aware replacement for a ternary linear layer."""

    def __init__(
        self,
        layer_name: str,
        layer_meta: dict[str, Any],
        tensors: dict[str, torch.Tensor],
        *,
        bias: torch.Tensor | None = None,
        weight_scale: torch.Tensor | None = None,
        rms_norm: nn.Module | None = None,
    ):
        super().__init__()
        self.layer_name = layer_name
        self.in_features = int(layer_meta["n_cols"])
        self.out_features = int(layer_meta["n_rows"])
        backend = layer_meta.get("backend", "cpu")
        self._cuda_backend = backend == "cuda"
        self._weight_scale_mode = layer_meta.get("weight_scale_mode", "multiply")
        if self._cuda_backend:
            self.multiplier = _PreprocessedRSRCudaMultiplier(
                layer_name, layer_meta, tensors
            )
        else:
            self.multiplier = _PreprocessedRSRMultiplier(
                layer_name, layer_meta, tensors
            )
        self.rms_norm = copy.deepcopy(rms_norm)

        if bias is None:
            self.register_buffer("bias", None, persistent=False)
        else:
            self.register_buffer("bias", bias.detach().clone(), persistent=True)

        if weight_scale is None:
            self.register_buffer("weight_scale", None, persistent=False)
        else:
            self.register_buffer(
                "weight_scale",
                weight_scale.detach().clone(),
                persistent=True,
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Layer {self.layer_name!r} expected hidden size {self.in_features}, "
                f"got {inputs.shape[-1]}"
            )

        if self.rms_norm is not None:
            inputs = self.rms_norm(inputs)

        # The CUDA kernel has its own int8 quantization fused in, so applying
        # _bitnet_act_quant first would double-quantize (quantize → dequantize
        # → re-quantize), introducing rounding errors that compound across
        # hundreds of layers.  The CPU kernel works with float32 directly, so
        # it needs the fake-quantized (dequantized) values from act_quant.
        if not self._cuda_backend:
            inputs = _bitnet_act_quant(inputs)
        else:
            inputs = inputs.float()
        flat_inputs = inputs.reshape(-1, self.in_features)
        out_device = getattr(self.multiplier, "device", torch.device("cpu"))
        flat_outputs = torch.empty(
            (flat_inputs.shape[0], self.out_features),
            dtype=torch.float32,
            device=out_device,
        )
        for idx in range(flat_inputs.shape[0]):
            flat_outputs[idx] = self.multiplier(flat_inputs[idx])

        output = flat_outputs.reshape(*inputs.shape[:-1], self.out_features)

        if self.bias is not None:
            output = output + self.bias.to(device=output.device, dtype=output.dtype)

        if self.weight_scale is not None:
            ws = self.weight_scale.to(device=output.device, dtype=output.dtype)
            if self._weight_scale_mode == "divide":
                output = output / ws
            else:
                output = output * ws

        if inputs.device.type != "cpu" or inputs.dtype != output.dtype:
            output = output.to(device=inputs.device, dtype=inputs.dtype)
        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


def _load_non_quantized_state(model_dir: Path) -> dict[str, torch.Tensor]:
    path = model_dir / "non_quantized_weights.safetensors"
    if not path.exists():
        return {}
    return safetensors_load(str(path))


def _load_layer_tensors(
    handle,
    layer_name: str,
    backend: str = "cpu",
) -> dict[str, torch.Tensor]:
    prefix = f"{layer_name}."
    expected_keys = _RSR_CUDA_TENSOR_KEYS if backend == "cuda" else _RSR_TENSOR_KEYS
    tensors = {}
    keys = set(handle.keys())
    if backend == "cuda" and prefix + "packed" in keys:
        raise KeyError(
            f"Found legacy CUDA tensor {prefix + 'packed'!r}. "
            "Re-run integrations/hf/model_prep.py to regenerate CUDA "
            "artifacts for the retained v2.0 kernel."
        )
    for key in expected_keys:
        tensor_key = prefix + key
        if tensor_key not in keys:
            raise KeyError(f"Missing {tensor_key!r} in RSR safetensors")
        tensors[key] = handle.get_tensor(tensor_key)
    return tensors


def _build_base_model(model_dir: Path) -> nn.Module:
    config = _load_local_config(model_dir)
    with no_init_weights():
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
    return model


def _replace_ternary_layers(
    model: nn.Module,
    model_dir: Path,
    layer_meta: dict[str, Any],
    *,
    aux_state: dict[str, torch.Tensor] | None = None,
    require_weight_scale: bool = False,
) -> None:
    aux_state = aux_state or {}
    rsr_path = model_dir / "rsr_weights.safetensors"
    with safe_open(rsr_path, framework="pt", device="cpu") as handle:
        for layer_name, meta in layer_meta.items():
            original = _resolve_module(model, layer_name)
            layer_prefix = f"{layer_name}."
            layer_aux = {
                key[len(layer_prefix) :]: tensor
                for key, tensor in aux_state.items()
                if key.startswith(layer_prefix)
            }
            if any(key.startswith("rms_norm.") for key in layer_aux):
                raise ValueError(
                    f"Unsupported auxiliary state for {layer_name!r}: "
                    "ternary-layer rms_norm is not yet reconstructable from "
                    "local preprocessed artifacts."
                )
            if require_weight_scale and "weight_scale" not in layer_aux:
                raise ValueError(
                    f"Preprocessed artifacts are missing {layer_name!r}.weight_scale. "
                    "Re-run integrations/hf/model_prep.py to regenerate them."
                )
            tensors = _load_layer_tensors(
                handle,
                layer_name,
                backend=meta.get("backend", "cpu"),
            )
            replacement = RSRLinear(
                layer_name,
                meta,
                tensors,
                bias=getattr(original, "bias", None),
                weight_scale=layer_aux.get("weight_scale"),
            )
            replacement.train(original.training)
            _set_module(model, layer_name, replacement)


def _materialize_meta_buffers(model: nn.Module) -> None:
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    if rotary is not None and hasattr(rotary, "inv_freq") and rotary.inv_freq.is_meta:
        inv_freq, attention_scaling = rotary.rope_init_fn(rotary.config, device="cpu")
        rotary.register_buffer("inv_freq", inv_freq, persistent=False)
        rotary.original_inv_freq = rotary.inv_freq
        rotary.attention_scaling = attention_scaling


def load_preprocessed_model(
    model_dir: str | Path,
    *,
    tokenizer_name_or_path: str | None = None,
    device: str = "cpu",
    dtype: str | None = None,
):
    """Load a causal LM backed by RSR-preprocessed ternary linears.

    Args:
        model_dir: Directory produced by ``integrations.hf.model_prep``.
        tokenizer_name_or_path: Optional tokenizer source. Defaults to the
            original model name stored in ``rsr_config.json``.
        device: Target device for non-quantized model state.
        dtype: Optional torch dtype name, e.g. ``bfloat16`` or ``float32``.

    Returns:
        ``(model, tokenizer)``
    """
    model_dir = Path(model_dir)
    rsr_config = _read_json(model_dir / "rsr_config.json")
    config_dict = _read_json(model_dir / "config.json")

    model = _build_base_model(model_dir)
    non_quantized = _load_non_quantized_state(model_dir)
    model_state_keys = set(model.state_dict().keys())
    base_state = {
        key: tensor for key, tensor in non_quantized.items() if key in model_state_keys
    }
    aux_state = {
        key: tensor
        for key, tensor in non_quantized.items()
        if key not in model_state_keys
    }

    state_result = model.load_state_dict(base_state, strict=False, assign=True)
    if state_result.unexpected_keys:
        raise ValueError(
            f"Unexpected non-quantized keys: {state_result.unexpected_keys[:10]}"
        )

    model.tie_weights()
    _replace_ternary_layers(
        model,
        model_dir,
        rsr_config["layers"],
        aux_state=aux_state,
        require_weight_scale=(
            config_dict.get("quantization_config", {}).get("quant_method") == "bitnet"
        ),
    )
    _materialize_meta_buffers(model)

    unresolved_meta = [
        name for name, param in model.named_parameters() if param.is_meta
    ]
    if unresolved_meta:
        raise ValueError(
            "Model still contains meta parameters after RSR layer replacement: "
            f"{unresolved_meta[:10]}"
        )
    unresolved_meta_buffers = [
        name for name, buf in model.named_buffers() if buf.is_meta
    ]
    if unresolved_meta_buffers:
        raise ValueError(
            "Model still contains meta buffers after RSR layer replacement: "
            f"{unresolved_meta_buffers[:10]}"
        )

    if dtype:
        torch_dtype = getattr(torch, dtype)
        model = model.to(dtype=torch_dtype)
    model = model.to(device)
    model.eval()

    tokenizer_source = tokenizer_name_or_path or rsr_config.get("model_name")
    if tokenizer_source is None:
        raise ValueError("No tokenizer source found. Pass --tokenizer explicitly.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Keep the interface aligned with normal transformers models.
    model.config._name_or_path = rsr_config.get("model_name", str(model_dir))
    return model, tokenizer


def load_hf_model(
    model_dir: str | Path,
    *,
    hf_model_name_or_path: str | None = None,
    tokenizer_name_or_path: str | None = None,
    device: str = "cpu",
    dtype: str | None = None,
    quantize: str | None = None,
):
    """Load the reference Hugging Face model without RSR replacement.

    Args:
        model_dir: Directory containing rsr_config.json (used to resolve
            default model name and tokenizer).
        hf_model_name_or_path: Override HF model ID or local path.
        tokenizer_name_or_path: Override tokenizer source.
        device: Target device (ignored when *quantize* is set — bitsandbytes
            handles placement via ``device_map``).
        dtype: Torch dtype name, e.g. ``"float32"``, ``"float16"``,
            ``"bfloat16"``.
        quantize: Quantization mode for bitsandbytes:
            ``"8bit"`` — load_in_8bit,
            ``"4bit"`` — load_in_4bit (NF4, bfloat16 compute dtype).
    """
    model_dir = Path(model_dir)
    rsr_config = _read_json(model_dir / "rsr_config.json")
    model_source = (
        hf_model_name_or_path or rsr_config.get("model_name") or str(model_dir)
    )

    load_kwargs: dict[str, Any] = {}

    if quantize == "8bit":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = "auto"
    elif quantize == "4bit":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs["device_map"] = "auto"
    else:
        torch_dtype = getattr(torch, dtype) if dtype else None
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)

    # bitsandbytes models are already placed by device_map; skip .to()
    if quantize not in ("8bit", "4bit"):
        model = model.to(device)
    model.eval()

    tokenizer_source = tokenizer_name_or_path or model_source
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@torch.inference_mode()
def generate_text(
    model: nn.Module,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    use_chat_template: bool = True,
    stream: bool = False,
    **generate_kwargs,
) -> str:
    """Tokenize a prompt, call ``model.generate()``, and decode the new text."""
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        inputs = tokenizer(prompt, return_tensors="pt")

    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    streamer = None
    if stream:
        streamer = TextStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
        **generate_kwargs,
    )  # type: ignore

    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0, prompt_length:]
    if stream:
        print()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a Hugging Face causal LM using RSR-preprocessed ternary weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default="integrations/hf",
        help="Directory containing rsr_config.json and the safetensors artifacts.",
    )
    parser.add_argument(
        "--backend",
        default="rsr",
        choices=["rsr", "hf"],
        help="Run the RSR-backed model or the standard Hugging Face reference model.",
    )
    parser.add_argument(
        "--hf-model",
        default=None,
        help="Optional Hugging Face model ID or local path for --backend hf.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer source. Defaults to rsr_config.json:model_name.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Target device. Auto-detected from model-dir name suffix (_cpu/_cuda) if omitted.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help="Optional dtype cast after model construction.",
    )
    parser.add_argument(
        "--quantize",
        default=None,
        choices=["8bit", "4bit"],
        help="Quantization mode for --backend hf (requires bitsandbytes).",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt to generate from.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Skip tokenizer.apply_chat_template and tokenize the raw prompt directly.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream decoded text to stdout as tokens are generated.",
    )
    return parser.parse_args(argv)


def _detect_device_from_dir(model_dir: str) -> str:
    """Infer device from model directory name suffix (_cpu / _cuda)."""
    name = Path(model_dir).name
    if name.endswith("_cuda"):
        return "cuda"
    if name.endswith("_cpu"):
        return "cpu"
    return "cpu"


def main(argv=None):
    args = parse_args(argv)

    if args.device is None:
        args.device = _detect_device_from_dir(args.model_dir)
        import warnings

        warnings.warn(
            f"--device not specified; auto-detected '{args.device}' "
            f"from model directory name '{Path(args.model_dir).name}'"
        )

    if args.backend == "rsr":
        model, tokenizer = load_preprocessed_model(
            args.model_dir,
            tokenizer_name_or_path=args.tokenizer,
            device=args.device,
            dtype=args.dtype,
        )
    else:
        model, tokenizer = load_hf_model(
            args.model_dir,
            hf_model_name_or_path=args.hf_model,
            tokenizer_name_or_path=args.tokenizer,
            device=args.device,
            dtype=args.dtype,
            quantize=args.quantize,
        )
    text = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        use_chat_template=not args.no_chat_template,
        stream=args.stream,
    )
    if not args.stream:
        print(text)


if __name__ == "__main__":
    main()
