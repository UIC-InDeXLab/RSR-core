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
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    @contextmanager
    def no_init_weights():
        yield

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from multiplier.bit_1_58.cpu._rsr_v3_common import bitnet_act_quant  # noqa: E402
from multiplier.bit_1_58.cpu.rsr_runtime import (  # noqa: E402
    RSRBatchContext,
    RSRBatchMultiplier,
    RSRBatchMultiplierV31,
    RSRPreprocessedMultiplier,
    select_cpu_tensor_keys,
    uses_v33,
)
from multiplier.bit_1_58.cuda.rsr_runtime import (  # noqa: E402
    CUDA_TENSOR_KEYS,
    RSRPreprocessedCudaMultiplier,
)


class GreenTextStreamer(TextStreamer):
    """TextStreamer that prints generated tokens in ANSI green."""

    _GREEN = "\033[32m"
    _RESET = "\033[0m"

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        print(
            f"{self._GREEN}{text}{self._RESET}",
            flush=True,
            end="" if not stream_end else "\n",
        )


def _bitnet_act_quant(activation: torch.Tensor) -> torch.Tensor:
    """Compatibility wrapper around the shared BitNet activation quantizer."""
    return bitnet_act_quant(activation)


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


class RSRLinear(nn.Module):
    """BitNet-aware replacement for a ternary linear layer.

    Supports three CPU execution modes (selected automatically):

    * **batch** — when a :class:`RSRBatchContext` is attached, the batch
      kernel runs act_quant once and computes all grouped layers in a
      single C call.
    * **fused** — single-layer fused act_quant + GEMV (one C call instead
      of Python act_quant + C GEMV).
    * **legacy** — original path (Python act_quant then C GEMV).  Used only
      when the fused/batch kernel is unavailable.

    CUDA layers always use the existing CUDA kernel (unchanged).
    """

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
            self.multiplier = RSRPreprocessedCudaMultiplier(
                layer_name, layer_meta, tensors
            )
        else:
            self.multiplier = RSRPreprocessedMultiplier(
                layer_name, layer_meta, tensors
            )
        self.rms_norm = copy.deepcopy(rms_norm)

        # Batch context (set later by _replace_ternary_layers if this layer
        # belongs to a group that shares the same input).
        self._batch_ctx: RSRBatchContext | None = None

        # Scratch buffer for fused single-layer path (allocated lazily).
        self._v_scratch: np.ndarray | None = None

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

    def _ensure_scratch(self) -> np.ndarray:
        if self._v_scratch is None:
            self._v_scratch = np.empty(self.in_features, dtype=np.float32)
        return self._v_scratch

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Layer {self.layer_name!r} expected hidden size {self.in_features}, "
                f"got {inputs.shape[-1]}"
            )

        if self.rms_norm is not None:
            inputs = self.rms_norm(inputs)

        input_device = inputs.device
        input_dtype = inputs.dtype

        if self._cuda_backend:
            # CUDA path: run the kernel in float32, then restore the caller's
            # activation dtype so surrounding bf16 modules keep matching dtypes.
            flat_inputs = inputs.float().reshape(-1, self.in_features)
            out_device = self.multiplier.device
            flat_outputs = torch.empty(
                (flat_inputs.shape[0], self.out_features),
                dtype=torch.float32,
                device=out_device,
            )
            for idx in range(flat_inputs.shape[0]):
                flat_outputs[idx] = self.multiplier(flat_inputs[idx])

        elif self._batch_ctx is not None and inputs.shape[:-1].numel() == 1:
            # Batch path: act_quant is fused inside the C batch kernel.
            # Only used for single-token (autoregressive decode).  Multi-token
            # inputs (prefill) fall through to the fused single-layer path
            # because RSRBatchContext counts calls across layers — feeding it
            # multiple tokens from the same layer corrupts its cache counter.
            flat_inputs = inputs.float().reshape(-1, self.in_features)
            flat_outputs = torch.empty(
                (flat_inputs.shape[0], self.out_features),
                dtype=torch.float32,
            )
            for idx in range(flat_inputs.shape[0]):
                flat_outputs[idx] = self._batch_ctx.get_output(
                    self.layer_name, flat_inputs[idx],
                )
            out_device = torch.device("cpu")

        else:
            # Fused single-layer path: act_quant + GEMV in one C call (v3.3 and v3.1).
            flat_inputs = inputs.float().reshape(-1, self.in_features)
            scratch = self._ensure_scratch()
            flat_outputs = torch.empty(
                (flat_inputs.shape[0], self.out_features),
                dtype=torch.float32,
            )
            for idx in range(flat_inputs.shape[0]):
                flat_outputs[idx] = self.multiplier.fused_call(
                    flat_inputs[idx], scratch,
                )
            out_device = torch.device("cpu")

        output = flat_outputs.reshape(*inputs.shape[:-1], self.out_features)

        if self.bias is not None:
            output = output + self.bias.to(device=output.device, dtype=output.dtype)

        if self.weight_scale is not None:
            ws = self.weight_scale.to(device=output.device, dtype=output.dtype)
            if self._weight_scale_mode == "divide":
                output = output / ws
            else:
                output = output * ws

        if output.device != input_device or output.dtype != input_dtype:
            output = output.to(device=input_device, dtype=input_dtype)
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
    layer_meta: dict[str, Any],
    backend: str = "cpu",
) -> dict[str, torch.Tensor]:
    prefix = f"{layer_name}."
    if backend == "cuda":
        expected_keys = CUDA_TENSOR_KEYS
    else:
        n_cols = int(layer_meta["n_cols"])
        k = int(layer_meta["k"])
        expected_keys = select_cpu_tensor_keys(n_cols, k)
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


def _group_layers_for_batching(
    layer_meta: dict[str, Any],
) -> tuple[dict[str, list[str]], set[str]]:
    """Identify CPU layers that share the same input and can be batched.

    Returns ``(groups, grouped_names)`` where *groups* maps a group key to an
    ordered list of layer names and *grouped_names* is the flat set of all
    layer names that belong to some group.

    Grouping rules (transformer-specific):
      * ``self_attn.{q,k,v}_proj`` share *hidden_states*.
      * ``mlp.{gate,up}_proj``     share the post-attention hidden state.
      * Only CPU-backend layers are grouped (CUDA has its own pipeline).
    """
    _QKV = {"q_proj", "k_proj", "v_proj"}
    _GATE_UP = {"gate_proj", "up_proj"}

    groups: dict[str, list[str]] = {}
    grouped_names: set[str] = set()

    for layer_name, meta in layer_meta.items():
        if meta.get("backend", "cpu") != "cpu":
            continue
        parts = layer_name.rsplit(".", 1)
        if len(parts) != 2:
            continue
        prefix, suffix = parts
        if suffix in _QKV:
            key = prefix + "._batch_qkv"
        elif suffix in _GATE_UP:
            key = prefix + "._batch_gate_up"
        else:
            continue
        groups.setdefault(key, []).append(layer_name)
        grouped_names.add(layer_name)

    # Drop incomplete groups (need >= 2 layers to benefit from batching).
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    grouped_names = {n for names in groups.values() for n in names}
    return groups, grouped_names


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

    groups, grouped_names = _group_layers_for_batching(layer_meta)

    # First pass: create all RSRLinear replacements.
    replacements: dict[str, RSRLinear] = {}

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
                layer_meta=meta,
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
            replacements[layer_name] = replacement

    # Second pass: wire up batch contexts for grouped CPU layers.
    for _group_key, names in groups.items():
        mults = [replacements[n].multiplier for n in names]
        # Pick v3.3 or v3.1 batch multiplier based on first layer's kernel.
        if mults[0]._use_v33:
            batch_mult = RSRBatchMultiplier(mults)
        else:
            batch_mult = RSRBatchMultiplierV31(mults)
        ctx = RSRBatchContext(batch_mult, names)
        for name in names:
            replacements[name]._batch_ctx = ctx

    # Install all replacements into the model.
    for layer_name, replacement in replacements.items():
        _set_module(model, layer_name, replacement)


def _materialize_meta_buffers(model: nn.Module) -> None:
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    if rotary is not None and hasattr(rotary, "inv_freq") and rotary.inv_freq.is_meta:
        if hasattr(rotary, "rope_init_fn"):
            inv_freq, attention_scaling = rotary.rope_init_fn(rotary.config, device="cpu")
        elif hasattr(rotary, "compute_default_rope_parameters"):
            inv_freq, attention_scaling = rotary.compute_default_rope_parameters(rotary.config, device="cpu")
        else:
            return
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

    # The @torch.compile-decorated unpack_weights in transformers' BitNet
    # integration fails on CPU with dynamo.  Force eager execution.
    _prev_suppress = torch._dynamo.config.suppress_errors
    torch._dynamo.config.suppress_errors = True

    model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)

    torch._dynamo.config.suppress_errors = _prev_suppress

    # Work around a bug in transformers' BitNetDeserialize.convert: it unpacks
    # ternary weights with dtype=uint8 (the storage dtype) instead of the
    # model's compute dtype, so -1 wraps to 255 and F.linear gets a dtype
    # mismatch.  Only apply to BitNet models (detected via quantization_config).
    # Fix by reinterpreting uint8 as int8 then casting to the model's dtype.
    _is_bitnet = getattr(model.config, "quantization_config", None) is not None and (
        getattr(model.config.quantization_config, "quant_method", None) == "bitnet"
        or (isinstance(model.config.quantization_config, dict)
            and model.config.quantization_config.get("quant_method") == "bitnet")
    )
    if _is_bitnet:
        # Determine the correct target dtype: use the explicitly requested dtype,
        # otherwise infer from the non-quantized parameters already in the model.
        if dtype:
            _target_dtype = getattr(torch, dtype)
        else:
            _non_uint8 = [
                p.dtype for p in model.parameters() if p.dtype != torch.uint8
            ]
            _target_dtype = _non_uint8[0] if _non_uint8 else torch.bfloat16
        for _name, param in model.named_parameters():
            if param.dtype == torch.uint8:
                param.data = param.data.view(torch.int8).to(_target_dtype)

    # bitsandbytes models are already placed by device_map; skip .to()
    if quantize not in ("8bit", "4bit"):
        model = model.to(device)
    model.eval()

    tokenizer_source = tokenizer_name_or_path or model_source
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


_BOLD_CYAN = "\033[1;36m"
_RESET      = "\033[0m"


def _print_inference_stats(n_tokens: int, elapsed: float) -> None:
    """Print a bold-cyan summary table with token count, wall time, and throughput."""
    tok_per_sec = n_tokens / elapsed if elapsed > 0 else float("inf")
    rows = [
        ("tokens", str(n_tokens)),
        ("time",   f"{elapsed:.3f} s"),
        ("tok/s",  f"{tok_per_sec:.1f}"),
    ]
    w_label = max(len(r[0]) for r in rows) + 2
    w_value = max(len(r[1]) for r in rows) + 2
    top = f"┌{'─' * w_label}┬{'─' * w_value}┐"
    mid = f"├{'─' * w_label}┼{'─' * w_value}┤"
    bot = f"└{'─' * w_label}┴{'─' * w_value}┘"

    def _line(s: str) -> None:
        print(f"{_BOLD_CYAN}{s}{_RESET}")

    _line(top)
    for i, (label, value) in enumerate(rows):
        _line(f"│ {label:<{w_label - 2}} │ {value:>{w_value - 2}} │")
        if i < len(rows) - 1:
            _line(mid)
    _line(bot)


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
        streamer = GreenTextStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

    if stream:
        print(f"{_BOLD_CYAN}▶ response{_RESET}")

    t0 = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
        **generate_kwargs,
    )  # type: ignore
    elapsed = time.perf_counter() - t0

    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0, prompt_length:]
    if stream:
        print()
    _print_inference_stats(len(generated_ids), elapsed)
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
