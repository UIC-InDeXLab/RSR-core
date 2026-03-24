"""
Preprocesses HuggingFace quantized (ternary) models for RSR inference.

Workflow:
  1. Load a quantized model from HuggingFace (e.g., BitNet b1.58)
  2. Identify ternary (BitLinear) layers whose weights are in {-1, 0, +1}
  3. Apply RSR preprocessing to each ternary weight matrix
  4. Save preprocessed data as safetensors for future inference

Supported model formats:
  - Auto-dequantized: HF transformers loads BitNet weights as float32 {-1,0,+1}
    with AutoBitLinear modules (detected via weight_scale attribute).
  - Packed uint8: Some models store 4 ternary values per byte.
    Use unpack_ternary_weights() to convert before preprocessing.

Usage:
    python -m integrations.hf.model_prep \
        --model microsoft/bitnet-b1.58-2B-4T \
        --output ./preprocessed_model \
        --k 8 \
        --version adaptive
"""

import argparse
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path

import torch
from tqdm import tqdm

def _model_slug(model_name_or_path: str) -> str:
    """Derive a filesystem-safe directory name from a model ID or path.

    ``"microsoft/bitnet-b1.58-2B-4T"`` → ``"bitnet-b1.58-2B-4T"``
    ``"/home/user/my_model"``          → ``"my_model"``
    """
    basename = Path(model_name_or_path).name or model_name_or_path
    return re.sub(r"[^\w.\-]", "_", basename)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Weight unpacking
# ---------------------------------------------------------------------------

def unpack_ternary_weights(packed: torch.Tensor) -> torch.Tensor:
    """Unpack HuggingFace BitNet packed uint8 weights to ternary {-1, 0, +1}.

    HF packing format: 4 ternary values per uint8 byte (2 bits each).
    Encoding: code 0 -> -1, code 1 -> 0, code 2 -> +1
    Layout: packed shape (n_rows // 4, n_cols)
      bits [1:0] -> rows 0 .. n_rows/4-1
      bits [3:2] -> rows n_rows/4 .. n_rows/2-1
      bits [5:4] -> rows n_rows/2 .. 3*n_rows/4-1
      bits [7:6] -> rows 3*n_rows/4 .. n_rows-1

    Args:
        packed: uint8 tensor of shape (n_rows // 4, n_cols)

    Returns:
        int8 tensor of shape (n_rows, n_cols) with values in {-1, 0, +1}
    """
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=packed.device)
    # (4, packed_rows, n_cols) — extract all 4 groups in one vectorized pass
    codes = (packed.unsqueeze(0) >> shifts.view(4, 1, 1)) & 0x03
    return (codes.to(torch.int8) - 1).reshape(-1, packed.shape[1])


def pack_ternary_weights(weights: torch.Tensor) -> torch.Tensor:
    """Pack ternary {-1, 0, +1} int8 weights into HF uint8 format.

    Inverse of unpack_ternary_weights. Useful for testing.
    """
    n_rows, n_cols = weights.shape
    assert n_rows % 4 == 0, f"n_rows={n_rows} must be divisible by 4"
    packed_rows = n_rows // 4
    encoded = (weights + 1).to(torch.uint8)  # -1->0, 0->1, 1->2
    packed = torch.zeros(packed_rows, n_cols, dtype=torch.uint8, device=weights.device)
    for i in range(4):
        packed |= encoded[i * packed_rows : (i + 1) * packed_rows] << (i * 2)
    return packed


# ---------------------------------------------------------------------------
# Layer discovery
# ---------------------------------------------------------------------------

def _is_ternary_linear(module) -> bool:
    """Check whether *module* is a quantized ternary linear layer.

    Detection heuristics (any match is sufficient):
      1. Has a ``weight_scale`` attribute (BitNet AutoBitLinear).
      2. Weight dtype is uint8 (packed ternary format).
      3. Class name contains "BitLinear".
      4. Has ``online_quant`` attribute set to True (bf16 BitNet).
    """
    if not hasattr(module, "weight"):
        return False
    if hasattr(module, "weight_scale"):
        return True
    if module.weight.dtype == torch.uint8:
        return True
    cls_name = type(module).__name__
    if "BitLinear" in cls_name:
        return True
    if getattr(module, "online_quant", False):
        return True
    return False


def _detect_weight_scale_mode(module) -> str:
    """Detect how weight_scale is applied in the original model.

    - ``AutoBitLinear`` (native BitNet models): ``output *= weight_scale``
    - ``BitLinear`` (e.g. Llama with bitnet quant): ``output /= (weight_scale * input_scale)``

    Returns "multiply" or "divide".
    """
    cls_name = type(module).__name__
    if cls_name == "AutoBitLinear":
        return "multiply"
    # BitLinear and any other quantized ternary layer use divide semantics
    return "divide"


def get_ternary_layers(model) -> dict:
    """Find all quantized ternary linear layers in a HF model.

    Returns:
        Dict mapping module name -> module.
    """
    ternary = {}
    for name, module in model.named_modules():
        if _is_ternary_linear(module):
            ternary[name] = module
    return ternary


def get_non_quantized_state(model, ternary_layer_names: set) -> dict:
    """Get all persistent state that is NOT a ternary layer weight.

    This includes both parameters and buffers from ``model.state_dict()`` when
    available, so BitNet auxiliary tensors such as ``weight_scale`` are
    preserved alongside non-ternary module weights.

    For lightweight test doubles that only implement ``named_parameters()``,
    fall back to parameters only.
    """
    ternary_weight_keys = {f"{tl}.weight" for tl in ternary_layer_names}
    state = {}

    state_dict = getattr(model, "state_dict", None)
    if callable(state_dict):
        state_items = state_dict()
        if isinstance(state_items, Mapping):
            for name, tensor in state_items.items():
                if name not in ternary_weight_keys:
                    state[name] = tensor.detach().cpu()
            return state

    named_parameters = getattr(model, "named_parameters", None)
    if callable(named_parameters):
        for name, tensor in named_parameters():
            if name not in ternary_weight_keys:
                state[name] = tensor.detach().cpu()
        return state

    raise TypeError("model must define state_dict() or named_parameters()")


def get_non_quantized_params(model, ternary_layer_names: set) -> dict:
    """Backward-compatible alias for older callers/tests."""
    return get_non_quantized_state(model, ternary_layer_names)


# ---------------------------------------------------------------------------
# RSR preprocessing
# ---------------------------------------------------------------------------

def preprocess_layer_cpu(weight: torch.Tensor, k: int) -> dict:
    """Apply CPU RSR preprocessing to a single ternary weight matrix.

    Uses the non-square multiplier which automatically selects v3.3 or v3.1
    based on n_cols (threshold at 4096).

    Args:
        weight: int8 ternary matrix of shape (n_rows, n_cols), values in {-1, 0, +1}
        k: block height for RSR decomposition

    Returns:
        Dict of preprocessed int32 tensors. For v3.3: perms, group_ends,
        pos_masks, neg_masks, block_meta. For v3.1: perms, group_ends,
        scatter_offsets, scatter_rows, scatter_signs, block_meta.
    """
    import numpy as np
    from multiplier.bit_1_58.cpu.rsr_nonsquare import (
        RSRTernaryNonSquareMultiplier,
    )

    M = weight.to(torch.float32).cpu()
    mult = RSRTernaryNonSquareMultiplier(M, k)

    result = {
        "perms": torch.from_numpy(mult._perms_u16.astype(np.int32).copy()),
        "group_ends": torch.from_numpy(mult._group_ends_u16.astype(np.int32).copy()),
        "block_meta": torch.from_numpy(mult._block_meta.copy()),
    }

    if mult._use_v33:
        result["pos_masks"] = torch.from_numpy(
            mult._pos_masks.astype(np.int32).copy()
        )
        result["neg_masks"] = torch.from_numpy(
            mult._neg_masks.astype(np.int32).copy()
        )
    else:
        result["scatter_offsets"] = torch.from_numpy(
            mult._scatter_offsets.copy()
        )
        result["scatter_rows"] = torch.from_numpy(
            mult._scatter_rows.astype(np.int32).copy()
        )
        result["scatter_signs"] = torch.from_numpy(
            mult._scatter_signs.astype(np.int32).copy()
        )

    return result


def preprocess_layer_cuda(weight: torch.Tensor, k: int) -> dict:
    """Apply CUDA-targeted RSR preprocessing for the retained v2.0 kernel.

    This builds the compact metadata consumed by
    ``multiplier.bit_1_58.cuda.rsr_cuda_v2_0``. The work happens on CPU, so a
    CUDA device is not required at preprocessing time.

    Args:
        weight: int8 ternary matrix of shape (n_rows, n_cols), values in {-1, 0, +1}
        k: block height for RSR decomposition (must satisfy 0 < k <= 16)

    Returns:
        Dict with the compact tensors expected by the CUDA v2.0 runtime.
    """
    from multiplier.bit_1_58.cuda._prep_v2_common import prep_compact_u64

    n_rows, n_cols = weight.shape
    n_rows_padded = ((n_rows + k - 1) // k) * k
    perms, group_packed, block_meta, _ = prep_compact_u64(
        weight.to(torch.float32).cpu(),
        n_rows,
        n_cols,
        k,
        n_rows_padded,
        torch.device("cpu"),
    )
    return {
        "perms": perms.contiguous(),
        "group_packed": group_packed.contiguous(),
        "block_meta": block_meta.contiguous(),
    }


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_preprocessed(
    rsr_data: dict,
    non_quantized: dict,
    layer_meta: dict,
    model_config: dict,
    output_dir: Path,
    k: int,
    version: str,
    model_name: str,
):
    """Save preprocessed model to *output_dir*.

    Files written:
        rsr_weights.safetensors          — RSR preprocessed tensors
        non_quantized_weights.safetensors — embeddings, norms, scales, …
        rsr_config.json                  — layer metadata + preprocessing params
        config.json                      — original HF model config
    """
    from safetensors.torch import save_file as safetensors_save

    def _storage_identity(tensor: torch.Tensor) -> tuple[int, int]:
        storage = tensor.untyped_storage()
        return storage.data_ptr(), storage.nbytes()

    def _tensor_layout(tensor: torch.Tensor) -> tuple:
        return (
            tensor.dtype,
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
        )

    def _prepare_safetensors_state(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        prepared: dict[str, torch.Tensor] = {}
        by_storage: dict[tuple[int, int], tuple[str, tuple]] = {}

        for name, tensor in tensors.items():
            storage_id = _storage_identity(tensor)
            layout = _tensor_layout(tensor)
            if storage_id not in by_storage:
                by_storage[storage_id] = (name, layout)
                prepared[name] = tensor
                continue

            kept_name, kept_layout = by_storage[storage_id]
            if layout == kept_layout:
                continue

            # Distinct views into the same storage cannot be written as-is.
            prepared[name] = tensor.clone()

        return prepared

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. RSR preprocessed tensors
    rsr_tensors = {}
    for layer_name, arrays in rsr_data.items():
        for key, tensor in arrays.items():
            rsr_tensors[f"{layer_name}.{key}"] = tensor
    if rsr_tensors:
        safetensors_save(rsr_tensors, str(output_dir / "rsr_weights.safetensors"))

    # 2. Non-quantized parameters (embeddings, norms, weight_scale, …)
    if non_quantized:
        safetensors_save(
            _prepare_safetensors_state(non_quantized),
            str(output_dir / "non_quantized_weights.safetensors"),
        )

    # 3. RSR metadata
    rsr_config = {
        "model_name": model_name,
        "k": k,
        "version": version,
        "layers": layer_meta,
    }
    (output_dir / "rsr_config.json").write_text(json.dumps(rsr_config, indent=2))

    # 4. Original model config
    if model_config:
        (output_dir / "config.json").write_text(json.dumps(model_config, indent=2))


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def _load_best_k_map(path: str | Path | None, device: str) -> dict[str, int]:
    """Load a best-k JSON file and return a ``{\"NROWSxNCOLS\": k}`` mapping.

    When *path* is ``None``, try the default location produced by
    ``benchmarking.bit_1_58.bench_best_k``:
    ``<project_root>/benchmarking/bit_1_58/reports/best_k_{device}.json``

    Returns an empty dict if the file does not exist.
    """
    if path is None:
        path = _PROJECT_ROOT / "benchmarking" / "bit_1_58" / "reports" / f"best_k_{device}.json"
    else:
        path = Path(path)

    if not path.exists():
        return {}

    raw = json.loads(path.read_text())
    return {shape_key: int(entry["k"]) for shape_key, entry in raw.items()}


def preprocess_model(
    model_name_or_path: str,
    output_dir: str,
    k: int = 8,
    version: str = "adaptive",
    device: str = "cpu",
    trust_remote_code: bool = False,
    best_k_json: str | Path | None = None,
):
    """Preprocess a HuggingFace quantized model for RSR inference.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        output_dir: Directory to write preprocessed model into.
        k: Block height for RSR decomposition (used as fallback when
            *best_k_json* has no entry for a given layer shape).
        version: RSR version to use (default: adaptive).
        device: Device for model loading ("cpu" or "cuda").
        trust_remote_code: Whether to trust remote code when loading model.
        best_k_json: Path to a JSON file mapping ``"NROWSxNCOLS"`` to the
            best k for that shape.  Produced by
            ``python -m benchmarking.bit_1_58.bench_best_k``.
            When ``None`` (default) the standard location
            ``benchmarking/bit_1_58/reports/best_k_{device}.json`` is tried
            automatically; if the file does not exist, *k* is used for every
            layer.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    output_path = Path(output_dir) / f"{_model_slug(model_name_or_path)}_{device}"

    # --- load model --------------------------------------------------------
    print(f"Loading config from {model_name_or_path} ...")
    config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code,
    )

    print(f"Loading model from {model_name_or_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch.float32,
        device_map=device,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    # --- discover layers ---------------------------------------------------
    ternary_layers = get_ternary_layers(model)
    if not ternary_layers:
        print("No quantized (ternary) layers found — nothing to do.")
        return

    non_quantized = get_non_quantized_state(model, set(ternary_layers.keys()))
    print(
        f"Found {len(ternary_layers)} quantized layers, "
        f"{len(non_quantized)} non-quantized tensors."
    )

    # --- load per-shape best k (if available) --------------------------------
    best_k_map = _load_best_k_map(best_k_json, device)
    if best_k_map:
        print(f"Loaded per-shape best-k map ({len(best_k_map)} entries). "
              f"Fallback k={k}.")
    else:
        print(f"No best-k map found; using k={k} for all layers.")

    # --- preprocess --------------------------------------------------------
    rsr_data: dict[str, dict] = {}
    layer_meta: dict[str, dict] = {}

    preprocess_fn = preprocess_layer_cuda if device == "cuda" else preprocess_layer_cpu

    for name, module in tqdm(ternary_layers.items(), desc="Preprocessing"):
        w = module.weight.data.cpu()

        if w.dtype == torch.uint8:
            # Packed format: 4 ternary values per byte
            weight = unpack_ternary_weights(w)
        elif getattr(module, "online_quant", False):
            # Online-quantized model (e.g. bitnet-bf16): weights are
            # full-precision, quantize to ternary and derive weight_scale.
            w_float = w.float()
            mean_abs = w_float.abs().mean().clamp_(min=1e-5)
            weight = (w_float / mean_abs).round().clamp(-1, 1).to(torch.int8)
            # Store the derived weight_scale so inference can rescale
            non_quantized[f"{name}.weight_scale"] = mean_abs.unsqueeze(0)
        else:
            # Already dequantized float {-1, 0, +1}
            weight = w

        n_rows, n_cols = weight.shape
        shape_key = f"{n_rows}x{n_cols}"
        layer_k = best_k_map.get(shape_key, k)
        arrays = preprocess_fn(weight, layer_k)
        rsr_data[name] = arrays

        meta = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "k": layer_k,
            "backend": device,
            "weight_scale_mode": _detect_weight_scale_mode(module),
        }
        # CUDA v2.0: record row padding and block count for runtime setup
        if device == "cuda":
            n_rows_padded = ((n_rows + layer_k - 1) // layer_k) * layer_k
            meta["n_rows_padded"] = n_rows_padded
            meta["num_blocks"] = n_rows_padded // layer_k
        layer_meta[name] = meta

    # --- save --------------------------------------------------------------
    print(f"Saving to {output_path} ...")
    save_preprocessed(
        rsr_data=rsr_data,
        non_quantized=non_quantized,
        layer_meta=layer_meta,
        model_config=config.to_dict() if hasattr(config, "to_dict") else {},
        output_dir=output_path,
        k=k,
        version=version,
        model_name=model_name_or_path,
    )

    # --- summary -----------------------------------------------------------
    total_rsr = sum(
        t.numel() * t.element_size()
        for arrays in rsr_data.values()
        for t in arrays.values()
    )
    total_nq = sum(t.numel() * t.element_size() for t in non_quantized.values())
    print(
        f"Done.\n"
        f"  RSR data:        {total_rsr / 1e6:>8.1f} MB\n"
        f"  Non-quantized:   {total_nq / 1e6:>8.1f} MB\n"
        f"  Total:           {(total_rsr + total_nq) / 1e6:>8.1f} MB"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Preprocess a HuggingFace quantized (ternary) model for RSR inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m", required=True,
        help="HuggingFace model ID or local path (e.g. microsoft/bitnet-b1.58-2B-4T)",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for preprocessed model",
    )
    parser.add_argument(
        "--k", type=int, default=8,
        help="Block height for RSR decomposition",
    )
    parser.add_argument(
        "--version", default="adaptive",
        help="RSR multiplier version to use",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Device for model loading",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="Trust remote code when loading model",
    )
    parser.add_argument(
        "--best-k-json", default=None,
        help="Path to best-k JSON (NROWSxNCOLS -> k) produced by "
             "benchmarking.bit_1_58.bench_best_k. "
             "Default: benchmarking/bit_1_58/reports/best_k_{device}.json",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    preprocess_model(
        model_name_or_path=args.model,
        output_dir=args.output,
        k=args.k,
        version=args.version,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        best_k_json=args.best_k_json,
    )


if __name__ == "__main__":
    main()
