"""
Benchmark RSR vs HF ternary inference on causal LMs.

Usage:
    python -m benchmarking.llms.bench_inference \
        --model-dir integrations/hf/preprocessed \
        --device cuda \
        --prompt "Write the numbers from one to two hundred in words separated by commas only:" \
        --max-new-tokens 64 --warmup 1 --repeats 3
"""

import argparse
import time
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _timed_generate(model, tokenizer, prompt, max_new_tokens, use_chat_template):
    import torch
    from integrations.hf.model_infer import generate_text

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    text = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        use_chat_template=use_chat_template,
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0
    return text, elapsed


def bench_one(
    label, load_fn, prompt, max_new_tokens, use_chat_template, warmup, repeats
):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    print("Loading model ...")
    t0 = time.perf_counter()
    model, tokenizer = load_fn()
    load_time = time.perf_counter() - t0
    print(f"  Load time: {load_time:.2f}s")

    # Warmup
    for i in range(warmup):
        text, dt = _timed_generate(
            model, tokenizer, prompt, max_new_tokens, use_chat_template
        )
        print(f"  Warmup {i+1}: {dt:.3f}s")

    # Timed runs
    times = []
    for i in range(repeats):
        text, dt = _timed_generate(
            model, tokenizer, prompt, max_new_tokens, use_chat_template
        )
        times.append(dt)
        n_tokens = len(tokenizer.encode(text))
        print(f"  Run {i+1}: {dt:.3f}s  ({n_tokens} tokens, {n_tokens/dt:.1f} tok/s)")

    avg = sum(times) / len(times)
    n_tokens = len(tokenizer.encode(text))
    print(f"  Average: {avg:.3f}s  ({n_tokens/avg:.1f} tok/s)")
    print(f"  Output: {text[:200]}...")

    import torch

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()

    return {
        "label": label,
        "avg_time": avg,
        "tok_per_s": n_tokens / avg,
        "n_tokens": n_tokens,
    }


def _discover_model_dirs(model_dir, device):
    """Find preprocessed model directories matching the given device suffix.

    Accepts either a single preprocessed model directory or a parent directory
    containing multiple preprocessed model subdirectories.
    """
    p = Path(model_dir)
    suffix = f"_{device}"
    # Single model directory: ends with the device suffix and has a config.
    if p.name.endswith(suffix) and (p / "rsr_config.json").exists():
        return [p]
    # Otherwise treat as parent directory containing multiple models.
    dirs = sorted(
        d
        for d in p.iterdir()
        if d.is_dir() and d.name.endswith(suffix) and (d / "rsr_config.json").exists()
    )
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RSR vs HF ternary inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Single preprocessed model directory or parent directory containing multiple",
    )
    parser.add_argument(
        "--prompt",
        required=False,
        default="Write the numbers from one to two hundred in words separated by commas only:",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument(
        "--device",
        required=True,
        choices=["cpu", "cuda"],
        help="Device to benchmark on (also selects model variants)",
    )
    _ALL_HF_DTYPES = {
        "cpu": ["float32", "bfloat16"],
        "cuda": ["float32", "bfloat16"],
    }
    # float16 available explicitly but excluded from defaults (ternary
    # models often overflow float16's limited range).
    _EXTRA_DTYPES = ["float16"]
    all_backend_names = (
        ["rsr"]
        + [f"hf_{d}" for d in _ALL_HF_DTYPES["cuda"]]
        + [f"hf_{d}" for d in _EXTRA_DTYPES]
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        choices=all_backend_names,
        help="Backends to benchmark (default: rsr + all HF dtypes for device)",
    )
    args = parser.parse_args()

    from integrations.hf.model_infer import load_preprocessed_model, load_hf_model

    if args.backends is None:
        args.backends = ["rsr"] + [f"hf_{d}" for d in _ALL_HF_DTYPES[args.device]]

    model_dirs = _discover_model_dirs(args.model_dir, args.device)
    if not model_dirs:
        parser.error(
            f"No preprocessed directories ending with '_{args.device}' found in {args.model_dir}"
        )

    use_chat = not args.no_chat_template
    all_results = []

    # Build full task list for progress tracking
    tasks = []
    for model_dir in model_dirs:
        model_name = model_dir.name.removesuffix(f"_{args.device}")
        loaders = {}
        if "rsr" in args.backends:
            md = str(model_dir)
            loaders["RSR"] = lambda md=md: load_preprocessed_model(
                md,
                device=args.device,
                dtype="bfloat16",
            )
        is_bf16_model = "bf16" in model_name or "bfloat16" in model_name
        hf_dtypes = (
            ["bfloat16"]
            if is_bf16_model
            else _ALL_HF_DTYPES[args.device] + _EXTRA_DTYPES
        )
        for dtype in hf_dtypes:
            if f"hf_{dtype}" in args.backends:
                md = str(model_dir)
                loaders[f"HF {dtype}"] = lambda md=md, dt=dtype: load_hf_model(
                    md,
                    device=args.device,
                    dtype=dt,
                )
        for label, load_fn in loaders.items():
            tasks.append((model_name, model_dir, label, load_fn))

    total = len(tasks)
    for i, (model_name, model_dir, label, load_fn) in enumerate(tasks):
        print(f"\n[{i+1}/{total}] {model_name} / {label}")
        try:
            r = bench_one(
                label,
                load_fn,
                args.prompt,
                args.max_new_tokens,
                use_chat,
                args.warmup,
                args.repeats,
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            all_results.append(
                {
                    "model": model_name,
                    "label": label,
                    "avg_time": float("nan"),
                    "tok_per_s": float("nan"),
                    "n_tokens": 0,
                    "error": str(exc),
                }
            )
            # Try to reset CUDA state; after a device-side assert the
            # context may be unrecoverable so we guard the cleanup too.
            import torch

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            import gc

            gc.collect()
            continue
        r["model"] = model_name
        all_results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY  (device={args.device})")
    print(f"{'='*60}")
    print(f"  {'Model':<30} {'Backend':<15} {'Avg Time':>10} {'Tok/s':>10}")
    print(f"  {'-'*65}")
    for r in all_results:
        if "error" in r:
            print(f"  {r['model']:<30} {r['label']:<15} {'FAILED':>10} {'—':>10}")
        else:
            print(
                f"  {r['model']:<30} {r['label']:<15} {r['avg_time']:>9.3f}s {r['tok_per_s']:>9.1f}"
            )


if __name__ == "__main__":
    main()
