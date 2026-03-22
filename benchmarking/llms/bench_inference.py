"""
Benchmark RSR vs HF ternary inference on causal LMs.

Usage:
    python -m benchmarking.llms.bench_inference \
        --model-dir integrations/hf/preprocessed/bitnet-b1.58-2B-4T_cuda \
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
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        use_chat_template=use_chat_template,
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0
    return text, elapsed


def bench_one(label, load_fn, prompt, max_new_tokens, use_chat_template, warmup, repeats):
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
        text, dt = _timed_generate(model, tokenizer, prompt, max_new_tokens, use_chat_template)
        print(f"  Warmup {i+1}: {dt:.3f}s")

    # Timed runs
    times = []
    for i in range(repeats):
        text, dt = _timed_generate(model, tokenizer, prompt, max_new_tokens, use_chat_template)
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
    import gc; gc.collect()

    return {"label": label, "avg_time": avg, "tok_per_s": n_tokens / avg, "n_tokens": n_tokens}


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RSR vs HF ternary inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", required=True,
                        help="Preprocessed model directory")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--device", default=None,
                        help="Device (auto-detected from dir name if omitted)")
    parser.add_argument("--backends", nargs="+",
                        default=["rsr", "hf_float32", "hf_bfloat16"],
                        help="Backends to benchmark")
    args = parser.parse_args()

    from integrations.hf.model_infer import (
        load_preprocessed_model, load_hf_model, _detect_device_from_dir,
    )

    device = args.device or _detect_device_from_dir(args.model_dir)
    use_chat = not args.no_chat_template

    loaders = {}
    if "rsr" in args.backends:
        loaders["RSR (CUDA kernel)"] = lambda: load_preprocessed_model(
            args.model_dir, device=device,
        )
    if "hf_float32" in args.backends:
        loaders["HF float32"] = lambda: load_hf_model(
            args.model_dir, device=device, dtype="float32",
        )
    if "hf_bfloat16" in args.backends:
        loaders["HF bfloat16"] = lambda: load_hf_model(
            args.model_dir, device=device, dtype="bfloat16",
        )

    results = []
    for label, load_fn in loaders.items():
        r = bench_one(label, load_fn, args.prompt, args.max_new_tokens,
                      use_chat, args.warmup, args.repeats)
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY  ({Path(args.model_dir).name})")
    print(f"{'='*60}")
    print(f"  {'Backend':<25} {'Avg Time':>10} {'Tok/s':>10}")
    print(f"  {'-'*45}")
    for r in results:
        print(f"  {r['label']:<25} {r['avg_time']:>9.3f}s {r['tok_per_s']:>9.1f}")


if __name__ == "__main__":
    main()
