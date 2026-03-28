"""
FastAPI backend for RSR-core UI.

Wraps existing preprocessing, inference, and benchmarking scripts
so the React frontend can drive them via REST endpoints.
"""

import csv
import json
import importlib
import inspect
import asyncio
import gc
import sys
import time
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

app = FastAPI(title="RSR-core UI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PREPROCESSED_DIR = _PROJECT_ROOT / "integrations" / "hf" / "preprocessed"
REPORTS_DIR_158 = _PROJECT_ROOT / "benchmarking" / "bit_1_58" / "reports"
REPORTS_DIR_1 = _PROJECT_ROOT / "benchmarking" / "bit_1" / "reports"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_csv_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


# ── HuggingFace model search ────────────────────────────────────────────────

@app.get("/api/hf/search")
async def hf_search_models(q: str = "1.58"):
    """Search HuggingFace Hub for ternary / 1.58-bit models."""
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        models = list(api.list_models(search=q, limit=30, sort="downloads", direction=-1))
    except Exception as e:
        raise HTTPException(500, str(e))
    return [
        {
            "id": m.id,
            "downloads": m.downloads,
            "likes": m.likes,
            "pipeline_tag": m.pipeline_tag,
            "tags": m.tags[:10] if m.tags else [],
        }
        for m in models
    ]


# ── Preprocessed model management ───────────────────────────────────────────

@app.get("/api/models")
async def list_models():
    """List all preprocessed models."""
    if not PREPROCESSED_DIR.exists():
        return []
    models = []
    for d in sorted(PREPROCESSED_DIR.iterdir()):
        if not d.is_dir():
            continue
        config_path = d / "rsr_config.json"
        if not config_path.exists():
            continue
        cfg = _read_json(config_path)
        size_mb = sum(f.stat().st_size for f in d.iterdir() if f.is_file()) / 1e6
        models.append({
            "name": d.name,
            "path": str(d),
            "model_name": cfg.get("model_name", ""),
            "k": cfg.get("k"),
            "version": cfg.get("version"),
            "num_layers": len(cfg.get("layers", {})),
            "size_mb": round(size_mb, 1),
            "device": "cuda" if d.name.endswith("_cuda") else "cpu",
        })
    return models


@app.get("/api/models/{model_dir_name}/config")
async def get_model_config(model_dir_name: str):
    """Get full RSR config for a preprocessed model."""
    d = PREPROCESSED_DIR / model_dir_name
    if not d.exists():
        raise HTTPException(404, "Model not found")
    cfg = _read_json(d / "rsr_config.json")
    hf_cfg = {}
    if (d / "config.json").exists():
        hf_cfg = _read_json(d / "config.json")
    return {"rsr_config": cfg, "hf_config": hf_cfg}


@app.delete("/api/models/{model_dir_name}")
async def delete_model(model_dir_name: str):
    """Delete a preprocessed model directory."""
    import shutil
    d = PREPROCESSED_DIR / model_dir_name
    if not d.exists():
        raise HTTPException(404, "Model not found")
    shutil.rmtree(d)
    return {"status": "deleted", "name": model_dir_name}


# ── Preprocessing ───────────────────────────────────────────────────────────

class PreprocessRequest(BaseModel):
    model: str
    k: int = 8
    version: str = "adaptive"
    device: str = "cpu"
    trust_remote_code: bool = False
    use_best_k: bool = False


# Track running jobs
_jobs: dict[str, dict] = {}
_job_lock = threading.Lock()


@app.post("/api/preprocess")
async def start_preprocessing(req: PreprocessRequest):
    """Start preprocessing a model (runs in background thread)."""
    job_id = f"{req.model.replace('/', '_')}_{req.device}_{int(time.time())}"

    def run():
        with _job_lock:
            _jobs[job_id] = {"status": "running", "progress": "Loading model...", "stage": "loading", "current": 0, "total": 0}
        try:
            from integrations.hf.model_prep import preprocess_model

            def on_progress(stage, current, total, detail):
                with _job_lock:
                    _jobs[job_id] = {
                        "status": "running",
                        "progress": detail,
                        "stage": stage,
                        "current": current,
                        "total": total,
                    }

            best_k_path = None
            if req.use_best_k:
                candidate = REPORTS_DIR_158 / f"best_k_{req.device}.json"
                if candidate.exists():
                    best_k_path = str(candidate)
            preprocess_model(
                model_name_or_path=req.model,
                output_dir=str(PREPROCESSED_DIR),
                k=req.k,
                version=req.version,
                device=req.device,
                trust_remote_code=req.trust_remote_code,
                best_k_json=best_k_path,
                progress_callback=on_progress,
            )
            with _job_lock:
                _jobs[job_id] = {"status": "completed", "progress": "Done", "stage": "done", "current": 1, "total": 1}
        except Exception as e:
            import traceback
            with _job_lock:
                _jobs[job_id] = {"status": "error", "progress": str(e), "traceback": traceback.format_exc()}

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "started"}


@app.get("/api/preprocess/{job_id}")
async def get_preprocess_status(job_id: str):
    """Check preprocessing job status."""
    with _job_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


# ── Inference ────────────────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    model_dir: str
    prompt: str
    max_new_tokens: int = 64
    use_chat_template: bool = True
    device: str | None = None
    dtype: str | None = None
    backend: str = "rsr"  # "rsr" or "hf"


@app.post("/api/inference")
async def run_inference(req: InferenceRequest):
    """Run inference on a preprocessed model."""
    import torch

    model_dir = Path(req.model_dir)
    if not model_dir.is_absolute():
        model_dir = PREPROCESSED_DIR / req.model_dir

    if not model_dir.exists():
        raise HTTPException(404, f"Model directory not found: {model_dir}")

    device = req.device
    if device is None:
        device = "cuda" if model_dir.name.endswith("_cuda") else "cpu"

    try:
        from integrations.hf.model_infer import (
            load_preprocessed_model,
            load_hf_model,
            generate_text,
        )

        t_load_start = time.perf_counter()
        if req.backend == "rsr":
            model, tokenizer = load_preprocessed_model(
                str(model_dir), device=device, dtype=req.dtype,
            )
        else:
            model, tokenizer = load_hf_model(
                str(model_dir), device=device, dtype=req.dtype,
            )
        t_load = time.perf_counter() - t_load_start

        t_gen_start = time.perf_counter()

        # Use generate_text but capture timing ourselves
        if req.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": req.prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            inputs = tokenizer(req.prompt, return_tensors="pt")

        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.inference_mode():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_infer_start = time.perf_counter()
            output_ids = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_infer = time.perf_counter() - t_infer_start

        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0, prompt_length:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        n_tokens = len(generated_ids)
        tok_per_sec = n_tokens / t_infer if t_infer > 0 else 0

        # Clean up
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "text": text,
            "n_tokens": n_tokens,
            "load_time_s": round(t_load, 3),
            "inference_time_s": round(t_infer, 3),
            "tok_per_sec": round(tok_per_sec, 1),
            "device": device,
            "backend": req.backend,
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(500, detail={"error": str(e), "traceback": tb})


# ── Multiplier registry ─────────────────────────────────────────────────────

@app.get("/api/multipliers")
async def list_multipliers():
    """Discover all available multiplier implementations."""
    results = []

    for bit_width, bit_dir_name in [("1-bit", "bit_1"), ("1.58-bit", "bit_1_58")]:
        for platform in ["cpu", "cuda"]:
            pkg_dir = _PROJECT_ROOT / "multiplier" / bit_dir_name / platform
            if not pkg_dir.exists():
                continue
            for py_file in sorted(pkg_dir.glob("*.py")):
                if py_file.stem.startswith("_") or py_file.stem == "__init__":
                    continue
                module_path = f"multiplier.{bit_dir_name}.{platform}.{py_file.stem}"
                try:
                    mod = importlib.import_module(module_path)
                    classes = [
                        name for name, obj in inspect.getmembers(mod, inspect.isclass)
                        if obj.__module__ == module_path and name.endswith("Multiplier")
                    ]
                except Exception:
                    classes = [f"(import error: {py_file.stem})"]

                for cls_name in classes:
                    results.append({
                        "bit_width": bit_width,
                        "platform": platform,
                        "module": py_file.stem,
                        "class_name": cls_name,
                        "file": str(py_file.relative_to(_PROJECT_ROOT)),
                    })

    # Also add pytorch baselines
    for bit_dir_name, bit_width in [("bit_1", "1-bit"), ("bit_1_58", "1.58-bit")]:
        pt_path = _PROJECT_ROOT / "multiplier" / bit_dir_name / "pytorch.py"
        if pt_path.exists():
            try:
                mod = importlib.import_module(f"multiplier.{bit_dir_name}.pytorch")
                for name, obj in inspect.getmembers(mod, inspect.isclass):
                    if obj.__module__ == mod.__name__ and name.endswith("Multiplier"):
                        results.append({
                            "bit_width": bit_width,
                            "platform": "pytorch",
                            "module": "pytorch",
                            "class_name": name,
                            "file": str(pt_path.relative_to(_PROJECT_ROOT)),
                        })
            except Exception:
                pass

    return results


# ── Benchmark reports ────────────────────────────────────────────────────────

@app.get("/api/benchmarks/reports")
async def list_benchmark_reports():
    """List available benchmark report files."""
    reports = []
    for label, rdir in [("bit_1_58", REPORTS_DIR_158), ("bit_1", REPORTS_DIR_1)]:
        if not rdir.exists():
            continue
        for f in sorted(rdir.iterdir()):
            if f.suffix in (".csv", ".json", ".png"):
                reports.append({
                    "category": label,
                    "filename": f.name,
                    "path": str(f),
                    "type": f.suffix[1:],
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })
    return reports


@app.get("/api/benchmarks/best-k-availability")
async def best_k_availability():
    """Check which devices have best-k reports and return their contents."""
    result = {}
    for dev in ("cpu", "cuda"):
        path = REPORTS_DIR_158 / f"best_k_{dev}.json"
        if path.exists():
            result[dev] = _read_json(path)
    return result


@app.get("/api/benchmarks/best-k/{device}")
async def get_best_k(device: str):
    """Get best-k JSON for a device."""
    path = REPORTS_DIR_158 / f"best_k_{device}.json"
    if not path.exists():
        raise HTTPException(404, f"No best-k report for {device}")
    return _read_json(path)


@app.get("/api/benchmarks/shapes/{category}/{device}")
async def get_shapes_results(category: str, device: str):
    """Get shape benchmark CSV data."""
    rdir = REPORTS_DIR_158 if category == "bit_1_58" else REPORTS_DIR_1
    path = rdir / f"results_shapes_{device}.csv"
    if not path.exists():
        raise HTTPException(404, f"No results for {category}/{device}")
    return _read_csv_rows(path)


@app.get("/api/benchmarks/plots/{category}/{filename}")
async def get_benchmark_plot(category: str, filename: str):
    """Serve a benchmark plot image."""
    from fastapi.responses import FileResponse
    rdir = REPORTS_DIR_158 if category == "bit_1_58" else REPORTS_DIR_1
    path = rdir / filename
    if not path.exists():
        raise HTTPException(404, "Plot not found")
    return FileResponse(path, media_type="image/png")


# ── Run benchmark ────────────────────────────────────────────────────────────

class BenchmarkRequest(BaseModel):
    type: str  # "best_k", "shapes_cpu", "shapes_cuda", "inference"
    device: str = "cpu"
    # For inference benchmark
    model_dir: str | None = None
    prompt: str = "Hello, how are you?"
    max_new_tokens: int = 32
    warmup: int = 1
    repeats: int = 3
    backends: list[str] | None = None


@app.post("/api/benchmarks/run")
async def run_benchmark(req: BenchmarkRequest):
    """Start a benchmark run (background thread)."""
    job_id = f"bench_{req.type}_{req.device}_{int(time.time())}"

    def run():
        with _job_lock:
            _jobs[job_id] = {"status": "running", "progress": f"Running {req.type}..."}
        try:
            if req.type == "inference" and req.model_dir:
                from benchmarking.llms.bench_inference import bench_one
                from integrations.hf.model_infer import load_preprocessed_model, load_hf_model

                model_dir = req.model_dir
                if not Path(model_dir).is_absolute():
                    model_dir = str(PREPROCESSED_DIR / model_dir)

                results = []
                backends = req.backends or ["rsr", "hf_bfloat16"]

                for backend in backends:
                    if backend == "rsr":
                        load_fn = lambda: load_preprocessed_model(
                            model_dir, device=req.device, dtype="bfloat16",
                        )
                        label = "RSR"
                    else:
                        dtype = backend.replace("hf_", "")
                        load_fn = lambda dt=dtype: load_hf_model(
                            model_dir, device=req.device, dtype=dt,
                        )
                        label = f"HF {backend.replace('hf_', '')}"

                    try:
                        r = bench_one(
                            label, load_fn, req.prompt,
                            req.max_new_tokens,
                            True, req.warmup, req.repeats,
                        )
                        results.append(r)
                    except Exception as e:
                        results.append({"label": label, "error": str(e)})

                with _job_lock:
                    _jobs[job_id] = {"status": "completed", "results": results}
            else:
                with _job_lock:
                    _jobs[job_id] = {"status": "error", "progress": f"Unknown benchmark type: {req.type}"}
        except Exception as e:
            with _job_lock:
                _jobs[job_id] = {"status": "error", "progress": str(e)}

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return {"job_id": job_id}


class MatvecBenchmarkRequest(BaseModel):
    shapes: list[str]          # e.g. ["4096x4096", "2560x6912"]
    k_values: list[int] = [2, 4, 6, 8, 10]
    device: str = "cpu"        # "cpu" or "cuda"
    bit_width: str = "1.58"    # "1" or "1.58"
    warmup: int = 5
    repeats: int = 20


@app.post("/api/benchmarks/run-matvec")
async def run_matvec_benchmark(req: MatvecBenchmarkRequest):
    """Start a kernel-level matvec benchmark (background thread)."""
    job_id = f"bench_matvec_{req.device}_{int(time.time())}"

    def run():
        import numpy as np
        import torch

        with _job_lock:
            _jobs[job_id] = {
                "status": "running",
                "progress": "Discovering multipliers...",
                "current": 0,
                "total": len(req.shapes),
            }

        try:
            # Parse shapes
            shapes = []
            for s in req.shapes:
                for sep in ("x", "X", ","):
                    if sep in s:
                        parts = s.split(sep)
                        shapes.append((int(parts[0].strip()), int(parts[1].strip())))
                        break

            bit_dir = "bit_1_58" if req.bit_width == "1.58" else "bit_1"
            is_cuda = req.device == "cuda"

            # --- Discover multipliers ---
            # Baselines (no k)
            pt_mod = importlib.import_module(f"multiplier.{bit_dir}.pytorch")
            baselines = []
            for name, obj in inspect.getmembers(pt_mod, inspect.isclass):
                if obj.__module__ == pt_mod.__name__ and name.endswith("Multiplier"):
                    label = name.replace("Multiplier", "").replace("Pytorch", "pytorch_").strip("_")
                    if not label:
                        label = "pytorch"
                    baselines.append((label, obj))
            # Keep only fp32 and bf16 for brevity
            baselines = [
                (l, c) for l, c in baselines
                if any(tag in l.lower() for tag in ("pytorch", "fp32", "bf16"))
            ]
            if not baselines:
                baselines = [(name, obj) for name, obj in baselines[:2]]

            # RSR multipliers (need k)
            rsr_versions = []
            platform = "cuda" if is_cuda else "cpu"
            pkg_dir = _PROJECT_ROOT / "multiplier" / bit_dir / platform
            if pkg_dir.exists():
                for py_file in sorted(pkg_dir.glob("*.py")):
                    if py_file.stem.startswith("_") or py_file.stem in ("__init__", "base"):
                        continue
                    module_path = f"multiplier.{bit_dir}.{platform}.{py_file.stem}"
                    try:
                        mod = importlib.import_module(module_path)
                        cls = next(
                            (obj for _, obj in inspect.getmembers(mod, inspect.isclass)
                             if obj.__module__ == module_path and obj.__name__.endswith("Multiplier")),
                            None,
                        )
                        if cls is None:
                            continue
                        needs_k = "k" in inspect.signature(cls.__init__).parameters
                        if needs_k:
                            rsr_versions.append((py_file.stem, cls))
                    except Exception:
                        continue

            # Pick primary RSR version (prefer "nonsquare" or last available)
            primary_rsr = None
            for stem, cls in rsr_versions:
                if "nonsquare" in stem or "v2_0" in stem:
                    primary_rsr = ("RSR", cls)
                    break
            if primary_rsr is None and rsr_versions:
                primary_rsr = ("RSR", rsr_versions[-1][1])

            # --- Bench helpers ---
            def bench_cpu(multiplier, v, warmup, repeats):
                for _ in range(warmup):
                    multiplier(v)
                times = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    multiplier(v)
                    t1 = time.perf_counter()
                    times.append(t1 - t0)
                return float(np.median(times))

            def bench_cuda(multiplier, v, warmup, repeats):
                for _ in range(warmup):
                    multiplier(v)
                torch.cuda.synchronize()
                times = []
                for _ in range(repeats):
                    start_ev = torch.cuda.Event(enable_timing=True)
                    end_ev = torch.cuda.Event(enable_timing=True)
                    start_ev.record()
                    multiplier(v)
                    end_ev.record()
                    torch.cuda.synchronize()
                    times.append(start_ev.elapsed_time(end_ev) / 1000.0)
                return float(np.median(times))

            bench_fn = bench_cuda if is_cuda else bench_cpu

            # --- Run benchmarks ---
            results = []

            for idx, (n_rows, n_cols) in enumerate(shapes):
                with _job_lock:
                    _jobs[job_id]["progress"] = f"Benchmarking {n_rows}x{n_cols}..."
                    _jobs[job_id]["current"] = idx

                # Create matrix and vector
                if req.bit_width == "1.58":
                    M = torch.randint(-1, 2, (n_rows, n_cols), dtype=torch.float32)
                else:
                    M = torch.randint(0, 2, (n_rows, n_cols), dtype=torch.float32)

                v_device = "cuda" if is_cuda else "cpu"
                v = torch.randn(n_cols, dtype=torch.float32, device=v_device)

                # Baseline timings
                baseline_results = {}
                for label, cls in baselines:
                    try:
                        m_input = M.cuda() if is_cuda else M
                        mul = cls(m_input)
                        t = bench_fn(mul, v, req.warmup, req.repeats)
                        baseline_results[label] = round(t * 1e3, 4)
                    except Exception:
                        baseline_results[label] = None

                # RSR per k
                if primary_rsr:
                    rsr_label, rsr_cls = primary_rsr
                    for k in req.k_values:
                        if n_rows % k != 0:
                            continue
                        try:
                            mul = rsr_cls(M, k)
                            t = bench_fn(mul, v, req.warmup, req.repeats)
                            rsr_ms = round(t * 1e3, 4)
                            # Pick a reference baseline for speedup
                            ref_key = next(
                                (key for key in ("pytorch_BF16", "pytorch_bf16", "pytorch")
                                 if key in baseline_results and baseline_results[key] is not None),
                                None,
                            )
                            fp32_key = next(
                                (key for key in ("pytorch", "pytorch_FP32", "pytorch_fp32")
                                 if key in baseline_results and baseline_results[key] is not None),
                                None,
                            )
                            row = {
                                "shape": f"{n_rows}x{n_cols}",
                                "n_rows": n_rows,
                                "n_cols": n_cols,
                                "k": k,
                                "rsr_ms": rsr_ms,
                            }
                            # Attach all baselines
                            for bl, val in baseline_results.items():
                                row[f"{bl}_ms"] = val
                            # Compute speedups
                            if fp32_key and baseline_results[fp32_key]:
                                row["fp32_ms"] = baseline_results[fp32_key]
                                row["speedup_vs_fp32"] = round(baseline_results[fp32_key] / rsr_ms, 3)
                            if ref_key and baseline_results[ref_key]:
                                row["bf16_ms"] = baseline_results[ref_key]
                                row["speedup_vs_bf16"] = round(baseline_results[ref_key] / rsr_ms, 3)
                            results.append(row)
                        except Exception as e:
                            results.append({
                                "shape": f"{n_rows}x{n_cols}",
                                "n_rows": n_rows,
                                "n_cols": n_cols,
                                "k": k,
                                "error": str(e),
                            })

                # Clean up
                del M
                if is_cuda:
                    torch.cuda.empty_cache()
                gc.collect()

            with _job_lock:
                _jobs[job_id] = {
                    "status": "completed",
                    "results": results,
                    "current": len(shapes),
                    "total": len(shapes),
                }

        except Exception as e:
            import traceback
            with _job_lock:
                _jobs[job_id] = {
                    "status": "error",
                    "progress": str(e),
                    "traceback": traceback.format_exc(),
                }

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/benchmarks/job/{job_id}")
async def get_benchmark_status(job_id: str):
    """Check benchmark job status."""
    with _job_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


# ── System info ──────────────────────────────────────────────────────────────

@app.get("/api/system")
async def system_info():
    """Basic system info for the UI."""
    import torch
    return {
        "project_root": str(_PROJECT_ROOT),
        "preprocessed_dir": str(PREPROCESSED_DIR),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8042)
