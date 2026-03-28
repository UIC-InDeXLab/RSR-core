const BASE = "/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail;
    try {
      const json = await res.json();
      // FastAPI wraps detail in { detail: ... }
      const d = json.detail ?? json;
      if (typeof d === "object" && d.error) {
        const err = new Error(d.error);
        err.traceback = d.traceback || null;
        throw err;
      }
      throw new Error(typeof d === "string" ? d : JSON.stringify(d));
    } catch (e) {
      if (e.traceback !== undefined) throw e;
      const text = e.message || `${res.status}`;
      throw new Error(text);
    }
  }
  return res.json();
}

// HuggingFace
export const searchHFModels = (q) => request(`/hf/search?q=${encodeURIComponent(q)}`);

// Models
export const listModels = () => request("/models");
export const getModelConfig = (name) => request(`/models/${name}/config`);
export const deleteModel = (name) => request(`/models/${name}`, { method: "DELETE" });

// Preprocessing
export const startPreprocess = (body) =>
  request("/preprocess", { method: "POST", body: JSON.stringify(body) });
export const getPreprocessStatus = (jobId) => request(`/preprocess/${jobId}`);

// Inference
export const runInference = (body) =>
  request("/inference", { method: "POST", body: JSON.stringify(body) });

// Best-k availability
export const getBestKAvailability = () =>
  request("/benchmarks/best-k-availability");

// Multipliers
export const listMultipliers = () => request("/multipliers");

// Benchmarks
export const listBenchmarkReports = () => request("/benchmarks/reports");
export const getBestK = (device) => request(`/benchmarks/best-k/${device}`);
export const getShapesResults = (category, device) =>
  request(`/benchmarks/shapes/${category}/${device}`);
export const runBenchmark = (body) =>
  request("/benchmarks/run", { method: "POST", body: JSON.stringify(body) });
export const getBenchmarkJob = (jobId) => request(`/benchmarks/job/${jobId}`);

// System
export const getSystemInfo = () => request("/system");
