import { useEffect, useState } from "react";
import {
  listBenchmarkReports,
  getBestK,
  getShapesResults,
  runBenchmark,
  getBenchmarkJob,
  listModels,
} from "../api";
import Card from "../components/Card";
import Badge from "../components/Badge";
import Spinner from "../components/Spinner";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
} from "recharts";

function BestKChart({ data }) {
  const chartData = Object.entries(data).map(([shape, info]) => ({
    shape,
    rsr_ms: info.rsr_ms,
    fp32_ms: info.fp32_ms,
    bf16_ms: info.bf16_ms,
    speedup_fp32: info.speedup_vs_fp32,
    speedup_bf16: info.speedup_vs_bf16,
    k: info.k,
  }));

  return (
    <div className="space-y-6">
      <div>
        <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Latency Comparison (ms)
        </h4>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} margin={{ top: 5, right: 20, bottom: 60, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis
              dataKey="shape"
              tick={{ fill: "#9ca3af", fontSize: 11 }}
              angle={-45}
              textAnchor="end"
              height={60}
            />
            <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8 }}
              labelStyle={{ color: "#e5e7eb" }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="fp32_ms" name="FP32" fill="#6b7280" radius={[2, 2, 0, 0]} />
            <Bar dataKey="bf16_ms" name="BF16" fill="#8b5cf6" radius={[2, 2, 0, 0]} />
            <Bar dataKey="rsr_ms" name="RSR" fill="#06b6d4" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div>
        <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Speedup over Baselines
        </h4>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={chartData} margin={{ top: 5, right: 20, bottom: 60, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis
              dataKey="shape"
              tick={{ fill: "#9ca3af", fontSize: 11 }}
              angle={-45}
              textAnchor="end"
              height={60}
            />
            <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} label={{ value: "x speedup", angle: -90, position: "insideLeft", style: { fill: "#9ca3af", fontSize: 11 } }} />
            <Tooltip
              contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8 }}
              labelStyle={{ color: "#e5e7eb" }}
              formatter={(v) => `${v.toFixed(2)}x`}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="speedup_fp32" name="vs FP32" fill="#10b981" radius={[2, 2, 0, 0]} />
            <Bar dataKey="speedup_bf16" name="vs BF16" fill="#f59e0b" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function ShapesChart({ data, category }) {
  if (!data || data.length === 0) return <p className="text-gray-500 text-sm">No data</p>;

  // Group by shape, pick best RSR time per shape
  const shapeMap = {};
  for (const row of data) {
    const shape = `${row.rows}x${row.cols}`;
    if (!shapeMap[shape]) shapeMap[shape] = { shape };
    const methods = Object.keys(row).filter((k) => !["rows", "cols", "k"].includes(k));
    for (const m of methods) {
      const val = parseFloat(row[m]);
      if (!isNaN(val)) {
        if (!shapeMap[shape][m] || val < shapeMap[shape][m]) {
          shapeMap[shape][m] = val;
        }
      }
    }
  }

  const chartData = Object.values(shapeMap);
  const methods = Object.keys(chartData[0] || {}).filter((k) => k !== "shape");
  const colors = { pytorch: "#6b7280", BitNet: "#8b5cf6", RSR: "#06b6d4", "T-MAC": "#f59e0b" };

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 60, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis
          dataKey="shape"
          tick={{ fill: "#9ca3af", fontSize: 11 }}
          angle={-45}
          textAnchor="end"
          height={60}
        />
        <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} label={{ value: "ms", angle: -90, position: "insideLeft", style: { fill: "#9ca3af", fontSize: 11 } }} />
        <Tooltip
          contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8 }}
          labelStyle={{ color: "#e5e7eb" }}
          formatter={(v) => `${v.toFixed(3)} ms`}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {methods.map((m) => (
          <Line
            key={m}
            type="monotone"
            dataKey={m}
            name={m}
            stroke={colors[m] || "#06b6d4"}
            strokeWidth={2}
            dot={{ r: 3 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export default function BenchmarksPage() {
  const [tab, setTab] = useState("best_k");
  const [device, setDevice] = useState("cpu");
  const [bestK, setBestK] = useState(null);
  const [shapes158, setShapes158] = useState(null);
  const [shapes1, setShapes1] = useState(null);
  const [reports, setReports] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [benchJob, setBenchJob] = useState(null);
  const [benchResult, setBenchResult] = useState(null);

  useEffect(() => {
    listBenchmarkReports().then(setReports).catch(() => {});
    listModels().then(setModels).catch(() => {});
  }, []);

  useEffect(() => {
    if (tab === "best_k") {
      getBestK(device).then(setBestK).catch(() => setBestK(null));
    } else if (tab === "shapes") {
      getShapesResults("bit_1_58", device).then(setShapes158).catch(() => setShapes158(null));
      getShapesResults("bit_1", device).then(setShapes1).catch(() => setShapes1(null));
    }
  }, [tab, device]);

  const handleRunInferenceBench = async () => {
    if (!selectedModel) return;
    setBenchResult(null);
    try {
      const res = await runBenchmark({
        type: "inference",
        device,
        model_dir: selectedModel,
        backends: ["rsr", `hf_bfloat16`],
        warmup: 1,
        repeats: 2,
      });
      setBenchJob(res.job_id);

      // Poll
      const poll = setInterval(async () => {
        try {
          const status = await getBenchmarkJob(res.job_id);
          if (status.status !== "running") {
            clearInterval(poll);
            setBenchJob(null);
            if (status.results) setBenchResult(status.results);
          }
        } catch {
          clearInterval(poll);
        }
      }, 3000);
    } catch (e) {
      alert(e.message);
    }
  };

  const TABS = [
    { id: "best_k", label: "Best K" },
    { id: "shapes", label: "Shape Benchmarks" },
    { id: "inference", label: "Inference Benchmark" },
    { id: "reports", label: "Report Files" },
  ];

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Benchmarks</h1>
          <p className="text-gray-500 text-sm mt-1">Performance analysis and comparisons</p>
        </div>
        <select
          value={device}
          onChange={(e) => setDevice(e.target.value)}
          className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-cyan-500"
        >
          <option value="cpu">CPU</option>
          <option value="cuda">CUDA</option>
        </select>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 bg-gray-900 p-1 rounded-xl border border-gray-800">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex-1 px-4 py-2 rounded-lg text-sm transition-colors ${
              tab === t.id
                ? "bg-cyan-500/10 text-cyan-400 font-medium"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Best K tab */}
      {tab === "best_k" && (
        <Card title={`Best K per Shape (${device})`}>
          {bestK ? (
            <div className="space-y-6">
              <BestKChart data={bestK} />
              <div>
                <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Data Table</h4>
                <div className="overflow-auto rounded-lg border border-gray-800">
                  <table className="w-full text-xs">
                    <thead className="bg-gray-800">
                      <tr>
                        <th className="text-left px-3 py-2 text-gray-400">Shape</th>
                        <th className="text-right px-3 py-2 text-gray-400">Best k</th>
                        <th className="text-right px-3 py-2 text-gray-400">RSR (ms)</th>
                        <th className="text-right px-3 py-2 text-gray-400">FP32 (ms)</th>
                        <th className="text-right px-3 py-2 text-gray-400">BF16 (ms)</th>
                        <th className="text-right px-3 py-2 text-gray-400">vs FP32</th>
                        <th className="text-right px-3 py-2 text-gray-400">vs BF16</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(bestK)
                        .sort(([a], [b]) => a.localeCompare(b))
                        .map(([shape, info]) => (
                          <tr key={shape} className="border-t border-gray-800 hover:bg-gray-800/50">
                            <td className="px-3 py-1.5 text-gray-300 font-mono">{shape}</td>
                            <td className="text-right px-3 py-1.5 text-cyan-400 font-medium">{info.k}</td>
                            <td className="text-right px-3 py-1.5 text-gray-300">{info.rsr_ms?.toFixed(3)}</td>
                            <td className="text-right px-3 py-1.5 text-gray-400">{info.fp32_ms?.toFixed(3)}</td>
                            <td className="text-right px-3 py-1.5 text-gray-400">{info.bf16_ms?.toFixed(3)}</td>
                            <td className="text-right px-3 py-1.5 text-green-400">{info.speedup_vs_fp32?.toFixed(2)}x</td>
                            <td className="text-right px-3 py-1.5 text-yellow-400">{info.speedup_vs_bf16?.toFixed(2)}x</td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            <p className="text-gray-500 text-sm">
              No best-k report found for {device}. Run{" "}
              <code className="text-gray-400">python -m benchmarking.bit_1_58.bench_best_k --device {device}</code>
            </p>
          )}
        </Card>
      )}

      {/* Shapes tab */}
      {tab === "shapes" && (
        <div className="space-y-6">
          <Card title={`1.58-bit Shape Benchmarks (${device})`}>
            {shapes158 ? (
              <ShapesChart data={shapes158} category="bit_1_58" />
            ) : (
              <p className="text-gray-500 text-sm">No shape results for bit_1_58/{device}</p>
            )}
          </Card>
          <Card title={`1-bit Shape Benchmarks (${device})`}>
            {shapes1 ? (
              <ShapesChart data={shapes1} category="bit_1" />
            ) : (
              <p className="text-gray-500 text-sm">No shape results for bit_1/{device}</p>
            )}
          </Card>
        </div>
      )}

      {/* Inference benchmark tab */}
      {tab === "inference" && (
        <Card title="Inference Benchmark (RSR vs HF)">
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-gray-500 mb-1 block">Model</label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-cyan-500"
                >
                  <option value="">Select a model...</option>
                  {models
                    .filter((m) => m.device === device)
                    .map((m) => (
                      <option key={m.name} value={m.name}>{m.name}</option>
                    ))}
                </select>
              </div>
              <div className="flex items-end">
                <button
                  onClick={handleRunInferenceBench}
                  disabled={!selectedModel || benchJob}
                  className="w-full py-2 bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
                >
                  {benchJob && <Spinner size="w-4 h-4" />}
                  {benchJob ? "Running..." : "Run Benchmark"}
                </button>
              </div>
            </div>

            {benchResult && (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {benchResult.map((r, i) => (
                    <div key={i} className="bg-gray-800 rounded-xl p-4 border border-gray-700">
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="text-sm font-medium text-gray-200">{r.label}</h4>
                        {r.error ? (
                          <Badge label="failed" variant="error" />
                        ) : (
                          <Badge label="ok" variant="completed" />
                        )}
                      </div>
                      {r.error ? (
                        <p className="text-xs text-red-400">{r.error}</p>
                      ) : (
                        <div className="grid grid-cols-2 gap-2">
                          <div className="text-center">
                            <p className="text-xl font-bold text-cyan-400">{r.tok_per_s?.toFixed(1)}</p>
                            <p className="text-xs text-gray-500">tok/s</p>
                          </div>
                          <div className="text-center">
                            <p className="text-xl font-bold text-cyan-400">{r.avg_time?.toFixed(3)}s</p>
                            <p className="text-xs text-gray-500">avg time</p>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {/* Comparison bar */}
                {benchResult.filter((r) => !r.error).length >= 2 && (() => {
                  const rsr = benchResult.find((r) => r.label === "RSR");
                  const hf = benchResult.find((r) => r.label !== "RSR" && !r.error);
                  if (!rsr || !hf) return null;
                  const speedup = hf.avg_time / rsr.avg_time;
                  return (
                    <div className="bg-gray-800 rounded-xl p-4 border border-gray-700 text-center">
                      <p className="text-sm text-gray-400 mb-1">RSR Speedup</p>
                      <p className={`text-3xl font-bold ${speedup >= 1 ? "text-green-400" : "text-red-400"}`}>
                        {speedup.toFixed(2)}x
                      </p>
                      <p className="text-xs text-gray-500 mt-1">vs {hf.label}</p>
                    </div>
                  );
                })()}
              </div>
            )}
          </div>
        </Card>
      )}

      {/* Reports tab */}
      {tab === "reports" && (
        <Card title="Available Report Files">
          {reports.length === 0 ? (
            <p className="text-gray-500 text-sm">No reports found.</p>
          ) : (
            <div className="overflow-auto rounded-lg border border-gray-800">
              <table className="w-full text-xs">
                <thead className="bg-gray-800">
                  <tr>
                    <th className="text-left px-3 py-2 text-gray-400">Category</th>
                    <th className="text-left px-3 py-2 text-gray-400">File</th>
                    <th className="text-right px-3 py-2 text-gray-400">Type</th>
                    <th className="text-right px-3 py-2 text-gray-400">Size</th>
                  </tr>
                </thead>
                <tbody>
                  {reports.map((r, i) => (
                    <tr key={i} className="border-t border-gray-800 hover:bg-gray-800/50">
                      <td className="px-3 py-1.5 text-gray-400">{r.category}</td>
                      <td className="px-3 py-1.5 text-gray-200 font-mono">{r.filename}</td>
                      <td className="text-right px-3 py-1.5">
                        <Badge label={r.type} />
                      </td>
                      <td className="text-right px-3 py-1.5 text-gray-400">{r.size_kb} KB</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
