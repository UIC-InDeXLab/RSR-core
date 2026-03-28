import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { listModels, listMultipliers, getSystemInfo } from "../api";
import Card from "../components/Card";

function StatCard({ label, value, sub, to, loading }) {
  const inner = (
    <div className="text-center">
      {loading ? (
        <div className="flex justify-center">
          <div className="h-8 w-16 rounded bg-gray-700 animate-pulse" />
        </div>
      ) : (
        <p className="text-3xl font-bold text-cyan-400">{value}</p>
      )}
      <p className="text-sm text-gray-400 mt-1">{label}</p>
      {loading ? (
        <div className="flex justify-center mt-0.5">
          <div className="h-3 w-24 rounded bg-gray-700 animate-pulse" />
        </div>
      ) : (
        sub && <p className="text-xs text-gray-600 mt-0.5">{sub}</p>
      )}
    </div>
  );
  if (to) return <Link to={to} className="block hover:scale-105 transition-transform">{inner}</Link>;
  return inner;
}

export default function DashboardPage() {
  const [models, setModels] = useState([]);
  const [multipliers, setMultipliers] = useState([]);
  const [sys, setSys] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      listModels().then(setModels).catch(() => {}),
      listMultipliers().then(setMultipliers).catch(() => {}),
      getSystemInfo().then(setSys).catch(() => {}),
    ]).finally(() => setLoading(false));
  }, []);

  const cpuModels = models.filter((m) => m.device === "cpu").length;
  const cudaModels = models.filter((m) => m.device === "cuda").length;
  const totalSize = models.reduce((s, m) => s + m.size_mb, 0);

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="text-gray-500 text-sm mt-1">RSR-core project overview</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card><StatCard label="Preprocessed Models" value={models.length} sub={`${cpuModels} CPU / ${cudaModels} CUDA`} to="/models" loading={loading} /></Card>
        <Card><StatCard label="Multipliers" value={multipliers.length} to="/multipliers" loading={loading} /></Card>
        <Card><StatCard label="Total Size" value={`${(totalSize / 1024).toFixed(1)} GB`} sub="preprocessed data" loading={loading} /></Card>
        <Card><StatCard label="CUDA" value={sys?.cuda_available ? "Available" : "N/A"} sub={sys?.cuda_device || "CPU only"} loading={loading} /></Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card title="Quick Actions">
          <div className="space-y-2">
            <Link to="/preprocess" className="block w-full text-left px-4 py-3 rounded-lg bg-gray-800 hover:bg-gray-750 hover:bg-cyan-500/5 border border-gray-700 transition-colors">
              <span className="text-sm font-medium text-gray-200">Preprocess a Model</span>
              <span className="block text-xs text-gray-500 mt-0.5">Search HuggingFace and apply RSR preprocessing</span>
            </Link>
            <Link to="/inference" className="block w-full text-left px-4 py-3 rounded-lg bg-gray-800 hover:bg-cyan-500/5 border border-gray-700 transition-colors">
              <span className="text-sm font-medium text-gray-200">Run Inference</span>
              <span className="block text-xs text-gray-500 mt-0.5">Generate text with RSR-accelerated models</span>
            </Link>
            <Link to="/benchmarks" className="block w-full text-left px-4 py-3 rounded-lg bg-gray-800 hover:bg-cyan-500/5 border border-gray-700 transition-colors">
              <span className="text-sm font-medium text-gray-200">View Benchmarks</span>
              <span className="block text-xs text-gray-500 mt-0.5">Compare RSR performance vs baselines</span>
            </Link>
          </div>
        </Card>

        <Card title="Preprocessed Models">
          {loading ? (
            <div className="space-y-2">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="flex items-center justify-between px-3 py-2 bg-gray-800 rounded-lg">
                  <div className="space-y-1.5">
                    <div className="h-4 w-32 rounded bg-gray-700 animate-pulse" />
                    <div className="h-3 w-20 rounded bg-gray-700 animate-pulse" />
                  </div>
                  <div className="h-5 w-10 rounded bg-gray-700 animate-pulse" />
                </div>
              ))}
            </div>
          ) : models.length === 0 ? (
            <p className="text-gray-500 text-sm">No preprocessed models yet. <Link to="/preprocess" className="text-cyan-400 hover:underline">Preprocess one</Link>.</p>
          ) : (
            <div className="space-y-2 max-h-64 overflow-auto">
              {models.map((m) => (
                <div key={m.name} className="flex items-center justify-between px-3 py-2 bg-gray-800 rounded-lg">
                  <div>
                    <p className="text-sm text-gray-200">{m.name}</p>
                    <p className="text-xs text-gray-500">{m.num_layers} layers, k={m.k}</p>
                  </div>
                  <span className={`text-xs px-2 py-0.5 rounded ${m.device === "cuda" ? "bg-green-500/10 text-green-400" : "bg-blue-500/10 text-blue-400"}`}>
                    {m.device}
                  </span>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
