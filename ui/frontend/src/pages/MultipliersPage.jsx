import { useEffect, useState } from "react";
import { listMultipliers } from "../api";
import Card from "../components/Card";
import Badge from "../components/Badge";

export default function MultipliersPage() {
  const [multipliers, setMultipliers] = useState([]);
  const [filter, setFilter] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listMultipliers()
      .then(setMultipliers)
      .finally(() => setLoading(false));
  }, []);

  const filtered = multipliers.filter(
    (m) =>
      m.class_name.toLowerCase().includes(filter.toLowerCase()) ||
      m.module.toLowerCase().includes(filter.toLowerCase()) ||
      m.bit_width.includes(filter) ||
      m.platform.includes(filter)
  );

  const grouped = {};
  for (const m of filtered) {
    const key = `${m.bit_width} / ${m.platform}`;
    (grouped[key] ??= []).push(m);
  }

  const platformColor = (p) => {
    if (p === "cpu") return "cpu";
    if (p === "cuda") return "cuda";
    return "default";
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Multipliers</h1>
          <p className="text-gray-500 text-sm mt-1">
            All discovered multiplier implementations ({multipliers.length} total)
          </p>
        </div>
        <input
          type="text"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter..."
          className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-cyan-500 w-48"
        />
      </div>

      {loading ? (
        <p className="text-gray-500 text-sm">Loading...</p>
      ) : (
        Object.entries(grouped).map(([group, items]) => (
          <Card key={group} title={group}>
            <div className="space-y-1">
              {items.map((m, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between px-4 py-2.5 rounded-lg hover:bg-gray-800 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <Badge label={m.platform} variant={platformColor(m.platform)} />
                    <div>
                      <span className="text-sm font-medium text-gray-200">{m.class_name}</span>
                      <span className="text-xs text-gray-500 ml-2">({m.module})</span>
                    </div>
                  </div>
                  <span className="text-xs text-gray-600 font-mono">{m.file}</span>
                </div>
              ))}
            </div>
          </Card>
        ))
      )}

      {/* Version selection logic card */}
      <Card title="Kernel Selection Logic">
        <div className="space-y-4 text-sm text-gray-400">
          <div>
            <h4 className="text-gray-200 font-medium mb-1">1.58-bit (Ternary) CPU</h4>
            <ul className="list-disc list-inside space-y-1 text-xs">
              <li><span className="text-cyan-400">v3.3</span> (bitmask scatter): selected when <code className="text-gray-300">n_cols &ge; 4096</code> and <code className="text-gray-300">k &le; 16</code></li>
              <li><span className="text-cyan-400">v3.1</span> (direct scatter): selected otherwise</li>
              <li>Non-square multiplier auto-selects between v3.1 and v3.3</li>
            </ul>
          </div>
          <div>
            <h4 className="text-gray-200 font-medium mb-1">1.58-bit (Ternary) CUDA</h4>
            <ul className="list-disc list-inside space-y-1 text-xs">
              <li><span className="text-cyan-400">v2.0</span>: single CUDA implementation for ternary</li>
            </ul>
          </div>
          <div>
            <h4 className="text-gray-200 font-medium mb-1">1-bit (Binary) CUDA</h4>
            <ul className="list-disc list-inside space-y-1 text-xs">
              <li><span className="text-cyan-400">v5.7</span>: optimized for <code className="text-gray-300">k=8, n &le; 4096</code></li>
              <li><span className="text-cyan-400">v5.8</span>: optimized for <code className="text-gray-300">k=16, n &le; 8192</code></li>
              <li><span className="text-cyan-400">v5.9</span>: general-purpose fallback</li>
              <li><span className="text-cyan-400">v5.10</span>: dispatcher that auto-selects the best kernel</li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
}
