import { useEffect, useState } from "react";
import { listModels, getModelConfig, deleteModel } from "../api";
import Card from "../components/Card";
import Badge from "../components/Badge";

export default function ModelsPage() {
  const [models, setModels] = useState([]);
  const [selected, setSelected] = useState(null);
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    listModels()
      .then(setModels)
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const handleSelect = async (m) => {
    setSelected(m.name);
    try {
      const cfg = await getModelConfig(m.name);
      setConfig(cfg);
    } catch {
      setConfig(null);
    }
  };

  const handleDelete = async (name) => {
    if (!confirm(`Delete ${name}? This cannot be undone.`)) return;
    await deleteModel(name);
    setSelected(null);
    setConfig(null);
    load();
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Preprocessed Models</h1>
        <p className="text-gray-500 text-sm mt-1">Manage RSR-preprocessed model artifacts</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model list */}
        <div className="lg:col-span-1 space-y-2">
          {loading ? (
            <p className="text-gray-500 text-sm">Loading...</p>
          ) : models.length === 0 ? (
            <p className="text-gray-500 text-sm">No preprocessed models found.</p>
          ) : (
            models.map((m) => (
              <button
                key={m.name}
                onClick={() => handleSelect(m)}
                className={`w-full text-left p-4 rounded-xl border transition-colors ${
                  selected === m.name
                    ? "bg-cyan-500/10 border-cyan-500/30"
                    : "bg-gray-900 border-gray-800 hover:border-gray-700"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-200 truncate">{m.name}</span>
                  <Badge label={m.device} variant={m.device} />
                </div>
                <div className="mt-2 flex gap-4 text-xs text-gray-500">
                  <span>{m.num_layers} layers</span>
                  <span>{m.size_mb} MB</span>
                </div>
              </button>
            ))
          )}
        </div>

        {/* Detail panel */}
        <div className="lg:col-span-2">
          {selected && config ? (
            <Card
              title={selected}
              action={
                <button
                  onClick={() => handleDelete(selected)}
                  className="text-xs text-red-400 hover:text-red-300 px-2 py-1 rounded hover:bg-red-500/10 transition-colors"
                >
                  Delete
                </button>
              }
            >
              <div className="space-y-4">
                <div>
                  <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">RSR Config</h3>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div className="bg-gray-800 rounded-lg px-3 py-2">
                      <span className="text-gray-500">Model</span>
                      <p className="text-gray-200 truncate">{config.rsr_config.model_name}</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg px-3 py-2">
                      <span className="text-gray-500">Version</span>
                      <p className="text-gray-200">{config.rsr_config.version}</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg px-3 py-2">
                      <span className="text-gray-500">Layers</span>
                      <p className="text-gray-200">{Object.keys(config.rsr_config.layers || {}).length}</p>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Layer Details</h3>
                  <div className="max-h-80 overflow-auto rounded-lg border border-gray-800">
                    <table className="w-full text-xs">
                      <thead className="bg-gray-800 sticky top-0">
                        <tr>
                          <th className="text-left px-3 py-2 text-gray-400">Layer</th>
                          <th className="text-right px-3 py-2 text-gray-400">Rows</th>
                          <th className="text-right px-3 py-2 text-gray-400">Cols</th>
                          <th className="text-right px-3 py-2 text-gray-400">k</th>
                          <th className="text-right px-3 py-2 text-gray-400">Backend</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(config.rsr_config.layers || {}).map(([name, meta]) => (
                          <tr key={name} className="border-t border-gray-800 hover:bg-gray-800/50">
                            <td className="px-3 py-1.5 text-gray-300 font-mono truncate max-w-xs">{name}</td>
                            <td className="text-right px-3 py-1.5 text-gray-400">{meta.n_rows}</td>
                            <td className="text-right px-3 py-1.5 text-gray-400">{meta.n_cols}</td>
                            <td className="text-right px-3 py-1.5 text-cyan-400">{meta.k}</td>
                            <td className="text-right px-3 py-1.5 text-gray-400">{meta.backend}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {config.hf_config && config.hf_config.model_type && (
                  <div>
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">HF Model Config</h3>
                    <pre className="bg-gray-800 rounded-lg p-3 text-xs text-gray-400 overflow-auto max-h-48">
                      {JSON.stringify(config.hf_config, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </Card>
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-600 text-sm">
              Select a model to view details
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
