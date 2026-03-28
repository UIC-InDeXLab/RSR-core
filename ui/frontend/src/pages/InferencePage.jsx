import { useState, useEffect } from "react";
import { listModels, runInference } from "../api";
import Card from "../components/Card";
import Badge from "../components/Badge";
import Spinner from "../components/Spinner";

export default function InferencePage() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [prompt, setPrompt] = useState("Hello, how are you?");
  const [maxTokens, setMaxTokens] = useState(64);
  const [useChatTemplate, setUseChatTemplate] = useState(true);
  const [backend, setBackend] = useState("rsr");
  const [dtype, setDtype] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);
  const [errorTrace, setErrorTrace] = useState(null);

  useEffect(() => {
    listModels().then(setModels).catch(() => {});
  }, []);

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    setErrorTrace(null);
    setResult(null);
    try {
      const res = await runInference({
        model_dir: selectedModel,
        prompt,
        max_new_tokens: maxTokens,
        use_chat_template: useChatTemplate,
        backend,
        dtype: dtype || undefined,
      });
      setResult(res);
      setHistory((prev) => [{ ...res, prompt, timestamp: new Date().toISOString() }, ...prev]);
    } catch (e) {
      setError(e.message);
      setErrorTrace(e.traceback || null);
    }
    setLoading(false);
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Inference</h1>
        <p className="text-gray-500 text-sm mt-1">Generate text with RSR-accelerated models</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Config */}
        <div className="lg:col-span-2 space-y-4">
          <Card title="Configuration">
            <div className="space-y-4">
              <div>
                <label className="text-xs text-gray-500 mb-1 block">Model</label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-cyan-500"
                >
                  <option value="">Select a model...</option>
                  {models.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.name} ({m.size_mb} MB)
                    </option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-gray-500 mb-1 block">Backend</label>
                  <select
                    value={backend}
                    onChange={(e) => setBackend(e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-cyan-500"
                  >
                    <option value="rsr">RSR</option>
                    <option value="hf">HuggingFace (baseline)</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs text-gray-500 mb-1 block">Max Tokens</label>
                  <input
                    type="number"
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(Number(e.target.value))}
                    min={1}
                    max={512}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-cyan-500"
                  />
                </div>
              </div>

              <div>
                <label className="text-xs text-gray-500 mb-1 block">Dtype (optional)</label>
                <select
                  value={dtype}
                  onChange={(e) => setDtype(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-cyan-500"
                >
                  <option value="">Default</option>
                  <option value="float32">float32</option>
                  <option value="bfloat16">bfloat16</option>
                  <option value="float16">float16</option>
                </select>
              </div>

              <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useChatTemplate}
                  onChange={(e) => setUseChatTemplate(e.target.checked)}
                  className="rounded border-gray-600"
                />
                Use chat template
              </label>
            </div>
          </Card>
        </div>

        {/* Prompt & Output */}
        <div className="lg:col-span-3 space-y-4">
          <Card title="Prompt">
            <div className="space-y-3">
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={3}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-cyan-500 resize-none"
                placeholder="Enter your prompt..."
              />
              <button
                onClick={handleRun}
                disabled={!selectedModel || loading}
                className="w-full py-3 bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg text-sm font-semibold transition-colors flex items-center justify-center gap-2"
              >
                {loading && <Spinner size="w-4 h-4" />}
                {loading ? "Generating..." : "Generate"}
              </button>
            </div>
          </Card>

          {error && (
            <Card title="Error">
              <div className="space-y-3">
                <p className="text-red-400 text-sm">{error}</p>
                {errorTrace && (
                  <pre className="bg-red-950/30 border border-red-500/20 rounded-lg p-3 text-xs text-red-300 overflow-auto max-h-64 whitespace-pre-wrap">
                    {errorTrace}
                  </pre>
                )}
              </div>
            </Card>
          )}

          {result && (
            <Card title="Output">
              <div className="space-y-4">
                <p className="text-gray-200 text-sm whitespace-pre-wrap leading-relaxed">{result.text}</p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-3 border-t border-gray-800">
                  <div className="bg-gray-800 rounded-lg p-3 text-center">
                    <p className="text-lg font-bold text-cyan-400">{result.tok_per_sec}</p>
                    <p className="text-xs text-gray-500">tok/s</p>
                  </div>
                  <div className="bg-gray-800 rounded-lg p-3 text-center">
                    <p className="text-lg font-bold text-cyan-400">{result.n_tokens}</p>
                    <p className="text-xs text-gray-500">tokens</p>
                  </div>
                  <div className="bg-gray-800 rounded-lg p-3 text-center">
                    <p className="text-lg font-bold text-cyan-400">{result.inference_time_s}s</p>
                    <p className="text-xs text-gray-500">inference</p>
                  </div>
                  <div className="bg-gray-800 rounded-lg p-3 text-center">
                    <p className="text-lg font-bold text-cyan-400">{result.load_time_s}s</p>
                    <p className="text-xs text-gray-500">load time</p>
                  </div>
                </div>
              </div>
            </Card>
          )}

          {/* History */}
          {history.length > 1 && (
            <Card title="Run History">
              <div className="space-y-2 max-h-48 overflow-auto">
                {history.map((h, i) => (
                  <div key={i} className="flex items-center justify-between px-3 py-2 bg-gray-800 rounded-lg text-xs">
                    <div className="flex items-center gap-2">
                      <Badge label={h.backend === "rsr" ? "RSR" : "HF"} variant={h.backend === "rsr" ? "cuda" : "default"} />
                      <span className="text-gray-400 truncate max-w-40">{h.prompt}</span>
                    </div>
                    <div className="flex items-center gap-4 text-gray-400">
                      <span className="text-cyan-400 font-medium">{h.tok_per_sec} tok/s</span>
                      <span>{h.inference_time_s}s</span>
                      <span className="text-gray-500">{h.device}</span>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
