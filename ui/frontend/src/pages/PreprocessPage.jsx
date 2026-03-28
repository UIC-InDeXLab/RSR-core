import { useState, useEffect, useRef } from "react";
import { searchHFModels, startPreprocess, getPreprocessStatus, getBestKAvailability } from "../api";
import Card from "../components/Card";
import Badge from "../components/Badge";
import Spinner from "../components/Spinner";

export default function PreprocessPage() {
  const [query, setQuery] = useState("1.58");
  const [hfModels, setHfModels] = useState([]);
  const [searching, setSearching] = useState(false);
  const [selectedModel, setSelectedModel] = useState("");
  const [k, setK] = useState(8);
  const [version, setVersion] = useState("adaptive");
  const [device, setDevice] = useState("cpu");
  const [trustRemoteCode, setTrustRemoteCode] = useState(false);
  const [useBestK, setUseBestK] = useState(false);
  const [bestKData, setBestKData] = useState(null); // { cpu: {...}, cuda: {...} }
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const pollRef = useRef(null);

  // Load best-k availability on mount
  useEffect(() => {
    getBestKAvailability().then(setBestKData).catch(() => {});
  }, []);

  const bestKForDevice = bestKData?.[device];

  const search = async () => {
    setSearching(true);
    try {
      const results = await searchHFModels(query);
      setHfModels(results);
    } catch (e) {
      console.error(e);
    }
    setSearching(false);
  };

  useEffect(() => {
    search();
  }, []);

  // Poll job status
  useEffect(() => {
    if (!jobId) return;
    pollRef.current = setInterval(async () => {
      try {
        const status = await getPreprocessStatus(jobId);
        setJobStatus(status);
        if (status.status !== "running") {
          clearInterval(pollRef.current);
        }
      } catch {
        clearInterval(pollRef.current);
      }
    }, 2000);
    return () => clearInterval(pollRef.current);
  }, [jobId]);

  const handleStart = async () => {
    try {
      const res = await startPreprocess({
        model: selectedModel,
        k,
        version,
        device,
        trust_remote_code: trustRemoteCode,
        use_best_k: useBestK,
      });
      setJobId(res.job_id);
      setJobStatus({ status: "running", progress: "Starting..." });
    } catch (e) {
      alert(e.message);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Preprocess Model</h1>
        <p className="text-gray-500 text-sm mt-1">
          Search HuggingFace for ternary models and apply RSR preprocessing
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Search panel */}
        <Card title="1. Select Model">
          <div className="space-y-4">
            <div className="flex gap-2">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && search()}
                placeholder="Search HuggingFace models..."
                className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-cyan-500"
              />
              <button
                onClick={search}
                disabled={searching}
                className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-700 rounded-lg text-sm font-medium transition-colors"
              >
                {searching ? "..." : "Search"}
              </button>
            </div>

            {/* Manual entry */}
            <div>
              <label className="text-xs text-gray-500">Or enter model ID directly:</label>
              <input
                type="text"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                placeholder="microsoft/bitnet-b1.58-2B-4T"
                className="w-full mt-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-cyan-500"
              />
            </div>

            {/* Results */}
            <div className="max-h-72 overflow-auto space-y-1">
              {hfModels.map((m) => (
                <button
                  key={m.id}
                  onClick={() => setSelectedModel(m.id)}
                  className={`w-full text-left p-3 rounded-lg border transition-colors ${
                    selectedModel === m.id
                      ? "bg-cyan-500/10 border-cyan-500/30"
                      : "bg-gray-800 border-gray-700/50 hover:border-gray-600"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-200 font-medium truncate">{m.id}</span>
                    <span className="text-xs text-gray-500 shrink-0 ml-2">
                      {m.downloads?.toLocaleString()} dl
                    </span>
                  </div>
                  {m.pipeline_tag && (
                    <span className="text-xs text-gray-500">{m.pipeline_tag}</span>
                  )}
                </button>
              ))}
            </div>
          </div>
        </Card>

        {/* Config panel */}
        <div className="space-y-6">
          <Card title="2. Configure">
            <div className="space-y-4">
              <div>
                <label className="text-xs text-gray-500 mb-1 block">Device</label>
                <select
                  value={device}
                  onChange={(e) => setDevice(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-cyan-500"
                >
                  <option value="cpu">CPU</option>
                  <option value="cuda">CUDA</option>
                </select>
              </div>

              {/* Block height (k) */}
              <div className="space-y-2">
                {bestKForDevice ? (
                  <>
                    <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={useBestK}
                        onChange={(e) => setUseBestK(e.target.checked)}
                        className="rounded border-gray-600"
                      />
                      Use best k from benchmark
                    </label>
                    {useBestK ? (
                      <div className="bg-gray-800 rounded-lg p-3 space-y-2">
                        <p className="text-xs text-gray-500">
                          Optimal k per layer shape from benchmark. Fallback k={k} for unlisted shapes.
                        </p>
                        <div className="max-h-36 overflow-auto rounded border border-gray-700">
                          <table className="w-full text-xs">
                            <thead className="sticky top-0 bg-gray-900">
                              <tr>
                                <th className="text-left px-2 py-1 text-gray-500">Shape</th>
                                <th className="text-right px-2 py-1 text-gray-500">k</th>
                                <th className="text-right px-2 py-1 text-gray-500">RSR (ms)</th>
                                <th className="text-right px-2 py-1 text-gray-500">vs FP32</th>
                              </tr>
                            </thead>
                            <tbody>
                              {Object.entries(bestKForDevice)
                                .sort(([a], [b]) => a.localeCompare(b))
                                .map(([shape, info]) => (
                                  <tr key={shape} className="border-t border-gray-700">
                                    <td className="px-2 py-1 text-gray-300 font-mono">{shape}</td>
                                    <td className="text-right px-2 py-1 text-cyan-400 font-medium">{info.k}</td>
                                    <td className="text-right px-2 py-1 text-gray-400">{info.rsr_ms?.toFixed(3)}</td>
                                    <td className="text-right px-2 py-1 text-green-400">{info.speedup_vs_fp32?.toFixed(1)}x</td>
                                  </tr>
                                ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    ) : (
                      <div>
                        <label className="text-xs text-gray-500 mb-1 block">Block height (k)</label>
                        <select
                          value={k}
                          onChange={(e) => setK(Number(e.target.value))}
                          className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-cyan-500"
                        >
                          {[2, 4, 6, 8, 10, 12, 16].map((v) => (
                            <option key={v} value={v}>k = {v}</option>
                          ))}
                        </select>
                      </div>
                    )}
                  </>
                ) : (
                  <div>
                    <label className="text-xs text-gray-500 mb-1 block">Block height (k)</label>
                    <select
                      value={k}
                      onChange={(e) => setK(Number(e.target.value))}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-cyan-500"
                    >
                      {[2, 4, 6, 8, 10, 12, 16].map((v) => (
                        <option key={v} value={v}>k = {v}</option>
                      ))}
                    </select>
                    <p className="text-xs text-gray-600 mt-1">No best-k report for {device}. Run the benchmark to enable per-shape selection.</p>
                  </div>
                )}
              </div>

              <div>
                <label className="text-xs text-gray-500 mb-1 block">RSR Version</label>
                <input
                  type="text"
                  value={version}
                  onChange={(e) => setVersion(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-cyan-500"
                />
              </div>

              <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
                <input
                  type="checkbox"
                  checked={trustRemoteCode}
                  onChange={(e) => setTrustRemoteCode(e.target.checked)}
                  className="rounded border-gray-600"
                />
                Trust remote code
              </label>

              <button
                onClick={handleStart}
                disabled={!selectedModel || (jobStatus?.status === "running")}
                className="w-full py-3 bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg text-sm font-semibold transition-colors"
              >
                {jobStatus?.status === "running" ? "Processing..." : "Start Preprocessing"}
              </button>
            </div>
          </Card>

          {/* Status */}
          {jobStatus && (
            <Card title="3. Status">
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  {jobStatus.status === "running" && <Spinner />}
                  <Badge
                    label={jobStatus.status}
                    variant={jobStatus.status}
                  />
                </div>
                <p className="text-sm text-gray-400">{jobStatus.progress}</p>

                {/* Progress bar */}
                {jobStatus.status === "running" && jobStatus.total > 0 && (
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>
                        {jobStatus.stage === "loading" && "Loading model..."}
                        {jobStatus.stage === "preprocessing" && `Layer ${jobStatus.current + 1} / ${jobStatus.total}`}
                        {jobStatus.stage === "saving" && "Saving..."}
                      </span>
                      <span>
                        {jobStatus.stage === "preprocessing"
                          ? `${Math.round(((jobStatus.current + 1) / jobStatus.total) * 100)}%`
                          : jobStatus.stage === "saving" ? "100%" : ""}
                      </span>
                    </div>
                    <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-cyan-500 rounded-full transition-all duration-300"
                        style={{
                          width: jobStatus.stage === "loading"
                            ? "5%"
                            : jobStatus.stage === "saving"
                              ? "100%"
                              : `${Math.round(((jobStatus.current + 1) / jobStatus.total) * 100)}%`,
                        }}
                      />
                    </div>
                  </div>
                )}

                {jobStatus.status === "completed" && (
                  <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div className="h-full bg-green-500 rounded-full w-full" />
                  </div>
                )}

                {jobStatus.status === "error" && jobStatus.traceback && (
                  <pre className="bg-red-950/30 border border-red-500/20 rounded-lg p-3 text-xs text-red-300 overflow-auto max-h-64 whitespace-pre-wrap">
                    {jobStatus.traceback}
                  </pre>
                )}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
