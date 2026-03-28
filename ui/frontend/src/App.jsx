import { Routes, Route, NavLink } from "react-router-dom";
import { useEffect, useState } from "react";
import { getSystemInfo } from "./api";
import ModelsPage from "./pages/ModelsPage";
import PreprocessPage from "./pages/PreprocessPage";
import InferencePage from "./pages/InferencePage";
import MultipliersPage from "./pages/MultipliersPage";
import BenchmarksPage from "./pages/BenchmarksPage";
import DashboardPage from "./pages/DashboardPage";

const NAV = [
  { to: "/", label: "Dashboard", icon: "M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0h4" },
  { to: "/models", label: "Models", icon: "M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" },
  { to: "/preprocess", label: "Preprocess", icon: "M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" },
  { to: "/inference", label: "Inference", icon: "M13 10V3L4 14h7v7l9-11h-7z" },
  { to: "/multipliers", label: "Multipliers", icon: "M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" },
  { to: "/benchmarks", label: "Benchmarks", icon: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" },
];

function SidebarIcon({ d }) {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d={d} />
    </svg>
  );
}

export default function App() {
  const [sysInfo, setSysInfo] = useState(null);

  useEffect(() => {
    getSystemInfo().then(setSysInfo).catch(() => {});
  }, []);

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <aside className="w-56 bg-gray-900 border-r border-gray-800 flex flex-col">
        <div className="p-4 border-b border-gray-800">
          <h1 className="text-lg font-bold text-cyan-400 tracking-tight">RSR-core</h1>
          <p className="text-xs text-gray-500 mt-0.5">Dashboard</p>
        </div>
        <nav className="flex-1 p-2 space-y-0.5">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                  isActive
                    ? "bg-cyan-500/10 text-cyan-400"
                    : "text-gray-400 hover:text-gray-200 hover:bg-gray-800"
                }`
              }
            >
              <SidebarIcon d={n.icon} />
              {n.label}
            </NavLink>
          ))}
        </nav>
        {sysInfo && (
          <div className="p-3 border-t border-gray-800 text-xs text-gray-500 space-y-0.5">
            <p>PyTorch {sysInfo.torch_version}</p>
            <p>{sysInfo.cuda_available ? sysInfo.cuda_device : "CPU only"}</p>
          </div>
        )}
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto p-6">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/preprocess" element={<PreprocessPage />} />
          <Route path="/inference" element={<InferencePage />} />
          <Route path="/multipliers" element={<MultipliersPage />} />
          <Route path="/benchmarks" element={<BenchmarksPage />} />
        </Routes>
      </main>
    </div>
  );
}
