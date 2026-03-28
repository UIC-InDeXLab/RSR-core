const COLORS = {
  cpu: "bg-blue-500/10 text-blue-400 border-blue-500/20",
  cuda: "bg-green-500/10 text-green-400 border-green-500/20",
  running: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
  completed: "bg-green-500/10 text-green-400 border-green-500/20",
  error: "bg-red-500/10 text-red-400 border-red-500/20",
  default: "bg-gray-500/10 text-gray-400 border-gray-500/20",
};

export default function Badge({ label, variant = "default" }) {
  const cls = COLORS[variant] || COLORS.default;
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium border ${cls}`}>
      {label}
    </span>
  );
}
