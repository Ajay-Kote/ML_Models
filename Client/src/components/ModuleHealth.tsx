import { useEffect, useState } from "react";
import { Activity, CheckCircle, AlertCircle } from "lucide-react";
import { ApiError, getModelHealth, type ModelHealthResponse } from "../api";
import { EmptyState, MODULE_META } from "./shared";

export default function ModuleHealth() {
  const [health, setHealth] = useState<ModelHealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getModelHealth().then(setHealth).catch((caught) => {
      setError(caught instanceof ApiError ? caught.message : "Could not load model health.");
    });
  }, []);

  if (error) return <div className="p-6"><EmptyState><AlertCircle size={32} className="mx-auto mb-3" />{error}</EmptyState></div>;
  if (!health) return <div className="p-6"><EmptyState><Activity size={32} className="mx-auto mb-3 animate-pulse" />Loading model health...</EmptyState></div>;

  return <div className="p-6 flex flex-col gap-5"><div><h2 className="text-xl font-bold" style={{ color: "var(--foreground)" }}>Module Health</h2><p className="text-sm mt-0.5" style={{ color: "var(--muted-foreground)" }}>Live status from the existing fusion model and API runtime.</p></div><div className="rounded-xl p-5 flex items-center gap-6" style={{ background: "var(--primary)", color: "white" }}><div><p className="text-xs font-semibold uppercase tracking-widest opacity-80">Fusion Model</p><p className="font-mono text-2xl font-bold">{health.model_loaded ? "Loaded" : "Unavailable"}</p><p className="text-xs opacity-80">{health.model_file}</p></div><div className="h-12 w-px" style={{ background: "rgba(255,255,255,0.3)" }} /><div><p className="text-xs opacity-80">Predictions</p><p className="font-mono font-bold">{health.prediction_count}</p></div><div><p className="text-xs opacity-80">Avg latency</p><p className="font-mono font-bold">{health.average_latency_ms === null ? "—" : `${health.average_latency_ms} ms`}</p></div></div><div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">{health.modules.map(module => { const meta = MODULE_META[module.id]; return <div key={module.id} className="rounded-xl p-4 flex flex-col gap-3" style={{ background: "var(--card)", border: "1px solid var(--border)" }}><div className="flex items-center gap-2"><div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: `${meta?.color ?? "var(--primary)"}20`, color: meta?.color ?? "var(--primary)" }}>{meta ? <meta.icon size={16} /> : <Activity size={16} />}</div><div><p className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>{meta?.label ?? module.id}</p><span className="flex items-center gap-1 text-xs" style={{ color: "var(--risk-safe)" }}><CheckCircle size={11} />Configured</span></div></div><div className="border-t pt-3" style={{ borderColor: "var(--border)" }}><p className="text-xs" style={{ color: "var(--muted-foreground)" }}>Learned risk coefficient</p><p className="font-mono text-lg font-semibold" style={{ color: "var(--foreground)" }}>{module.learned_weight.toFixed(4)}</p></div></div>; })}</div><p className="text-xs" style={{ color: "var(--muted-foreground)" }}>Accuracy, F1, and ROC-AUC are unavailable because the current training script does not persist evaluation metrics. No values are fabricated.</p></div>;
}
