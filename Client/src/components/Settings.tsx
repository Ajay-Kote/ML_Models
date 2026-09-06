import { useCallback, useEffect, useState } from "react";
import { CheckCircle, Loader2, RefreshCw, Server, XCircle } from "lucide-react";
import { ApiError, getModelHealth, type ModelHealthResponse } from "../api";

export default function Settings() {
  const [health, setHealth] = useState<ModelHealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const checkConnection = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setHealth(await getModelHealth());
    } catch (caught) {
      setHealth(null);
      setError(caught instanceof ApiError ? caught.message : "Backend is unavailable.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkConnection();
  }, [checkConnection]);

  const connected = Boolean(health);
  return <div className="p-6 flex flex-col gap-5 max-w-3xl mx-auto"><div><h2 className="text-xl font-bold" style={{ color: "var(--foreground)" }}>Settings</h2><p className="text-sm mt-0.5" style={{ color: "var(--muted-foreground)" }}>Connection and runtime status for the Fusion backend.</p></div><div className="rounded-xl p-5" style={{ background: "var(--card)", border: "1px solid var(--border)" }}><div className="flex items-center justify-between gap-4"><div className="flex items-center gap-3"><div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ background: connected ? "var(--risk-safe-bg)" : "var(--risk-high-bg)", color: connected ? "var(--risk-safe)" : "var(--risk-high)" }}>{loading ? <Loader2 size={20} className="animate-spin" /> : connected ? <CheckCircle size={20} /> : <XCircle size={20} />}</div><div><p className="font-semibold" style={{ color: "var(--foreground)" }}>{loading ? "Checking backend..." : connected ? "Backend connected" : "Backend unavailable"}</p><p className="text-xs" style={{ color: "var(--muted-foreground)" }}>http://127.0.0.1:8000/model-health</p></div></div><button onClick={checkConnection} disabled={loading} className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium disabled:opacity-50" style={{ background: "var(--secondary)", color: "var(--foreground)", border: "1px solid var(--border)" }}><RefreshCw size={13} />Check again</button></div>{error && <p className="mt-4 text-xs" style={{ color: "var(--risk-high-text)" }}>{error}</p>}</div>{health && <div className="grid grid-cols-1 sm:grid-cols-3 gap-4"><div className="rounded-xl p-4" style={{ background: "var(--card)", border: "1px solid var(--border)" }}><Server size={16} style={{ color: "var(--primary)" }} /><p className="text-xs mt-3" style={{ color: "var(--muted-foreground)" }}>Fusion model</p><p className="font-mono text-sm font-semibold" style={{ color: "var(--foreground)" }}>{health.model_loaded ? "Loaded" : "Unavailable"}</p></div><div className="rounded-xl p-4" style={{ background: "var(--card)", border: "1px solid var(--border)" }}><p className="text-xs" style={{ color: "var(--muted-foreground)" }}>API status</p><p className="font-mono text-sm font-semibold capitalize" style={{ color: "var(--foreground)" }}>{health.status}</p></div><div className="rounded-xl p-4" style={{ background: "var(--card)", border: "1px solid var(--border)" }}><p className="text-xs" style={{ color: "var(--muted-foreground)" }}>Predictions handled</p><p className="font-mono text-sm font-semibold" style={{ color: "var(--foreground)" }}>{health.prediction_count}</p></div></div>}</div>;
}
