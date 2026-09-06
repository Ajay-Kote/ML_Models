import { useEffect, useState } from "react";
import { Activity, Bell, ChevronRight, ClockIcon, FileText, LayoutDashboard, Search, Settings, Shield, Upload } from "lucide-react";
import type { ComponentType, CSSProperties } from "react";
import type { PredictResponse } from "./api";
import Dashboard from "./components/Dashboard";
import Submit from "./components/Submit";
import CaseDetail from "./components/CaseDetail";
import ModuleHealth from "./components/ModuleHealth";
import SettingsScreen from "./components/Settings";

type Screen = "dashboard" | "submit" | "case-detail" | "module-health" | "reports" | "settings";
const navItems: { id: Screen; label: string; icon: ComponentType<{ size?: number; style?: CSSProperties }> }[] = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "submit", label: "Submit", icon: Upload },
  { id: "case-detail", label: "Case History", icon: ClockIcon },
  { id: "module-health", label: "Module Health", icon: Activity },
  { id: "reports", label: "Reports", icon: FileText },
  { id: "settings", label: "Settings", icon: Settings },
];

const ANALYSIS_STORAGE_KEY = "adaptive-risk-fusion:last-analysis";
type StoredAnalysis = { result: PredictResponse; caseNumber: number };

function restoreAnalysis(): StoredAnalysis | null {
  try {
    const saved = localStorage.getItem(ANALYSIS_STORAGE_KEY);
    return saved ? JSON.parse(saved) as StoredAnalysis : null;
  } catch {
    return null;
  }
}

function TopBar() {
  return <header className="flex items-center justify-between px-5 h-14 flex-shrink-0" style={{ borderBottom: "1px solid var(--border)", background: "var(--card)" }}><div className="flex items-center gap-2 flex-1 max-w-sm"><Search size={14} style={{ color: "var(--muted-foreground)" }} /><input placeholder="Search cases, verdicts, analysts..." className="flex-1 text-sm bg-transparent outline-none" style={{ color: "var(--foreground)" }} /></div><div className="flex items-center gap-2"><button className="relative w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: "var(--secondary)", color: "var(--muted-foreground)" }}><Bell size={15} /></button><div className="flex items-center gap-2 pl-2 ml-1 border-l" style={{ borderColor: "var(--border)" }}><div className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold" style={{ background: "var(--primary)", color: "white" }}>AC</div><div className="hidden sm:block"><p className="text-xs font-semibold" style={{ color: "var(--foreground)" }}>Analyst</p><p className="text-xs" style={{ color: "var(--muted-foreground)" }}>Backend connected</p></div></div></div></header>;
}

export default function App() {
  const [screen, setScreen] = useState<Screen>("dashboard");
  const [collapsed, setCollapsed] = useState(false);
  const [storedAnalysis] = useState(restoreAnalysis);
  const [lastResult, setLastResult] = useState<PredictResponse | null>(storedAnalysis?.result ?? null);
  const [caseNumber, setCaseNumber] = useState(storedAnalysis?.caseNumber ?? 1);
  const caseId = `ARF-${caseNumber.toString().padStart(4, "0")}`;

  useEffect(() => {
    if (!lastResult) return;
    try {
      localStorage.setItem(ANALYSIS_STORAGE_KEY, JSON.stringify({ result: lastResult, caseNumber }));
    } catch {
      // Storage can be unavailable in private or restricted browser contexts.
    }
  }, [lastResult, caseNumber]);
  return <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}><aside className="flex flex-col flex-shrink-0 transition-all duration-300" style={{ width: collapsed ? 56 : 200, background: "var(--sidebar-bg)", borderRight: "1px solid var(--border)" }}><div className="flex items-center gap-2.5 px-3 py-4 border-b" style={{ borderColor: "var(--border)" }}><div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0" style={{ background: "var(--sidebar-active)", color: "white" }}><Shield size={16} /></div>{!collapsed && <div className="overflow-hidden"><p className="text-xs font-bold" style={{ color: "var(--foreground)" }}>Adaptive Risk</p><p className="text-xs font-bold" style={{ color: "var(--sidebar-active)" }}>Fusion</p></div>}</div><nav className="flex flex-col gap-0.5 p-2 flex-1 overflow-y-auto scrollbar-hide">{navItems.map(({ id, label, icon: Icon }) => <button key={id} onClick={() => setScreen(id)} className="flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-sm font-medium text-left" style={{ background: screen === id ? "var(--sidebar-active-bg)" : "transparent", color: screen === id ? "var(--sidebar-active)" : "var(--sidebar-fg)" }} title={collapsed ? label : undefined}><Icon size={16} />{!collapsed && <span className="truncate">{label}</span>}</button>)}</nav><button onClick={() => setCollapsed(value => !value)} className="flex items-center justify-center p-3 border-t" style={{ borderColor: "var(--border)", color: "var(--sidebar-fg)" }}><ChevronRight size={14} style={{ transform: collapsed ? "none" : "rotate(180deg)" }} /></button></aside><div className="flex flex-col flex-1 min-w-0" style={{ background: "var(--background)" }}><TopBar /><div className="flex items-center justify-between px-6 py-3 border-b" style={{ borderColor: "var(--border)", background: "var(--card)" }}><div className="flex items-center gap-2">{(() => { const nav = navItems.find(item => item.id === screen)!; return <><nav.icon size={15} style={{ color: "var(--primary)" }} /><h1 className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>{nav.label}</h1></>; })()}</div>{screen === "dashboard" && <span className="font-mono text-xs" style={{ color: "var(--muted-foreground)" }}>Live backend data</span>}</div><main className="flex-1 overflow-y-auto scrollbar-hide">{screen === "dashboard" && <Dashboard result={lastResult} caseId={caseId} onCaseClick={() => setScreen("case-detail")} />}{screen === "submit" && <Submit onAnalyze={result => { setLastResult(result); setCaseNumber(value => value + 1); setScreen("case-detail"); }} />}{screen === "case-detail" && <CaseDetail result={lastResult} caseId={caseId} />}{screen === "module-health" && <ModuleHealth />}{screen === "settings" && <SettingsScreen />}{screen === "reports" && <div className="flex flex-col items-center justify-center h-full gap-3" style={{ color: "var(--muted-foreground)" }}><FileText size={40} /><p className="text-sm font-medium">Reports - coming soon</p></div>}</main></div></div>;
}
