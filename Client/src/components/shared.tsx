import { CheckCircle, AlertTriangle, XCircle, Link2, MessageSquare, QrCode, Receipt, Mail } from "lucide-react";
import type { ComponentType, CSSProperties, ReactNode } from "react";
import type { ContributingModule } from "../api";

export type Verdict = "Safe" | "Suspicious" | "High Risk";

export const MODULE_META: Record<string, { label: string; icon: ComponentType<{ size?: number; style?: CSSProperties }>; color: string }> = {
  url: { label: "URL Analysis", icon: Link2, color: "#3B82F6" },
  sms: { label: "SMS Text", icon: MessageSquare, color: "#8B5CF6" },
  qr: { label: "QR Code", icon: QrCode, color: "#06B6D4" },
  image: { label: "Payment Screenshot", icon: Receipt, color: "#F59E0B" },
  email: { label: "Email Analysis", icon: Mail, color: "#EC4899" },
};

export function toVerdict(probability: number): Verdict {
  if (probability >= 0.5) return "High Risk";
  if (probability >= 0.3) return "Suspicious";
  return "Safe";
}

function verdictColor(verdict: Verdict) {
  if (verdict === "Safe") return { bg: "var(--risk-safe-bg)", label: "var(--risk-safe-text)" };
  if (verdict === "Suspicious") return { bg: "var(--risk-suspicious-bg)", label: "var(--risk-suspicious-text)" };
  return { bg: "var(--risk-high-bg)", label: "var(--risk-high-text)" };
}

export function VerdictBadge({ verdict, size = "sm" }: { verdict: Verdict; size?: "sm" | "lg" | "xl" }) {
  const colors = verdictColor(verdict);
  const Icon = verdict === "Safe" ? CheckCircle : verdict === "Suspicious" ? AlertTriangle : XCircle;
  const iconSize = size === "xl" ? 20 : size === "lg" ? 16 : 12;
  const padding = size === "xl" ? "px-4 py-2 text-base" : size === "lg" ? "px-3 py-1.5 text-sm" : "px-2 py-0.5 text-xs";
  return <span className={`inline-flex items-center gap-1.5 font-semibold rounded-md ${padding}`} style={{ background: colors.bg, color: colors.label }}><Icon size={iconSize} />{verdict}</span>;
}

const inputIcons: Record<string, ComponentType<{ size?: number }>> = { url: Link2, sms: MessageSquare, qr: QrCode, image: Receipt, payment: Receipt, email: Mail };
const inputColors: Record<string, string> = { url: "#3B82F6", sms: "#8B5CF6", qr: "#06B6D4", image: "#F59E0B", payment: "#F59E0B", email: "#EC4899" };

export function InputChip({ type }: { type: string }) {
  const Icon = inputIcons[type] || Link2;
  const color = inputColors[type] || "#6B7280";
  return <span className="inline-flex items-center justify-center w-6 h-6 rounded" style={{ background: `${color}22`, color }} title={type.toUpperCase()}><Icon size={12} /></span>;
}

export function RiskGauge({ score }: { score: number }) {
  const radius = 80;
  const centerX = 110;
  const centerY = 100;
  const strokeWidth = 18;
  const startAngle = 220;
  const totalArc = 260;
  const circumference = 2 * Math.PI * radius;
  const arcLength = (totalArc / 360) * circumference;
  const offset = circumference - (score / 100) * arcLength;
  const angle = (startAngle - (score / 100) * totalArc) * Math.PI / 180;
  const needleX = centerX + 65 * Math.cos(angle);
  const needleY = centerY - 65 * Math.sin(angle);
  const verdict: Verdict = score < 35 ? "Safe" : score < 65 ? "Suspicious" : "High Risk";
  const color = verdict === "Safe" ? "var(--risk-safe)" : verdict === "Suspicious" ? "var(--risk-suspicious)" : "var(--risk-high)";
  return <div className="flex flex-col items-center gap-4"><svg viewBox="0 0 220 140" className="w-64"><circle cx={centerX} cy={centerY} r={radius} fill="none" stroke="var(--border)" strokeWidth={strokeWidth} strokeDasharray={`${arcLength} ${circumference - arcLength}`} transform={`rotate(${180 - startAngle} ${centerX} ${centerY})`} /><circle cx={centerX} cy={centerY} r={radius} fill="none" stroke={color} strokeWidth={strokeWidth} strokeDasharray={`${arcLength} ${circumference - arcLength}`} strokeDashoffset={offset} strokeLinecap="round" transform={`rotate(${180 - startAngle} ${centerX} ${centerY})`} /><line x1={centerX} y1={centerY} x2={needleX} y2={needleY} stroke={color} strokeWidth={2.5} strokeLinecap="round" /><circle cx={centerX} cy={centerY} r={5} fill={color} /><text x={centerX - 82} y={centerY + 26} fill="var(--muted-foreground)" fontSize={10}>0</text><text x={centerX + 70} y={centerY + 26} fill="var(--muted-foreground)" fontSize={10}>100</text><text x={centerX} y={centerY - 8} textAnchor="middle" fill="var(--foreground)" fontSize={30} fontWeight={700}>{score}</text><text x={centerX} y={centerY + 10} textAnchor="middle" fill="var(--muted-foreground)" fontSize={10}>FUSED RISK SCORE</text></svg><VerdictBadge verdict={verdict} size="xl" /></div>;
}

export function ConfBar({ value, color }: { value: number; color: string }) {
  return <div className="flex items-center gap-2 flex-1"><div className="flex-1 h-1.5 rounded-full" style={{ background: "var(--muted)" }}><div className="h-full rounded-full" style={{ width: `${value}%`, background: color }} /></div><span className="font-mono text-xs w-8 text-right" style={{ color: "var(--muted-foreground)" }}>{value}%</span></div>;
}

export function FusionContribChart({ modules }: { modules: ContributingModule[] }) {
  return <div className="flex flex-col gap-2">{modules.map(module => { const meta = MODULE_META[module.module]; const weight = module.learned_weight; return <div key={module.module} className="flex items-center gap-3"><span className="text-xs w-24 flex-shrink-0 font-medium" style={{ color: "var(--foreground)" }}>{meta?.label ?? module.module}</span><div className="flex-1 h-4 rounded-full overflow-hidden" style={{ background: "var(--muted)" }}><div className="h-full rounded-full flex items-center px-1.5" style={{ width: `${weight * 100}%`, background: meta?.color ?? "var(--primary)" }}>{weight > 0.15 && <span className="font-mono text-xs text-white font-semibold">{(weight * 100).toFixed(0)}%</span>}</div></div><span className="font-mono text-xs w-8 text-right" style={{ color: "var(--muted-foreground)" }}>{(weight * 100).toFixed(0)}%</span></div>; })}<p className="text-xs mt-1" style={{ color: "var(--muted-foreground)" }}>Learned module weights returned by the backend.</p></div>;
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <div className="rounded-xl p-10 text-center text-sm" style={{ background: "var(--card)", border: "1px solid var(--border)", color: "var(--muted-foreground)" }}>{children}</div>;
}
