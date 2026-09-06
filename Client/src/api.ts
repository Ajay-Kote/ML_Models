// src/api.ts
// Talks to the fusion_engine FastAPI backend (api.py).
// Change API_BASE_URL if you deploy the backend elsewhere.

export const API_BASE_URL = "http://127.0.0.1:8000";

export interface ContributingModule {
  module: "url" | "sms" | "qr" | "image" | "email";
  risk_probability: number; // 0..1
  label: string;
  learned_weight: number;
  explanation: string;
}

export interface PredictResponse {
  combined_risk_probability: number; // 0..1
  verdict: "safe" | "suspicious" | "high_risk" | string;
  override_triggered: boolean;
  override_module: string | null;
  contributing_modules: ContributingModule[];
  explanation: string;
  fusion_method: string;
}

export interface PredictInputs {
  url?: string;
  smsText?: string;
  emailRaw?: string;
  qrImage?: File | null;
  paymentImage?: File | null;
}

export interface ModelHealthResponse {
  status: "nominal" | "degraded" | string;
  updated_at: string;
  prediction_count: number;
  average_latency_ms: number | null;
  evaluation_metrics: {
    accuracy: number | null;
    f1: number | null;
    roc_auc: number | null;
  };
  model_loaded: boolean;
  model_file: string;
  modules: { id: string; learned_weight: number }[];
  error?: string;
}

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** POST /predict with whichever inputs are provided. */
export async function runPrediction(inputs: PredictInputs): Promise<PredictResponse> {
  const form = new FormData();
  if (inputs.url) form.append("url", inputs.url);
  if (inputs.smsText) form.append("sms_text", inputs.smsText);
  if (inputs.emailRaw) form.append("email_raw", inputs.emailRaw);
  if (inputs.qrImage) form.append("qr_image", inputs.qrImage);
  if (inputs.paymentImage) form.append("payment_image", inputs.paymentImage);

  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/predict`, { method: "POST", body: form });
  } catch (e) {
    throw new ApiError(
      "Could not reach the Fusion API. Is start_backend.bat running on http://127.0.0.1:8000?"
    );
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      /* response wasn't JSON */
    }
    throw new ApiError(detail, res.status);
  }

  return res.json();
}

/** GET runtime health and metadata for the existing fusion model. */
export async function getModelHealth(): Promise<ModelHealthResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/model-health`);
  } catch {
    throw new ApiError("Could not reach the Fusion API health endpoint.");
  }
  if (!res.ok) throw new ApiError(`Model health request failed (${res.status}).`, res.status);
  return res.json();
}

/** Maps the backend's 0..1 probability to the dashboard's 0..100 int score. */
export function toScore(p: number): number {
  return Math.round(p * 100);
}

/** Maps backend verdict strings to the dashboard's Verdict union. */
export function toVerdict(v: string): "Safe" | "Suspicious" | "High Risk" {
  if (v === "high_risk") return "High Risk";
  if (v === "suspicious") return "Suspicious";
  return "Safe";
}
