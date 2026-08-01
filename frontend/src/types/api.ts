/**
 * Shared API types. Mirrors the Pydantic schemas defined in
 * backend/app/schemas/. Expand this file as new endpoints are added —
 * one interface per response/request shape, named after the schema it mirrors.
 */

export interface HealthResponse {
  status: "ok" | "degraded";
  app_name: string;
  app_version: string;
  environment: string;
}

export type MediaType = "image" | "video";
export type AnalysisStatus = "pending" | "processing" | "completed" | "failed";
export type Verdict = "authentic" | "suspicious";
export type RiskLevel = "low" | "medium" | "high";

/** One entry in detector_breakdown.signals */
export interface SignalScore {
  score: number; // 0-1
  weight: number; // 0-1, this signal's share of the fused verdict
}

/**
 * Mirrors the JSON stored in Analysis.detector_breakdown (built by
 * fusion_engine.py + explainability/reasoning.py). Parse the raw
 * `detector_breakdown` string with JSON.parse() into this shape.
 */
export interface DetectorBreakdown {
  fused_suspicion_score: number;
  signals: Record<string, SignalScore>;
  unavailable: Record<string, string>;
  headline: string;
  reasons: string[];
  notes: string[];
  technical_summary: string;
  explanation_report: string;
  dl_result: unknown | null;
  forensic_signals: unknown[];
  sports_signals: unknown[];
}

/** Mirrors backend/app/schemas/analysis.py::AnalysisRead */
export interface AnalysisRecord {
  id: number;
  filename: string;
  media_type: MediaType;
  file_path: string | null;
  status: AnalysisStatus;
  verdict: Verdict | null;
  confidence_score: number | null;
  risk_level: RiskLevel | null;
  explanation: string | null;
  detector_breakdown: string | null;
  processing_duration_ms: number | null;
  created_at: string;
  completed_at: string | null;
}

/** Mirrors backend/app/schemas/analysis.py::AnalysisList */
export interface AnalysisListResponse {
  total: number;
  items: AnalysisRecord[];
}
