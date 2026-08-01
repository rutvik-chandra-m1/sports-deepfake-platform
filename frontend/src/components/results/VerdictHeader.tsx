import { StatusPill } from "@/components/status/StatusPill";
import type { AnalysisRecord, DetectorBreakdown } from "@/types/api";

interface VerdictHeaderProps {
  analysis: AnalysisRecord;
  breakdown: DetectorBreakdown | null;
}

const RISK_LABEL: Record<string, string> = { low: "Low risk", medium: "Medium risk", high: "High risk" };

export function VerdictHeader({ analysis, breakdown }: VerdictHeaderProps) {
  const isSuspicious = analysis.verdict === "suspicious";
  const headline = breakdown?.headline ?? (isSuspicious ? "Suspicious" : "Authentic");

  return (
    <div
      className={`rounded-lg border p-8 ${
        isSuspicious
          ? "border-suspicious/30 bg-suspicious-dim"
          : "border-authentic/30 bg-authentic-dim"
      }`}
    >
      <div className="flex flex-wrap items-start justify-between gap-6">
        <div>
          <h1 className="font-display text-2xl font-semibold text-text">{headline}</h1>
          <p className="mt-2 truncate font-mono text-xs text-text-faint">{analysis.filename}</p>
        </div>

        <div className="flex items-center gap-6">
          {analysis.confidence_score !== null && (
            <div className="text-right">
              <div
                className={`font-display text-4xl font-semibold tabular-nums ${
                  isSuspicious ? "text-suspicious" : "text-authentic"
                }`}
              >
                {analysis.confidence_score.toFixed(0)}%
              </div>
              <div className="font-mono text-[10px] uppercase tracking-widest text-text-faint">
                Confidence
              </div>
            </div>
          )}

          {analysis.risk_level && (
            <StatusPill
              variant={isSuspicious ? "negative" : "positive"}
              label={RISK_LABEL[analysis.risk_level] ?? analysis.risk_level}
            />
          )}
        </div>
      </div>

      <dl className="mt-6 grid grid-cols-2 gap-4 border-t border-border pt-4 font-mono text-xs sm:grid-cols-4">
        <div>
          <dt className="text-text-faint">Type</dt>
          <dd className="text-text">{analysis.media_type}</dd>
        </div>
        <div>
          <dt className="text-text-faint">Processing time</dt>
          <dd className="text-text">
            {analysis.processing_duration_ms !== null
              ? `${(analysis.processing_duration_ms / 1000).toFixed(2)}s`
              : "—"}
          </dd>
        </div>
        <div>
          <dt className="text-text-faint">Uploaded</dt>
          <dd className="text-text">{new Date(analysis.created_at).toLocaleString()}</dd>
        </div>
        <div>
          <dt className="text-text-faint">Record ID</dt>
          <dd className="text-text">#{analysis.id}</dd>
        </div>
      </dl>
    </div>
  );
}
