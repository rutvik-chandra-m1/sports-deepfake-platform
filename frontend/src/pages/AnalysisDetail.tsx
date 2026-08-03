import { useState } from "react";
import { useParams, Link } from "react-router-dom";
import { StatusPill } from "@/components/status/StatusPill";
import { VerdictHeader } from "@/components/results/VerdictHeader";
import { SignalChart } from "@/components/results/SignalChart";
import { EvidenceViewer } from "@/components/results/EvidenceViewer";
import { ReasonsList } from "@/components/results/ReasonsList";
import { NotesList } from "@/components/results/NotesList";
import { useAnalysis } from "@/hooks/useAnalysis";
import { api, ApiError } from "@/services/api";
import type { DetectorBreakdown } from "@/types/api";

function parseBreakdown(raw: string | null): DetectorBreakdown | null {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as DetectorBreakdown;
  } catch {
    return null;
  }
}

export function AnalysisDetail() {
  const params = useParams<{ id: string }>();
  const id = Number(params.id);
  const { analysis, loading, error, refetch } = useAnalysis(id);
  const [rerunning, setRerunning] = useState(false);
  const [rerunError, setRerunError] = useState<string | null>(null);

  const handleRerun = async () => {
    setRerunning(true);
    setRerunError(null);
    try {
      await api.runAnalysis(id);
      refetch();
    } catch (err) {
      setRerunError(err instanceof ApiError ? err.message : "Could not restart analysis.");
    } finally {
      setRerunning(false);
    }
  };

  if (loading && !analysis) {
    return (
      <div className="rounded-lg border border-border bg-surface p-8">
        <StatusPill variant="pending" label="Loading..." animated />
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-suspicious/30 bg-suspicious-dim p-8">
        <p className="text-sm text-text">{error}</p>
        <Link
          to="/upload"
          className="mt-4 inline-block rounded-md border border-border-strong px-3 py-1.5 text-sm text-text transition-colors hover:bg-surface-hover"
        >
          Back to upload
        </Link>
      </div>
    );
  }

  if (!analysis) return null;

  const breakdown = parseBreakdown(analysis.detector_breakdown);

  return (
    <div className="flex flex-col gap-8">
      {(analysis.status === "pending" || analysis.status === "processing") && (
        <div className="scan-sweep rounded-lg border border-border bg-surface p-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="font-display text-lg font-semibold text-text">Analyzing {analysis.filename}</h1>
              <p className="mt-1 text-sm text-text-muted">
                Running the AI classifier, forensic checks, and sports-intelligence signals. This
                page updates automatically — typically a few seconds, longer on first run while
                models download.
              </p>
            </div>
            <StatusPill variant="pending" label={analysis.status} animated />
          </div>
        </div>
      )}

      {analysis.status === "failed" && (
        <div className="rounded-lg border border-suspicious/30 bg-suspicious-dim p-8">
          <div className="flex items-center justify-between">
            <h1 className="font-display text-lg font-semibold text-text">Analysis failed</h1>
            <StatusPill variant="negative" label="Failed" />
          </div>
          <p className="mt-3 whitespace-pre-wrap text-sm text-text">{analysis.explanation}</p>
          {rerunError && <p className="mt-2 text-xs text-suspicious">{rerunError}</p>}
          <button
            onClick={handleRerun}
            disabled={rerunning}
            className="mt-4 rounded-md border border-border-strong px-3 py-1.5 text-sm text-text transition-colors hover:bg-surface-hover disabled:opacity-50"
          >
            {rerunning ? "Restarting..." : "Retry analysis"}
          </button>
        </div>
      )}

      {analysis.status === "completed" && (
        <>
          <VerdictHeader analysis={analysis} breakdown={breakdown} />

          {breakdown && (
            <div className="rounded-lg border border-border bg-surface p-6">
              <div className="flex flex-col gap-8">
                <ReasonsList reasons={breakdown.reasons} />

                <EvidenceViewer analysisId={id} />

                <div>
                  <h2 className="font-display text-sm font-semibold uppercase tracking-wide text-text-muted">
                    Signal Breakdown
                  </h2>
                  <div className="mt-4">
                    <SignalChart breakdown={breakdown} />
                  </div>
                </div>

                <NotesList notes={breakdown.notes} />
              </div>
            </div>
          )}

          <div className="flex gap-3">
            <button
              onClick={handleRerun}
              disabled={rerunning}
              className="rounded-md border border-border-strong px-3 py-1.5 text-sm text-text transition-colors hover:bg-surface-hover disabled:opacity-50"
            >
              {rerunning ? "Restarting..." : "Re-run analysis"}
            </button>
            <Link
              to="/upload"
              className="rounded-md border border-border-strong px-3 py-1.5 text-sm text-text transition-colors hover:bg-surface-hover"
            >
              Upload another file
            </Link>
          </div>
          {rerunError && <p className="text-xs text-suspicious">{rerunError}</p>}
        </>
      )}
    </div>
  );
}
