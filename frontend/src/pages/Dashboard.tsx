import { Link } from "react-router-dom";
import { useBackendHealth } from "@/hooks/useBackendHealth";
import { useRecentAnalyses } from "@/hooks/useRecentAnalyses";
import { StatusPill } from "@/components/status/StatusPill";
import { AnalysisListItem } from "@/components/analysis/AnalysisListItem";

export function Dashboard() {
  const { state, health, error, recheck } = useBackendHealth();
  const { items, loading: analysesLoading } = useRecentAnalyses(5);

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="font-display text-2xl font-semibold text-text">Dashboard</h1>
        <p className="mt-1 text-sm text-text-muted">
          Upload media for verification, then track it here.
        </p>
      </div>

      <div
        className={`rounded-lg border border-border bg-surface p-6 ${
          state === "checking" ? "scan-sweep" : ""
        }`}
      >
        <div className="flex items-center justify-between">
          <div>
            <h2 className="font-display text-sm font-semibold uppercase tracking-wide text-text-muted">
              Backend Connection
            </h2>
            <p className="mt-2 font-mono text-xs text-text-faint">
              {import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1"}
            </p>
          </div>

          {state === "checking" && (
            <StatusPill variant="pending" label="Checking..." animated />
          )}
          {state === "online" && <StatusPill variant="positive" label="Online" />}
          {state === "offline" && <StatusPill variant="negative" label="Offline" />}
        </div>

        {state === "online" && health && (
          <dl className="mt-6 grid grid-cols-1 gap-4 border-t border-border pt-6 font-mono text-sm sm:grid-cols-3">
            <div>
              <dt className="text-text-faint">App</dt>
              <dd className="text-text">{health.app_name}</dd>
            </div>
            <div>
              <dt className="text-text-faint">Version</dt>
              <dd className="text-text">{health.app_version}</dd>
            </div>
            <div>
              <dt className="text-text-faint">Environment</dt>
              <dd className="text-text">{health.environment}</dd>
            </div>
          </dl>
        )}

        {state === "offline" && (
          <div className="mt-6 border-t border-border pt-6">
            <p className="text-sm text-suspicious">{error}</p>
            <p className="mt-1 text-xs text-text-muted">
              Start the backend with <code className="font-mono">uvicorn app.main:app --reload</code> in{" "}
              <code className="font-mono">backend/</code>, then retry.
            </p>
            <button
              onClick={recheck}
              className="mt-4 rounded-md border border-border-strong px-3 py-1.5 text-sm text-text transition-colors hover:bg-surface-hover"
            >
              Retry
            </button>
          </div>
        )}
      </div>

      <div className="rounded-lg border border-border bg-surface p-6">
        <div className="flex items-center justify-between">
          <h2 className="font-display text-sm font-semibold uppercase tracking-wide text-text-muted">
            Recent Analyses
          </h2>
          <Link to="/upload" className="text-xs text-accent hover:underline">
            Upload new
          </Link>
        </div>

        <div className="mt-4 flex flex-col gap-2">
          {analysesLoading && <p className="text-sm text-text-muted">Loading...</p>}

          {!analysesLoading && items.length === 0 && (
            <p className="font-mono text-xs uppercase tracking-widest text-text-faint">
              No analyses yet — upload something to get started.
            </p>
          )}

          {items.map((record) => (
            <AnalysisListItem key={record.id} record={record} />
          ))}
        </div>
      </div>
    </div>
  );
}
