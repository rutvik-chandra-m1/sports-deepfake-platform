import { Link } from "react-router-dom";
import { StatusPill } from "@/components/status/StatusPill";
import type { AnalysisRecord } from "@/types/api";

function pillFor(record: AnalysisRecord) {
  if (record.status === "pending" || record.status === "processing") {
    return <StatusPill variant="pending" label={record.status} animated />;
  }
  if (record.status === "failed") {
    return <StatusPill variant="negative" label="Failed" />;
  }
  if (record.verdict === "suspicious") {
    return <StatusPill variant="negative" label="Suspicious" />;
  }
  return <StatusPill variant="positive" label="Authentic" />;
}

interface AnalysisListItemProps {
  record: AnalysisRecord;
  onDelete?: (id: number) => void;
}

export function AnalysisListItem({ record, onDelete }: AnalysisListItemProps) {
  return (
    <div className="flex items-center gap-2">
      <Link
        to={`/analysis/${record.id}`}
        className="flex min-w-0 flex-1 items-center justify-between gap-4 rounded-md border border-border px-4 py-3 transition-colors hover:bg-surface-hover"
      >
        <div className="min-w-0">
          <p className="truncate text-sm text-text">{record.filename}</p>
          <p className="mt-0.5 font-mono text-xs text-text-faint">
            {record.media_type} · {new Date(record.created_at).toLocaleString()}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-3">
          {record.confidence_score !== null && (
            <span className="font-mono text-xs text-text-muted">
              {record.confidence_score.toFixed(0)}%
            </span>
          )}
          {pillFor(record)}
        </div>
      </Link>

      {onDelete && (
        <button
          onClick={() => onDelete(record.id)}
          aria-label={`Delete ${record.filename}`}
          className="shrink-0 rounded-md border border-border px-3 py-3 text-text-faint transition-colors hover:border-suspicious/40 hover:text-suspicious"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0-1 14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2L4 6h16Z" />
          </svg>
        </button>
      )}
    </div>
  );
}
