import type { HistoryFilters as HistoryFiltersType } from "@/hooks/useAnalysisHistory";

interface HistoryFiltersProps {
  filters: HistoryFiltersType;
  onChange: (filters: HistoryFiltersType) => void;
}

const SELECT_CLASSES =
  "rounded-md border border-border-strong bg-bg-raised px-3 py-2 text-sm text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-accent";

export function HistoryFilters({ filters, onChange }: HistoryFiltersProps) {
  return (
    <div className="flex flex-wrap gap-3">
      <input
        type="text"
        placeholder="Search filename..."
        value={filters.search}
        onChange={(event) => onChange({ ...filters, search: event.target.value })}
        className="min-w-[200px] flex-1 rounded-md border border-border-strong bg-bg-raised px-3 py-2 text-sm text-text placeholder:text-text-faint focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
      />

      <select
        value={filters.verdict}
        onChange={(event) =>
          onChange({ ...filters, verdict: event.target.value as HistoryFiltersType["verdict"] })
        }
        className={SELECT_CLASSES}
      >
        <option value="">All verdicts</option>
        <option value="authentic">Authentic</option>
        <option value="suspicious">Suspicious</option>
      </select>

      <select
        value={filters.status}
        onChange={(event) =>
          onChange({ ...filters, status: event.target.value as HistoryFiltersType["status"] })
        }
        className={SELECT_CLASSES}
      >
        <option value="">All statuses</option>
        <option value="pending">Pending</option>
        <option value="processing">Processing</option>
        <option value="completed">Completed</option>
        <option value="failed">Failed</option>
      </select>
    </div>
  );
}
