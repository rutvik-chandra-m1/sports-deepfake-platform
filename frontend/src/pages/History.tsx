import { useState } from "react";
import { useAnalysisHistory } from "@/hooks/useAnalysisHistory";
import { HistoryFilters } from "@/components/analysis/HistoryFilters";
import { AnalysisListItem } from "@/components/analysis/AnalysisListItem";

export function History() {
  const { items, total, page, pageCount, loading, error, filters, setFilters, setPage, deleteItem } =
    useAnalysisHistory();
  const [pendingDeleteId, setPendingDeleteId] = useState<number | null>(null);

  const handleDelete = async (id: number) => {
    if (pendingDeleteId !== id) {
      setPendingDeleteId(id);
      return;
    }
    await deleteItem(id);
    setPendingDeleteId(null);
  };

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="font-display text-2xl font-semibold text-text">History</h1>
        <p className="mt-1 text-sm text-text-muted">
          {total} analys{total === 1 ? "is" : "es"} recorded.
        </p>
      </div>

      <HistoryFilters filters={filters} onChange={setFilters} />

      {error && <p className="text-sm text-suspicious">{error}</p>}

      {loading && items.length === 0 && (
        <p className="font-mono text-xs uppercase tracking-widest text-text-faint">Loading...</p>
      )}

      {!loading && items.length === 0 && !error && (
        <div className="rounded-lg border border-border p-16 text-center">
          <p className="font-mono text-xs uppercase tracking-widest text-text-faint">
            {filters.search || filters.verdict || filters.status
              ? "No analyses match these filters."
              : "No history yet — upload something to get started."}
          </p>
        </div>
      )}

      <div className="flex flex-col gap-2">
        {items.map((record) => (
          <div key={record.id}>
            <AnalysisListItem record={record} onDelete={handleDelete} />
            {pendingDeleteId === record.id && (
              <p className="mt-1 pl-1 font-mono text-xs text-suspicious">
                Click delete again to confirm removing "{record.filename}".
              </p>
            )}
          </div>
        ))}
      </div>

      {pageCount > 1 && (
        <div className="flex items-center justify-between border-t border-border pt-4">
          <button
            onClick={() => setPage(page - 1)}
            disabled={page === 0}
            className="rounded-md border border-border-strong px-3 py-1.5 text-sm text-text transition-colors hover:bg-surface-hover disabled:opacity-40"
          >
            Previous
          </button>
          <span className="font-mono text-xs text-text-faint">
            Page {page + 1} of {pageCount}
          </span>
          <button
            onClick={() => setPage(page + 1)}
            disabled={page >= pageCount - 1}
            className="rounded-md border border-border-strong px-3 py-1.5 text-sm text-text transition-colors hover:bg-surface-hover disabled:opacity-40"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
