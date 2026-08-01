import { useEffect, useState } from "react";
import { api, ApiError } from "@/services/api";
import type { AnalysisRecord } from "@/types/api";

interface UseRecentAnalysesResult {
  items: AnalysisRecord[];
  loading: boolean;
  error: string | null;
}

/** Fetches the most recent N analyses for the dashboard's preview list.
 * Full browsing/filtering/pagination is the History page (Milestone 14). */
export function useRecentAnalyses(limit = 5): UseRecentAnalysesResult {
  const [items, setItems] = useState<AnalysisRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    api
      .listAnalyses({ limit })
      .then((response) => {
        if (cancelled) return;
        setItems(response.items);
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof ApiError ? err.message : "Failed to load recent analyses.");
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [limit]);

  return { items, loading, error };
}
