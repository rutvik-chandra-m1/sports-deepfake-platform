import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/services/api";
import type { AnalysisRecord } from "@/types/api";

const POLL_INTERVAL_MS = 2000;

interface UseAnalysisResult {
  analysis: AnalysisRecord | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

/**
 * Fetches an analysis record and, while its status is "pending" or
 * "processing", polls every 2s until it reaches "completed" or "failed".
 * Real background-task processing (unlike the synchronous TestClient in
 * the backend's own tests) genuinely takes time in a browser, so this
 * polling is load-bearing, not cosmetic.
 */
export function useAnalysis(id: number): UseAnalysisResult {
  const [analysis, setAnalysis] = useState<AnalysisRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refetchCount, setRefetchCount] = useState(0);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const refetch = useCallback(() => setRefetchCount((n) => n + 1), []);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const result = await api.getAnalysis(id);
        if (cancelled) return;

        setAnalysis(result);
        setLoading(false);
        setError(null);

        if (result.status === "pending" || result.status === "processing") {
          timeoutRef.current = setTimeout(poll, POLL_INTERVAL_MS);
        }
      } catch (err) {
        if (cancelled) return;
        setLoading(false);
        setError(err instanceof ApiError ? err.message : "Failed to load analysis.");
      }
    }

    setLoading(true);
    poll();

    return () => {
      cancelled = true;
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [id, refetchCount]);

  return { analysis, loading, error, refetch };
}
