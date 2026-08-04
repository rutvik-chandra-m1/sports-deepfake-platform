import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/services/api";
import type { AnalysisRecord } from "@/types/api";

/**
 * Fetches an analysis and polls while it is pending/processing.
 *
 * R10 changed the polling strategy. It used to poll every 2s forever: if a
 * record got stuck (which it could, before the backend recovered orphaned
 * jobs), an open tab would hammer the API indefinitely with no way out and no
 * message to the user.
 *
 * Now it backs off geometrically and gives up, reporting the timeout instead
 * of spinning silently. The backend fails orphaned records on restart, but
 * the client must not depend on that to stop.
 */

const INITIAL_POLL_MS = 1500;
const MAX_POLL_MS = 15_000;
const BACKOFF_FACTOR = 1.4;
/** Analyses take seconds; minutes means something is wrong. */
const MAX_POLL_DURATION_MS = 5 * 60 * 1000;

interface UseAnalysisResult {
  analysis: AnalysisRecord | null;
  loading: boolean;
  error: string | null;
  /** True when polling stopped because the record never settled. */
  timedOut: boolean;
  refetch: () => void;
}

export function useAnalysis(id: number): UseAnalysisResult {
  const [analysis, setAnalysis] = useState<AnalysisRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timedOut, setTimedOut] = useState(false);
  const [refetchCount, setRefetchCount] = useState(0);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const refetch = useCallback(() => {
    setTimedOut(false);
    setRefetchCount((n) => n + 1);
  }, []);

  useEffect(() => {
    let cancelled = false;
    let delay = INITIAL_POLL_MS;
    const startedAt = Date.now();

    async function poll() {
      try {
        const result = await api.getAnalysis(id);
        if (cancelled) return;

        setAnalysis(result);
        setLoading(false);
        setError(null);

        const settled = result.status === "completed" || result.status === "failed";
        if (settled) return;

        if (Date.now() - startedAt > MAX_POLL_DURATION_MS) {
          // Stop rather than poll forever. The record may be genuinely stuck;
          // the user gets told instead of watching a spinner indefinitely.
          setTimedOut(true);
          return;
        }

        timeoutRef.current = setTimeout(poll, delay);
        delay = Math.min(delay * BACKOFF_FACTOR, MAX_POLL_MS);
      } catch (err) {
        if (cancelled) return;
        setLoading(false);
        setError(err instanceof ApiError ? err.message : "Failed to load analysis.");
      }
    }

    setLoading(true);
    setTimedOut(false);
    poll();

    return () => {
      cancelled = true;
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [id, refetchCount]);

  return { analysis, loading, error, timedOut, refetch };
}
