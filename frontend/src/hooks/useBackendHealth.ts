import { useEffect, useState } from "react";
import { api, ApiError } from "@/services/api";
import type { HealthResponse } from "@/types/api";

type ConnectionState = "checking" | "online" | "offline";

interface UseBackendHealthResult {
  state: ConnectionState;
  health: HealthResponse | null;
  error: string | null;
  recheck: () => void;
}

/**
 * Checks backend connectivity on mount (and whenever `recheck()` is called).
 * Used today as a Milestone 3 smoke test that the frontend can actually
 * reach the FastAPI backend; the same hook will back a persistent connection
 * indicator once the real dashboard is built.
 */
export function useBackendHealth(): UseBackendHealthResult {
  const [state, setState] = useState<ConnectionState>("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setState("checking");
    setError(null);

    api
      .getHealth()
      .then((result) => {
        if (cancelled) return;
        setHealth(result);
        setState("online");
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setState("offline");
        setError(err instanceof ApiError ? err.message : "Unknown error");
      });

    return () => {
      cancelled = true;
    };
  }, [attempt]);

  return {
    state,
    health,
    error,
    recheck: () => setAttempt((n) => n + 1),
  };
}
