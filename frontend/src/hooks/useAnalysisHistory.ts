import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "@/services/api";
import type { AnalysisRecord } from "@/types/api";

const PAGE_SIZE = 10;
const SEARCH_DEBOUNCE_MS = 300;

export interface HistoryFilters {
  search: string;
  verdict: "" | "authentic" | "suspicious";
  status: "" | "pending" | "processing" | "completed" | "failed";
}

const DEFAULT_FILTERS: HistoryFilters = { search: "", verdict: "", status: "" };

interface UseAnalysisHistoryResult {
  items: AnalysisRecord[];
  total: number;
  page: number;
  pageCount: number;
  loading: boolean;
  error: string | null;
  filters: HistoryFilters;
  setFilters: (filters: HistoryFilters) => void;
  setPage: (page: number) => void;
  deleteItem: (id: number) => Promise<void>;
}

export function useAnalysisHistory(): UseAnalysisHistoryResult {
  const [filters, setFiltersState] = useState<HistoryFilters>(DEFAULT_FILTERS);
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [page, setPage] = useState(0);
  const [items, setItems] = useState<AnalysisRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refetchCount, setRefetchCount] = useState(0);

  // Debounce free-text search so we don't fire a request per keystroke.
  useEffect(() => {
    const timeout = setTimeout(() => setDebouncedSearch(filters.search), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(timeout);
  }, [filters.search]);

  const setFilters = useCallback((next: HistoryFilters) => {
    setFiltersState(next);
    setPage(0); // changing filters always resets to page 1
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    api
      .listAnalyses({
        offset: page * PAGE_SIZE,
        limit: PAGE_SIZE,
        search: debouncedSearch || undefined,
        verdict: filters.verdict || undefined,
        status: filters.status || undefined,
      })
      .then((response) => {
        if (cancelled) return;
        setItems(response.items);
        setTotal(response.total);
        setLoading(false);
        setError(null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof ApiError ? err.message : "Failed to load history.");
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [page, debouncedSearch, filters.verdict, filters.status, refetchCount]);

  const deleteItem = useCallback(async (id: number) => {
    await api.deleteAnalysis(id);
    setRefetchCount((n) => n + 1);
  }, []);

  return {
    items,
    total,
    page,
    pageCount: Math.max(1, Math.ceil(total / PAGE_SIZE)),
    loading,
    error,
    filters,
    setFilters,
    setPage,
    deleteItem,
  };
}
