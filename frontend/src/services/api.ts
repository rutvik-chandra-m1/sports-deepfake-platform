import type { AnalysisListResponse, AnalysisRecord, HealthResponse } from "@/types/api";

/**
 * Base URL of the backend API, e.g. http://localhost:8000/api/v1.
 * Configured via VITE_API_BASE_URL (see .env.example) so the same build
 * can point at different environments without a code change.
 */
const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

export class ApiError extends Error {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      headers: { Accept: "application/json", ...init?.headers },
      ...init,
    });
  } catch {
    throw new ApiError(
      "Could not reach the backend. Is it running at " + API_BASE_URL + "?",
    );
  }

  if (!response.ok) {
    throw new ApiError(
      `Request to ${path} failed with status ${response.status}`,
      response.status,
    );
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

/**
 * Uploads a file with progress reporting. Implemented with XMLHttpRequest
 * rather than fetch() because fetch has no upload-progress event yet.
 */
function uploadMedia(
  file: File,
  onProgress?: (percent: number) => void,
): Promise<AnalysisRecord> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE_URL}/media/upload`);

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    xhr.onload = () => {
      let body: unknown;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        body = null;
      }

      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body as AnalysisRecord);
      } else {
        const detail =
          body && typeof body === "object" && "detail" in body
            ? String((body as { detail: unknown }).detail)
            : `Upload failed with status ${xhr.status}`;
        reject(new ApiError(detail, xhr.status));
      }
    };

    xhr.onerror = () => {
      reject(new ApiError(`Could not reach the backend at ${API_BASE_URL}.`));
    };

    const formData = new FormData();
    formData.append("file", file);
    xhr.send(formData);
  });
}

export interface ListAnalysesParams {
  offset?: number;
  limit?: number;
  search?: string;
  verdict?: "authentic" | "suspicious";
  status?: "pending" | "processing" | "completed" | "failed";
}

function buildQuery(params: Record<string, string | number | undefined>): string {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") query.set(key, String(value));
  }
  const queryString = query.toString();
  return queryString ? `?${queryString}` : "";
}

/**
 * API client — one exported function per backend endpoint. Pages/components
 * should call these rather than using fetch() directly, so request shape,
 * error handling, and base URL logic all live in one place.
 */
export const api = {
  getHealth: () => request<HealthResponse>("/health"),
  uploadMedia,
  getAnalysis: (id: number) => request<AnalysisRecord>(`/analysis/${id}`),
  listAnalyses: ({ offset = 0, limit = 50, search, verdict, status }: ListAnalysesParams = {}) =>
    request<AnalysisListResponse>(
      `/analysis${buildQuery({ offset, limit, search, verdict, status })}`,
    ),
  runAnalysis: (id: number) =>
    request<AnalysisRecord>(`/analysis/${id}/run`, { method: "POST" }),
  deleteAnalysis: (id: number) => request<void>(`/analysis/${id}`, { method: "DELETE" }),
};
