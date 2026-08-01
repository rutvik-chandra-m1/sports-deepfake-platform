import { useCallback, useState } from "react";
import { api, ApiError } from "@/services/api";
import type { AnalysisRecord } from "@/types/api";

const ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".avi", ".mkv"];
const MAX_SIZE_BYTES = 200 * 1024 * 1024; // mirrors backend default MAX_UPLOAD_SIZE_MB

export type UploadPhase = "idle" | "uploading" | "success" | "error";

interface UseFileUploadResult {
  phase: UploadPhase;
  progress: number;
  result: AnalysisRecord | null;
  error: string | null;
  upload: (file: File) => void;
  reset: () => void;
}

function validateFile(file: File): string | null {
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    return `"${ext}" isn't supported. Allowed types: ${ALLOWED_EXTENSIONS.join(", ")}`;
  }
  if (file.size > MAX_SIZE_BYTES) {
    return `File is too large (${(file.size / (1024 * 1024)).toFixed(1)}MB). Max size is 200MB.`;
  }
  return null;
}

export function useFileUpload(): UseFileUploadResult {
  const [phase, setPhase] = useState<UploadPhase>("idle");
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<AnalysisRecord | null>(null);
  const [error, setError] = useState<string | null>(null);

  const upload = useCallback((file: File) => {
    const validationError = validateFile(file);
    if (validationError) {
      setPhase("error");
      setError(validationError);
      return;
    }

    setPhase("uploading");
    setProgress(0);
    setError(null);

    api
      .uploadMedia(file, setProgress)
      .then((record) => {
        setResult(record);
        setPhase("success");
      })
      .catch((err: unknown) => {
        setPhase("error");
        setError(err instanceof ApiError ? err.message : "Upload failed unexpectedly.");
      });
  }, []);

  const reset = useCallback(() => {
    setPhase("idle");
    setProgress(0);
    setResult(null);
    setError(null);
  }, []);

  return { phase, progress, result, error, upload, reset };
}
