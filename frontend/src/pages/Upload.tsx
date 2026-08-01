import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Dropzone } from "@/components/upload/Dropzone";
import { ProgressBar } from "@/components/status/ProgressBar";
import { StatusPill } from "@/components/status/StatusPill";
import { useFileUpload } from "@/hooks/useFileUpload";

export function Upload() {
  const { phase, progress, result, error, upload, reset } = useFileUpload();
  const navigate = useNavigate();

  // Once the upload finishes, hand off to the analysis detail page — it
  // polls until the pipeline completes, so there's nothing more to do here.
  useEffect(() => {
    if (phase === "success" && result) {
      navigate(`/analysis/${result.id}`);
    }
  }, [phase, result, navigate]);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="font-display text-2xl font-semibold text-text">Upload</h1>
        <p className="mt-1 text-sm text-text-muted">
          Upload an athlete interview, match clip, or broadcast still for verification. You'll be
          taken to the results page once it's uploaded — analysis runs automatically.
        </p>
      </div>

      {phase === "idle" && <Dropzone onFileSelected={upload} />}

      {(phase === "uploading" || phase === "success") && (
        <div className="rounded-lg border border-border bg-surface p-8">
          <div className="mb-4 flex items-center justify-between">
            <span className="font-display text-sm font-semibold text-text">Uploading...</span>
            <StatusPill variant="pending" label="Uploading" animated />
          </div>
          <ProgressBar percent={progress} />
        </div>
      )}

      {phase === "error" && (
        <div className="rounded-lg border border-suspicious/30 bg-suspicious-dim p-8">
          <div className="flex items-center justify-between">
            <span className="font-display text-sm font-semibold text-text">Upload failed</span>
            <StatusPill variant="negative" label="Error" />
          </div>
          <p className="mt-3 text-sm text-text">{error}</p>
          <button
            onClick={reset}
            className="mt-4 rounded-md border border-border-strong px-3 py-1.5 text-sm text-text transition-colors hover:bg-surface-hover"
          >
            Try again
          </button>
        </div>
      )}
    </div>
  );
}
