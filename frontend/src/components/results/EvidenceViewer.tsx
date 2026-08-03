import { useState } from "react";

/**
 * Visual evidence panel (R7).
 *
 * Until this existed, a user could read a verdict about their image but never
 * see the image, let alone what the model reacted to.
 *
 * The attention tab is labelled carefully on purpose: attention rollout shows
 * WHERE the model looked, not WHY it decided. Presenting a heatmap as "the
 * fake region" is the standard way these visualisations mislead, so the
 * caption states the limit directly rather than in a tooltip.
 */

type EvidenceTab = "original" | "attention" | "compression_analysis" | "frequency_analysis";

interface EvidenceOption {
  id: EvidenceTab;
  label: string;
  path: (id: number) => string;
  caption: string;
}

const OPTIONS: EvidenceOption[] = [
  {
    id: "original",
    label: "Original",
    path: (id) => `/analysis/${id}/media`,
    caption: "The image as uploaded, before any analysis.",
  },
  {
    id: "attention",
    label: "Model attention",
    path: (id) => `/analysis/${id}/evidence/attention`,
    caption:
      "Where the model looked — not why it decided. Bright regions influenced the model's " +
      "summary of the image; on their own they are not evidence of manipulation.",
  },
  {
    id: "compression_analysis",
    label: "Compression (ELA)",
    path: (id) => `/analysis/${id}/evidence/compression_analysis`,
    caption:
      "Error Level Analysis: brightness shows how much each region changed when re-compressed. " +
      "Uniform response is typical of an untouched photo; isolated bright patches can indicate splicing.",
  },
  {
    id: "frequency_analysis",
    label: "Frequency (FFT)",
    path: (id) => `/analysis/${id}/evidence/frequency_analysis`,
    caption:
      "Frequency spectrum. Natural photos fall off smoothly from the centre; regular grid-like " +
      "peaks can indicate generator upsampling artifacts.",
  },
];

const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

export function EvidenceViewer({ analysisId }: { analysisId: number }) {
  const [active, setActive] = useState<EvidenceTab>("original");
  const [failed, setFailed] = useState<Partial<Record<EvidenceTab, boolean>>>({});

  const option = OPTIONS.find((o) => o.id === active)!;

  return (
    <section>
      <h2 className="font-display text-sm font-semibold uppercase tracking-wide text-text-muted">
        Visual Evidence
      </h2>

      <div className="mt-4 flex flex-wrap gap-2" role="tablist" aria-label="Visual evidence views">
        {OPTIONS.map((o) => (
          <button
            key={o.id}
            role="tab"
            aria-selected={active === o.id}
            onClick={() => setActive(o.id)}
            className={`rounded-md border px-3 py-1.5 text-sm transition-colors ${
              active === o.id
                ? "border-accent/50 bg-accent/10 text-text"
                : "border-border-strong text-text-muted hover:bg-surface-hover hover:text-text"
            }`}
          >
            {o.label}
          </button>
        ))}
      </div>

      <div className="mt-4 overflow-hidden rounded-lg border border-border bg-surface-raised">
        {failed[active] ? (
          <p className="p-8 text-center text-sm text-text-muted">
            This view could not be generated.
            {active === "attention" &&
              " The attention map needs the AI backbone, which may still be downloading."}
          </p>
        ) : (
          <img
            src={`${API_BASE_URL}${option.path(analysisId)}`}
            alt={option.label}
            className="max-h-[28rem] w-full object-contain"
            onError={() => setFailed((f) => ({ ...f, [active]: true }))}
          />
        )}
      </div>

      <p className="mt-3 text-xs leading-relaxed text-text-muted">{option.caption}</p>
    </section>
  );
}
