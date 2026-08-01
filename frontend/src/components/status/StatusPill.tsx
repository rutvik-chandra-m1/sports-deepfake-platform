type StatusVariant = "positive" | "negative" | "pending" | "neutral";

interface StatusPillProps {
  variant: StatusVariant;
  label: string;
  /** Show the pulsing scan-sweep treatment (reserved for active analysis states). */
  animated?: boolean;
}

const VARIANT_STYLES: Record<StatusVariant, string> = {
  positive: "bg-authentic-dim text-authentic border-authentic/30",
  negative: "bg-suspicious-dim text-suspicious border-suspicious/30",
  pending: "bg-processing/10 text-processing border-processing/30",
  neutral: "bg-surface-hover text-text-muted border-border-strong",
};

const DOT_STYLES: Record<StatusVariant, string> = {
  positive: "bg-authentic",
  negative: "bg-suspicious",
  pending: "bg-processing animate-pulse",
  neutral: "bg-text-faint",
};

/**
 * Small pill used to communicate binary/tri-state outcomes: today it shows
 * backend connectivity (online/offline/checking); from Milestone 10 onward
 * the same component renders the Authentic / Suspicious / Processing verdict
 * on analysis results, so the color language stays consistent everywhere.
 */
export function StatusPill({ variant, label, animated = false }: StatusPillProps) {
  return (
    <span
      className={`relative inline-flex items-center gap-2 rounded-full border px-3 py-1 font-mono text-xs tracking-wide ${VARIANT_STYLES[variant]} ${
        animated ? "scan-sweep" : ""
      }`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${DOT_STYLES[variant]}`} />
      {label}
    </span>
  );
}
