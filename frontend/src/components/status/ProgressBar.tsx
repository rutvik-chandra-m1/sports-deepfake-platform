interface ProgressBarProps {
  percent: number;
  label?: string;
}

export function ProgressBar({ percent, label }: ProgressBarProps) {
  return (
    <div>
      {label && (
        <div className="mb-2 flex items-center justify-between font-mono text-xs text-text-muted">
          <span>{label}</span>
          <span>{percent}%</span>
        </div>
      )}
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-hover">
        <div
          className="h-full rounded-full bg-accent transition-[width] duration-150 ease-out"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
