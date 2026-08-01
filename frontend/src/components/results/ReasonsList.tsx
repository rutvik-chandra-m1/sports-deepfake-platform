interface ReasonsListProps {
  reasons: string[];
}

export function ReasonsList({ reasons }: ReasonsListProps) {
  if (reasons.length === 0) return null;

  return (
    <div>
      <h2 className="font-display text-sm font-semibold uppercase tracking-wide text-text-muted">
        Reasons
      </h2>
      <ul className="mt-3 flex flex-col gap-2">
        {reasons.map((reason, index) => (
          <li key={index} className="flex gap-3 text-sm text-text">
            <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-text-faint" />
            <span>{reason}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
