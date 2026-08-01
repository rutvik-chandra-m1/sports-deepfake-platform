interface NotesListProps {
  notes: string[];
}

export function NotesList({ notes }: NotesListProps) {
  if (notes.length === 0) return null;

  return (
    <div>
      <h2 className="font-display text-xs font-semibold uppercase tracking-wide text-text-faint">
        Notes
      </h2>
      <ul className="mt-2 flex flex-col gap-1.5">
        {notes.map((note, index) => (
          <li key={index} className="font-mono text-xs text-text-faint">
            — {note}
          </li>
        ))}
      </ul>
    </div>
  );
}
