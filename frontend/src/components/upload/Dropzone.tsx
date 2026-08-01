import { useRef, useState, type DragEvent, type ChangeEvent } from "react";

interface DropzoneProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
  accept?: string;
}

export function Dropzone({ onFileSelected, disabled = false, accept }: DropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    if (disabled) return;

    const file = event.dataTransfer.files?.[0];
    if (file) onFileSelected(file);
  };

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) onFileSelected(file);
    event.target.value = ""; // allow re-selecting the same file later
  };

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(event) => {
        if (!disabled && (event.key === "Enter" || event.key === " ")) {
          inputRef.current?.click();
        }
      }}
      onDragOver={(event) => {
        event.preventDefault();
        if (!disabled) setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed p-16 text-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
        disabled
          ? "cursor-not-allowed border-border opacity-50"
          : isDragging
            ? "border-accent bg-accent/5"
            : "border-border-strong hover:border-text-faint"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        disabled={disabled}
        onChange={handleInputChange}
        className="hidden"
      />
      <svg
        width="32"
        height="32"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        className="text-text-faint"
        aria-hidden="true"
      >
        <path d="M12 16V4m0 0L7 9m5-5 5 5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <div>
        <p className="text-sm text-text">
          Drag and drop a file, or <span className="text-accent">browse</span>
        </p>
        <p className="mt-1 font-mono text-xs text-text-faint">
          Images: JPG, PNG, WEBP · Video: MP4, MOV, AVI, MKV · Max 200MB
        </p>
      </div>
    </div>
  );
}
