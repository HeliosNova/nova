import type { TextareaHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

interface Props extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string;
}

export default function FormTextarea({ label, error, className, id, ...rest }: Props) {
  const textareaId = id || label?.toLowerCase().replace(/\s+/g, "-");
  return (
    <div className="space-y-1">
      {label && (
        <label htmlFor={textareaId} className="block text-xs font-medium text-nova-text-dim">
          {label}
        </label>
      )}
      <textarea
        id={textareaId}
        className={cn(
          "w-full resize-none rounded border border-nova-border bg-nova-bg px-3 py-2 text-sm outline-none transition-colors focus:border-nova-accent",
          error && "border-nova-error",
          className,
        )}
        {...rest}
      />
      {error && <p className="text-xs text-nova-error">{error}</p>}
    </div>
  );
}
