import type { SelectHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

interface Option {
  value: string;
  label: string;
}

interface Props extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  options: Option[];
  error?: string;
  placeholder?: string;
}

export default function FormSelect({ label, options, error, placeholder, className, id, ...rest }: Props) {
  const selectId = id || label?.toLowerCase().replace(/\s+/g, "-");
  return (
    <div className="space-y-1">
      {label && (
        <label htmlFor={selectId} className="block text-xs font-medium text-nova-text-dim">
          {label}
        </label>
      )}
      <select
        id={selectId}
        className={cn(
          "w-full rounded border border-nova-border bg-nova-bg px-3 py-1.5 text-sm outline-none transition-colors focus:border-nova-accent",
          error && "border-nova-error",
          className,
        )}
        {...rest}
      >
        {placeholder && <option value="">{placeholder}</option>}
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
      {error && <p className="text-xs text-nova-error">{error}</p>}
    </div>
  );
}
