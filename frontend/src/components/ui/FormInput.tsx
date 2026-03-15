import type { InputHTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/utils";

interface Props extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  icon?: ReactNode;
  error?: string;
}

export default function FormInput({ label, icon, error, className, id, ...rest }: Props) {
  const inputId = id || label?.toLowerCase().replace(/\s+/g, "-");
  return (
    <div className="space-y-1">
      {label && (
        <label htmlFor={inputId} className="block text-xs font-medium text-nova-text-dim">
          {label}
        </label>
      )}
      <div className="relative">
        {icon && (
          <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-nova-text-dim">
            {icon}
          </span>
        )}
        <input
          id={inputId}
          className={cn(
            "w-full rounded border border-nova-border bg-nova-bg px-3 py-1.5 text-sm outline-none transition-colors focus:border-nova-accent",
            icon ? "pl-8" : undefined,
            error && "border-nova-error",
            className,
          )}
          {...rest}
        />
      </div>
      {error && <p className="text-xs text-nova-error">{error}</p>}
    </div>
  );
}
