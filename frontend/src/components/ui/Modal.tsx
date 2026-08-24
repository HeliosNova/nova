import { useEffect, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { cn } from "@/lib/utils";
import { X } from "lucide-react";

const sizeMap = {
  sm: "max-w-sm",
  md: "max-w-md",
  lg: "max-w-lg",
  xl: "max-w-xl",
} as const;

interface Props {
  open: boolean;
  onClose: () => void;
  title?: string;
  size?: keyof typeof sizeMap;
  footer?: ReactNode;
  children: ReactNode;
}

export default function Modal({ open, onClose, title, size = "md", footer, children }: Props) {
  const dialogRef = useRef<HTMLDivElement>(null);

  // Scroll lock + auto-focus
  useEffect(() => {
    if (!open) return;
    document.body.style.overflow = "hidden";
    // Auto-focus first focusable element
    requestAnimationFrame(() => {
      if (dialogRef.current) {
        const focusable = dialogRef.current.querySelector<HTMLElement>(
          "button, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])"
        );
        focusable?.focus();
      }
    });
    return () => {
      document.body.style.overflow = "";
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "Tab" && dialogRef.current) {
        const focusable = dialogRef.current.querySelectorAll<HTMLElement>(
          "button, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])"
        );
        if (focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  // Portal to <body> so the overlay escapes any ancestor with backdrop-filter /
  // transform (e.g. the cosmos region panels), which would otherwise trap this
  // fixed overlay inside that element's clipped stacking context.
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-fade-in">
      <div
        ref={dialogRef}
        className={cn(
          "w-full rounded-lg border border-nova-border bg-nova-surface/90 shadow-xl backdrop-blur-xl animate-scale-in",
          sizeMap[size],
        )}
        role="dialog"
        aria-modal="true"
        aria-label={title || "Dialog"}
      >
        {title && (
          <div className="flex items-center justify-between border-b border-nova-border px-5 py-3">
            <h2 className="text-sm font-semibold">{title}</h2>
            <button
              onClick={onClose}
              className="rounded p-1 text-nova-text-dim hover:text-nova-text hover:bg-nova-border"
              aria-label="Close"
            >
              <X size={16} />
            </button>
          </div>
        )}
        <div className="px-5 py-4">{children}</div>
        {footer && (
          <div className="flex justify-end gap-2 border-t border-nova-border px-5 py-3">
            {footer}
          </div>
        )}
      </div>
    </div>,
    document.body,
  );
}
