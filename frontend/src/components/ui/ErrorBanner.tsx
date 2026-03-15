import { AlertTriangle, X, RefreshCw } from "lucide-react";

interface Props {
  message: string;
  onDismiss?: () => void;
  onRetry?: () => void;
}

export default function ErrorBanner({ message, onDismiss, onRetry }: Props) {
  return (
    <div className="mx-4 my-2 flex items-center gap-3 rounded-lg border border-nova-error/30 bg-nova-error/10 px-4 py-2.5 text-sm text-nova-error animate-fade-in">
      <AlertTriangle size={16} className="shrink-0" />
      <span className="flex-1">{message}</span>
      {onRetry && (
        <button onClick={onRetry} className="hover:text-nova-text transition-colors" aria-label="Retry">
          <RefreshCw size={14} />
        </button>
      )}
      {onDismiss && (
        <button onClick={onDismiss} className="hover:text-nova-text transition-colors" aria-label="Dismiss">
          <X size={14} />
        </button>
      )}
    </div>
  );
}
