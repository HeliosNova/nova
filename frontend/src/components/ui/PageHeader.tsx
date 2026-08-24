import { createContext, useContext, type ReactNode } from "react";

/** True when a legacy page is mounted inside the cosmos Systems drawer, which
 *  already renders its own panel header — PageHeader then drops the duplicate
 *  icon + <h1> and keeps only the action buttons (e.g. "New Monitor"). */
export const EmbeddedChromeContext = createContext(false);

interface Props {
  icon?: ReactNode;
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}

export default function PageHeader({ icon, title, subtitle, actions }: Props) {
  const embedded = useContext(EmbeddedChromeContext);
  if (embedded) {
    return actions ? <div className="mb-4 flex items-center justify-end gap-2">{actions}</div> : null;
  }
  return (
    <div className="mb-6 flex items-center justify-between">
      <div className="flex items-center gap-3">
        {icon && <span className="text-nova-text-dim">{icon}</span>}
        <div>
          <h1 className="text-xl font-semibold">{title}</h1>
          {subtitle && <p className="mt-0.5 text-sm text-nova-text-dim">{subtitle}</p>}
        </div>
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}
