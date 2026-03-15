import type { ReactNode } from "react";

interface Props {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
}

export default function EmptyState({ icon, title, description, action }: Props) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      {icon && <span className="mb-3 text-nova-border">{icon}</span>}
      <p className="text-sm font-medium text-nova-text-dim">{title}</p>
      {description && <p className="mt-1 text-xs text-nova-text-dim">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
