import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface Props {
  children: ReactNode;
  className?: string;
}

export default function Card({ children, className }: Props) {
  return (
    <div className={cn("rounded-lg border border-nova-border bg-nova-surface p-4", className)}>
      {children}
    </div>
  );
}
