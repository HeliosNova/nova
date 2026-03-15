interface Props {
  lines?: number;
}

const widths = ["w-full", "w-3/4", "w-5/6", "w-2/3", "w-4/5", "w-1/2"];

export default function Skeleton({ lines = 4 }: Props) {
  return (
    <div className="space-y-3 py-2">
      {Array.from({ length: lines }, (_, i) => (
        <div
          key={i}
          className={`h-4 rounded bg-nova-border/50 animate-skeleton-pulse ${widths[i % widths.length]}`}
        />
      ))}
    </div>
  );
}
