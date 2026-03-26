import type { ReactNode } from "react";

export interface Column<T> {
  label: string;
  accessor: keyof T | ((row: T) => ReactNode);
  className?: string;
  /** Hide this column in card mode on mobile */
  hideOnMobile?: boolean;
}

interface Props<T> {
  columns: Column<T>[];
  data: T[];
  keyFn: (row: T, index: number) => string | number;
  renderCell?: (row: T, col: Column<T>, value: ReactNode) => ReactNode;
  /** Extra element rendered before the first column in each row (e.g. checkbox) */
  renderRowPrefix?: (row: T) => ReactNode;
  /** Extra element rendered at the end of each row (e.g. actions) */
  renderRowSuffix?: (row: T) => ReactNode;
  /** Optional header prefix (e.g. select-all checkbox) */
  headerPrefix?: ReactNode;
  /** Optional header suffix (e.g. "Actions" label) */
  headerSuffix?: ReactNode;
}

export default function ResponsiveTable<T>({
  columns,
  data,
  keyFn,
  renderCell,
  renderRowPrefix,
  renderRowSuffix,
  headerPrefix,
  headerSuffix,
}: Props<T>) {
  const getCellValue = (row: T, col: Column<T>): ReactNode => {
    if (typeof col.accessor === "function") {
      return col.accessor(row);
    }
    const val = row[col.accessor];
    if (val === null || val === undefined) return "";
    return String(val);
  };

  return (
    <>
      {/* Desktop: standard table (hidden below md) */}
      <div className="hidden md:block overflow-x-auto rounded-lg border border-nova-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-nova-border bg-nova-surface text-left text-xs text-nova-text-dim">
              {headerPrefix && <th className="px-3 py-2 w-8">{headerPrefix}</th>}
              {columns.map((col, i) => (
                <th key={i} className={`px-3 py-2 ${col.className || ""}`}>
                  {col.label}
                </th>
              ))}
              {headerSuffix && <th className="px-3 py-2 text-right">{headerSuffix}</th>}
            </tr>
          </thead>
          <tbody>
            {data.map((row, ri) => (
              <tr
                key={keyFn(row, ri)}
                className="border-b border-nova-border last:border-0 hover:bg-nova-surface/50"
              >
                {renderRowPrefix && <td className="px-3 py-2">{renderRowPrefix(row)}</td>}
                {columns.map((col, ci) => {
                  const value = getCellValue(row, col);
                  return (
                    <td key={ci} className={`px-3 py-2 ${col.className || ""}`}>
                      {renderCell ? renderCell(row, col, value) : value}
                    </td>
                  );
                })}
                {renderRowSuffix && <td className="px-3 py-2 text-right">{renderRowSuffix(row)}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Mobile: stacked cards (shown below md) */}
      <div className="md:hidden space-y-2">
        {data.map((row, ri) => (
          <div
            key={keyFn(row, ri)}
            className="rounded-lg border border-nova-border bg-nova-surface/30 p-3 space-y-1.5"
          >
            <div className="flex items-center justify-between">
              {renderRowPrefix && <div>{renderRowPrefix(row)}</div>}
              {renderRowSuffix && <div>{renderRowSuffix(row)}</div>}
            </div>
            {columns
              .filter((col) => !col.hideOnMobile)
              .map((col, ci) => {
                const value = getCellValue(row, col);
                return (
                  <div key={ci} className="flex justify-between gap-2 text-sm">
                    <span className="text-xs text-nova-text-dim shrink-0">{col.label}</span>
                    <span className="text-right truncate">
                      {renderCell ? renderCell(row, col, value) : value}
                    </span>
                  </div>
                );
              })}
          </div>
        ))}
      </div>
    </>
  );
}
