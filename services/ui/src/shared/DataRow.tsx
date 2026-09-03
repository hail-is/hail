interface DataRowProps { label: string; value: string; secondary?: string; delta?: string; indent?: boolean; bold?: boolean; colorClass?: string }

// A generic label/value table row with optional secondary (e.g. a percentage) and delta columns.
// Callers are responsible for formatting `value`/`secondary`/`delta` into display strings.
export function DataRow({ label, value, secondary, delta, indent = false, bold = false, colorClass }: DataRowProps) {
  return (
    <div className={`flex items-center py-2 border-b border-zinc-100 last:border-0 ${indent ? 'pl-6' : ''}`}>
      <span className={`flex-1 min-w-0 text-zinc-700 ${bold ? 'font-semibold' : ''}`}>{label}</span>
      <span className="shrink-0 w-16 text-right tabular-nums text-zinc-400 text-sm">{secondary ?? ''}</span>
      <span className={`shrink-0 w-28 text-right tabular-nums ${bold ? 'font-semibold' : ''} ${colorClass ?? ''}`}>{value}</span>
      {delta !== undefined && (
        <span className={`shrink-0 w-40 text-right tabular-nums text-zinc-400 text-sm ${bold ? 'font-semibold' : ''}`}>{delta}</span>
      )}
    </div>
  );
}
