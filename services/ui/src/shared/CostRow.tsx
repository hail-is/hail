import { fmt, fmtDelta } from './format';

interface CostRowProps { label: string; value: number; pctStr?: string; indent?: boolean; bold?: boolean; colorClass?: string; delta?: number }

export function CostRow({ label, value, pctStr, indent = false, bold = false, colorClass, delta }: CostRowProps) {
  return (
    <div className={`flex items-center py-2 border-b border-zinc-100 last:border-0 ${indent ? 'pl-6' : ''}`}>
      <span className={`flex-1 min-w-0 text-zinc-700 ${bold ? 'font-semibold' : ''}`}>{label}</span>
      <span className="shrink-0 w-16 text-right tabular-nums text-zinc-400 text-sm">{pctStr ?? ''}</span>
      <span className={`shrink-0 w-28 text-right tabular-nums ${bold ? 'font-semibold' : ''} ${colorClass ?? ''}`}>{fmt(value)}</span>
      {delta !== undefined && (
        <span className={`shrink-0 w-40 text-right tabular-nums text-zinc-400 text-sm ${bold ? 'font-semibold' : ''}`}>{fmtDelta(delta, value - delta)}</span>
      )}
    </div>
  );
}

export function RatioRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between py-2 border-b border-zinc-100 last:border-0">
      <span className="text-zinc-700">{label}</span>
      <span className="tabular-nums text-zinc-600">{value}</span>
    </div>
  );
}
