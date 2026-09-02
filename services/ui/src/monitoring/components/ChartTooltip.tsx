export type SeriesStats = Record<string, { mean: number; std: number } | null>;

interface ChartTooltipProps {
  active?: boolean;
  payload?: readonly { name?: string | number; value?: number | string | readonly (number | string)[]; dataKey?: string | number | ((obj: unknown) => unknown); color?: string; fill?: string }[];
  label?: string | number;
  stats: { mean: number; std: number } | null;
  seriesStats?: SeriesStats;
  format: (v: number) => string;
  stacked?: boolean;
  threshold?: number;
}

export function ChartTooltip({ active, payload, label, stats, seriesStats, format, stacked = false, threshold }: ChartTooltipProps) {
  if (!active || !payload?.length) return null;
  const numVal = (v?: number | string | readonly (number | string)[]) => typeof v === 'number' ? v : 0;
  const sigmaStr = (v: number, s: { mean: number; std: number } | null | undefined) => {
    if (!s || s.std === 0) return null;
    const z = (v - s.mean) / s.std;
    return `μ${z >= 0 ? '+' : '−'}${Math.abs(z).toFixed(1)}σ`;
  };
  const sorted = [...payload].sort((a, b) => numVal(b.value) - numVal(a.value));
  const visible = threshold !== undefined ? sorted.filter(p => numVal(p.value) >= threshold) : sorted;
  const otherTotal = threshold !== undefined ? sorted.filter(p => numVal(p.value) < threshold).reduce((s, p) => s + numVal(p.value), 0) : 0;
  const total = payload.reduce((s, p) => s + numVal(p.value), 0);
  const totalSigma = stacked && payload.length > 1 ? sigmaStr(total, stats) : null;
  return (
    <div className="bg-white border border-zinc-200 rounded shadow-lg px-3 py-2 text-sm min-w-max">
      <p className="font-medium text-zinc-700 mb-1">{label}</p>
      {visible.map(p => {
        const val = numVal(p.value);
        const key = typeof p.dataKey === 'string' ? p.dataKey : undefined;
        const sg = sigmaStr(val, key && seriesStats ? seriesStats[key] : (payload.length === 1 ? stats : null));
        return (
          <div key={key ?? String(p.name)} className="flex items-center gap-3 text-xs py-0.5">
            <span className="flex items-center gap-1 text-zinc-600 flex-1">
              <span style={{ color: p.fill ?? p.color }}>■</span>
              {p.name ?? ''}
            </span>
            <span className="tabular-nums font-medium text-zinc-800">{format(val)}</span>
            <span className="tabular-nums text-indigo-400 w-16 text-right">{sg ?? ''}</span>
          </div>
        );
      })}
      {threshold !== undefined && otherTotal > 0 && (
        <div className="flex items-center gap-3 text-xs py-0.5">
          <span className="flex items-center gap-1 text-zinc-400 flex-1">
            <span>■</span>
            (Other)
          </span>
          <span className="tabular-nums font-medium text-zinc-800">{format(otherTotal)}</span>
          <span className="tabular-nums text-indigo-400 w-16 text-right" />
        </div>
      )}
      {stacked && payload.length > 1 && (
        <div className="flex items-center gap-3 text-xs pt-1 mt-0.5 border-t border-zinc-100">
          <span className="text-zinc-500 font-medium flex-1">Total</span>
          <span className="tabular-nums font-medium text-zinc-800">{format(total)}</span>
          <span className="tabular-nums text-indigo-400 w-16 text-right">{totalSigma ?? ''}</span>
        </div>
      )}
    </div>
  );
}
