export function fmtDollars(v: number | null): string {
  if (v === null || v === undefined) return 'Unlimited';
  return '$' + v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function fmtCost(v: number): string {
  if (v === 0) return '$0';
  if (v < 0.01) return '<$0.01';
  return '$' + v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function fmtTimestamp(ms: number | null | undefined): string {
  if (ms == null) return '';
  const d = new Date(ms);
  if (isNaN(d.getTime())) return String(ms);
  return d.toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}
