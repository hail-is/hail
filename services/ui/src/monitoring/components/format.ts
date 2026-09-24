export function fmt(dollars: number): string {
  return dollars.toLocaleString('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 });
}

export function fmtDelta(delta: number, base: number): string {
  const sign = delta >= 0 ? '+' : '−';
  const absFmt = fmt(Math.abs(delta));
  if (base === 0 || !Number.isFinite(delta / base)) return `${sign}${absFmt}`;
  const pctVal = Math.abs(delta / base) * 100;
  const pctStr = pctVal < 10 ? pctVal.toFixed(1) : Math.round(pctVal).toString();
  return `${sign}${absFmt} / ${sign}${pctStr}%`;
}

export function pct(numerator: number, denominator: number): string {
  if (denominator === 0) return '—';
  return `${((numerator / denominator) * 100).toFixed(1)}%`;
}

export function makeYDollarFormatter(domainMax: number): (_v: number) => string {
  if (domainMax < 1000) return v => `$${Math.round(v)}`;
  if (domainMax < 10000) return v => `$${(v / 1000).toFixed(1)}k`;
  return v => `$${(v / 1000).toFixed(0)}k`;
}
