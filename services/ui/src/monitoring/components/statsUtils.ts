export function computeStats(values: number[]): { mean: number; std: number } | null {
  if (values.length === 0) return null;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const std = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
  return { mean, std };
}

export interface RegressionResult { slope: number; intercept: number; r2: number }

export function computeRegression(points: { x: number; y: number }[]): RegressionResult | null {
  const n = points.length;
  if (n < 2) return null;
  const sumX = points.reduce((s, p) => s + p.x, 0);
  const sumY = points.reduce((s, p) => s + p.y, 0);
  const sumXY = points.reduce((s, p) => s + p.x * p.y, 0);
  const sumX2 = points.reduce((s, p) => s + p.x * p.x, 0);
  const denom = n * sumX2 - sumX * sumX;
  if (denom === 0) return null;
  const slope = (n * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / n;
  const meanY = sumY / n;
  const ssTot = points.reduce((s, p) => s + (p.y - meanY) ** 2, 0);
  const ssRes = points.reduce((s, p) => s + (p.y - (slope * p.x + intercept)) ** 2, 0);
  const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  return { slope, intercept, r2 };
}

export function toPctRows<T extends Record<string, unknown>>(rows: T[], keys: string[]): T[] {
  return rows.map(row => {
    const values = new Map(Object.entries(row));
    const total = keys.reduce((s, k) => {
      const v = values.get(k);
      return s + (typeof v === 'number' ? v : 0);
    }, 0);
    if (total === 0) return row;
    for (const k of keys) {
      const v = values.get(k);
      if (typeof v === 'number') values.set(k, (v / total) * 100);
    }
    return Object.fromEntries(values) as T;
  });
}
