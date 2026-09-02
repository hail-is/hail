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
    const total = keys.reduce((s, k) => s + (typeof row[k] === 'number' ? (row[k] as number) : 0), 0);
    if (total === 0) return row;
    const result = { ...row } as Record<string, unknown>;
    for (const k of keys) {
      if (typeof result[k] === 'number') result[k] = ((result[k] as number) / total) * 100;
    }
    return result as T;
  });
}
