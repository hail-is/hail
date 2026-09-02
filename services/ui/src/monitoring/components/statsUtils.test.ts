import { describe, it, expect } from 'vitest';
import { computeStats, computeRegression, toPctRows } from './statsUtils';

describe('computeStats', () => {
  it('returns null for an empty array', () => {
    expect(computeStats([])).toBeNull();
  });

  it('computes mean and population std dev', () => {
    const stats = computeStats([2, 4, 4, 4, 5, 5, 7, 9]);
    expect(stats).not.toBeNull();
    expect(stats?.mean).toBeCloseTo(5);
    expect(stats?.std).toBeCloseTo(2);
  });

  it('reports zero std dev for identical values', () => {
    const stats = computeStats([3, 3, 3]);
    expect(stats).toEqual({ mean: 3, std: 0 });
  });
});

describe('computeRegression', () => {
  it('returns null with fewer than two points', () => {
    expect(computeRegression([])).toBeNull();
    expect(computeRegression([{ x: 1, y: 1 }])).toBeNull();
  });

  it('fits a perfect line exactly', () => {
    const points = [{ x: 0, y: 1 }, { x: 1, y: 3 }, { x: 2, y: 5 }];
    const reg = computeRegression(points);
    expect(reg).not.toBeNull();
    expect(reg?.slope).toBeCloseTo(2);
    expect(reg?.intercept).toBeCloseTo(1);
    expect(reg?.r2).toBeCloseTo(1);
  });

  it('returns null when all x values are identical (zero variance)', () => {
    expect(computeRegression([{ x: 1, y: 1 }, { x: 1, y: 2 }])).toBeNull();
  });
});

describe('toPctRows', () => {
  it('converts each row so the given keys sum to 100', () => {
    const rows = [{ a: 25, b: 75, label: 'x' }];
    const result = toPctRows(rows, ['a', 'b']);
    expect(result[0].a).toBeCloseTo(25);
    expect(result[0].b).toBeCloseTo(75);
    expect(result[0].label).toBe('x');
  });

  it('leaves a row unchanged when the tracked keys sum to zero', () => {
    const rows = [{ a: 0, b: 0 }];
    expect(toPctRows(rows, ['a', 'b'])).toEqual(rows);
  });

  it('ignores non-numeric fields', () => {
    const rows = [{ a: 10, b: 30, name: 'ignored' }];
    const result = toPctRows(rows, ['a', 'b']);
    expect(result[0].a).toBeCloseTo(25);
    expect(result[0].b).toBeCloseTo(75);
    expect(result[0].name).toBe('ignored');
  });
});
