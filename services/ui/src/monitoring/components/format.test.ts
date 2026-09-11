import { describe, it, expect } from 'vitest';
import { fmt, fmtDelta, pct, makeYDollarFormatter } from './format';

describe('fmt', () => {
  it('formats a positive amount as USD', () => {
    expect(fmt(1234.5)).toBe('$1,234.50');
  });

  it('formats zero', () => {
    expect(fmt(0)).toBe('$0.00');
  });

  it('formats a negative amount', () => {
    expect(fmt(-42)).toBe('-$42.00');
  });
});

describe('fmtDelta', () => {
  it('formats a positive delta with a plus sign and a one-decimal percentage under 10%', () => {
    expect(fmtDelta(5, 100)).toBe('+$5.00 / +5.0%');
  });

  it('formats a negative delta with a minus sign', () => {
    expect(fmtDelta(-5, 100)).toBe('−$5.00 / −5.0%');
  });

  it('rounds percentages >= 10% to whole numbers', () => {
    expect(fmtDelta(50, 100)).toBe('+$50.00 / +50%');
  });

  it('omits the percentage when base is zero', () => {
    expect(fmtDelta(10, 0)).toBe('+$10.00');
  });

  it('shows a 0% change when delta is zero and base is nonzero', () => {
    expect(fmtDelta(0, 100)).toBe('+$0.00 / +0.0%');
  });
});

describe('pct', () => {
  it('formats a ratio as a percentage', () => {
    expect(pct(25, 200)).toBe('12.5%');
  });

  it('returns an em dash when the denominator is zero', () => {
    expect(pct(5, 0)).toBe('—');
  });
});

describe('makeYDollarFormatter', () => {
  it('formats whole dollars below 1000', () => {
    expect(makeYDollarFormatter(500)(42.6)).toBe('$43');
  });

  it('formats thousands with one decimal below 10000', () => {
    expect(makeYDollarFormatter(5000)(1234)).toBe('$1.2k');
  });

  it('formats thousands with no decimal at or above 10000', () => {
    expect(makeYDollarFormatter(20000)(12345)).toBe('$12k');
  });
});
