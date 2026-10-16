import { describe, it, expect } from 'vitest';
import { costByCategory, formatCpuHours, hasFieldTerm, withFieldTerm } from './batchModels';

describe('withFieldTerm', () => {
  it('appends to a query with no term for that field', () => {
    expect(withFieldTerm('user = alice', 'billing_project', 'proj')).toBe('user = alice\nbilling_project = proj');
  });

  it('replaces an existing term for that field instead of ANDing a second one', () => {
    expect(withFieldTerm('billing_project != old\nuser = alice\n', 'billing_project', 'proj'))
      .toBe('user = alice\nbilling_project = proj');
  });

  it('removes the term when value is null, keeping the rest', () => {
    expect(withFieldTerm('user = alice\nstate = running', 'state', null)).toBe('user = alice');
  });

  it('does not touch a field whose name merely starts with the same text', () => {
    expect(withFieldTerm('user_tag = x', 'user', 'alice')).toBe('user_tag = x\nuser = alice');
  });

  it('handles an empty query', () => {
    expect(withFieldTerm('', 'billing_project', 'proj')).toBe('billing_project = proj');
  });
});

describe('hasFieldTerm', () => {
  it('matches = and == but not != or other values', () => {
    expect(hasFieldTerm('user = a\nbilling_project = proj', 'billing_project', 'proj')).toBe(true);
    expect(hasFieldTerm('billing_project==proj', 'billing_project', 'proj')).toBe(true);
    expect(hasFieldTerm('billing_project != proj', 'billing_project', 'proj')).toBe(false);
    expect(hasFieldTerm('billing_project = proj2', 'billing_project', 'proj')).toBe(false);
  });
});

describe('formatCpuHours', () => {
  it('converts msec_mcpu to CPU-hours', () => {
    expect(formatCpuHours(2 * 1000 * 3_600_000)).toBe('2');
    expect(formatCpuHours(1000 * 60_000)).toBe('0.02');
    expect(formatCpuHours(1)).toBe('< 0.01');
    expect(formatCpuHours(0)).toBe('0');
    expect(formatCpuHours(null)).toBe('');
  });
});

describe('costByCategory', () => {
  it('sums by leading resource segment, most expensive first', () => {
    expect(costByCategory([
      { resource: 'compute/n1-preemptible/1', cost: 1 },
      { resource: 'memory/n1-preemptible/1', cost: 0.5 },
      { resource: 'compute/n1-nonpreemptible/1', cost: 2 },
      { resource: 'service-fee/1', cost: 0.1 },
    ])).toEqual([
      { resource: 'compute', cost: 3 },
      { resource: 'memory', cost: 0.5 },
      { resource: 'service-fee', cost: 0.1 },
    ]);
  });
});
