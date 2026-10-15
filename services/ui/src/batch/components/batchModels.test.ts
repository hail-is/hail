import { describe, it, expect } from 'vitest';
import { costByCategory, isFilteredToBillingProject, withBillingProjectFilter } from './batchModels';

describe('withBillingProjectFilter', () => {
  it('appends to a query with no billing_project term', () => {
    expect(withBillingProjectFilter('user = alice', 'proj')).toBe('user = alice\nbilling_project = proj');
  });

  it('replaces an existing billing_project term instead of ANDing a second one', () => {
    expect(withBillingProjectFilter('billing_project != old\nuser = alice\n', 'proj'))
      .toBe('user = alice\nbilling_project = proj');
  });

  it('handles an empty query', () => {
    expect(withBillingProjectFilter('', 'proj')).toBe('billing_project = proj');
  });
});

describe('isFilteredToBillingProject', () => {
  it('matches = and == but not != or other projects', () => {
    expect(isFilteredToBillingProject('user = a\nbilling_project = proj', 'proj')).toBe(true);
    expect(isFilteredToBillingProject('billing_project==proj', 'proj')).toBe(true);
    expect(isFilteredToBillingProject('billing_project != proj', 'proj')).toBe(false);
    expect(isFilteredToBillingProject('billing_project = proj2', 'proj')).toBe(false);
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
