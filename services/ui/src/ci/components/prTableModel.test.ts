import { describe, expect, it } from 'vitest';
import type { PrConfig } from '../../shared/ciApi';
import { buildStateLabel, isBuildOutOfDate, matchesSearch, statusColumnNames } from './prTableModel';

function pr(overrides: Partial<PrConfig> = {}): PrConfig {
  return {
    number: 15837,
    title: 'Fix the flux capacitor',
    author: 'alice',
    assignees: [],
    reviewers: [],
    source_branch_name: 'alice/flux',
    build_state: 'success',
    review_state: 'approved',
    labels: ['prio:high'],
    out_of_date: false,
    gh_statuses: { 'ci-test': 'success' },
    batch_id: 1,
    ...overrides,
  };
}

describe('statusColumnNames', () => {
  it('returns the sorted union of status names', () => {
    const prs = [pr({ gh_statuses: { b: 'success', a: 'pending' } }), pr({ gh_statuses: { c: 'failure', a: 'success' } })];
    expect(statusColumnNames(prs)).toEqual(['a', 'b', 'c']);
  });

  it('is empty when there are no PRs', () => {
    expect(statusColumnNames([])).toEqual([]);
  });
});

describe('buildStateLabel', () => {
  it('shows pending when there is no build state', () => {
    expect(buildStateLabel(pr({ build_state: null, out_of_date: true }))).toBe('pending');
  });

  it('marks out-of-date builds', () => {
    expect(buildStateLabel(pr({ build_state: 'failure', out_of_date: true }))).toBe('failure *');
  });

  it('shows the build state otherwise', () => {
    expect(buildStateLabel(pr({ build_state: 'unauthorized' }))).toBe('unauthorized');
  });
});

describe('isBuildOutOfDate', () => {
  it('is true only for a build that ran against an older target', () => {
    expect(isBuildOutOfDate(pr({ out_of_date: true }))).toBe(true);
    expect(isBuildOutOfDate(pr({ out_of_date: false }))).toBe(false);
    expect(isBuildOutOfDate(pr({ build_state: 'failure', out_of_date: true }))).toBe(true);
    expect(isBuildOutOfDate(pr({ build_state: 'building', out_of_date: true }))).toBe(true);
    for (const build_state of [null, 'unauthorized']) {
      expect(isBuildOutOfDate(pr({ build_state, out_of_date: true }))).toBe(false);
    }
  });
});

describe('matchesSearch', () => {
  const names = ['ci-test', 'Gate'];

  it('matches everything on an empty or blank term', () => {
    expect(matchesSearch(pr(), names, '')).toBe(true);
    expect(matchesSearch(pr(), names, '   ')).toBe(true);
  });

  it('matches title, number, branch, author, labels and review state case-insensitively', () => {
    for (const term of ['FLUX CAP', '#15837', 'alice/flux', 'ALICE', 'prio:high', 'approved']) {
      expect(matchesSearch(pr(), names, term)).toBe(true);
    }
  });

  it('matches build state and status values', () => {
    expect(matchesSearch(pr({ build_state: null }), names, 'pending')).toBe(true);
    expect(matchesSearch(pr({ gh_statuses: { Gate: 'failure' } }), names, 'failure')).toBe(true);
  });

  it('ignores statuses that are not shown as columns', () => {
    expect(matchesSearch(pr({ gh_statuses: { hidden: 'errored' } }), names, 'errored')).toBe(false);
  });

  it('does not match unrelated text', () => {
    expect(matchesSearch(pr(), names, 'bob')).toBe(false);
  });
});
