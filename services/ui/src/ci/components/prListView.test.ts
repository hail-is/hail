import { describe, expect, it } from 'vitest';
import type { DeveloperBranchStatus, PrConfig } from '../../shared/ciApi';
import { matchesFilter, nextSort, sortPrs, type SortContext } from './prListView';

function pr(overrides: Partial<PrConfig> = {}): PrConfig {
  return {
    number: 1,
    title: 'A PR',
    author: 'alice',
    assignees: [],
    reviewers: [],
    source_branch_name: 'alice/a',
    build_state: 'success',
    review_state: null,
    labels: [],
    out_of_date: false,
    gh_statuses: {},
    batch_id: null,
    review_requests: [],
    reviews: [],
    merge_blockers: [],
    stack_entry: null,
    stack: null,
    display_only: false,
    ...overrides,
  };
}

const NOW = Date.parse('2026-11-04T12:00:00Z');
const ctx: SortContext = { branch: undefined, nowMs: NOW };

function numbers(prs: PrConfig[]): number[] {
  return prs.map((p) => p.number);
}

describe('matchesFilter', () => {
  const waitingOnBob = pr({ number: 1, author: 'alice', assignees: ['bob'] });
  const approvedByBob = pr({ number: 2, author: 'alice', assignees: ['bob'], reviews: [{ author: 'bob', state: 'APPROVED' }] });
  const carolsWaitingOnAlice = pr({
    number: 3,
    author: 'carol',
    review_requests: [{ reviewer: 'alice', is_team: false, as_code_owner: false, requested_at: '2026-11-01T00:00:00Z' }],
  });
  const codeownerOnly = pr({
    number: 4,
    author: 'carol',
    reviews: [{ author: 'bob', state: 'APPROVED' }],
    review_requests: [{ reviewer: null, is_team: true, as_code_owner: true, requested_at: null }],
  });
  const all = [waitingOnBob, approvedByBob, carolsWaitingOnAlice, codeownerOnly];

  function filtered(filter: Parameters<typeof matchesFilter>[1], viewer: string | null): number[] {
    return numbers(all.filter((p) => matchesFilter(p, filter, viewer)));
  }

  it('all shows everything', () => {
    expect(filtered('all', null)).toEqual([1, 2, 3, 4]);
  });

  it('mine is the viewer\'s own PRs', () => {
    expect(filtered('mine', 'alice')).toEqual([1, 2]);
    expect(filtered('mine', null)).toEqual([]);
  });

  it('waiting on me drops PRs the viewer already reviewed', () => {
    expect(filtered('waiting-on-me', 'bob')).toEqual([1]);
    expect(filtered('waiting-on-me', 'alice')).toEqual([3]);
    expect(filtered('waiting-on-me', null)).toEqual([]);
  });

  it('my open requests is the viewer\'s PRs still waiting on someone', () => {
    expect(filtered('my-requests', 'alice')).toEqual([1]);
    expect(filtered('my-requests', 'carol')).toEqual([3, 4]);
  });

  it('any open request includes open codeowner requests', () => {
    expect(filtered('open-requests', null)).toEqual([1, 3, 4]);
  });
});

describe('nextSort', () => {
  it('cycles asc → desc → unsorted', () => {
    const asc = nextSort(null, 'pr');
    expect(asc).toEqual({ column: 'pr', dir: 'asc' });
    const desc = nextSort(asc, 'pr');
    expect(desc).toEqual({ column: 'pr', dir: 'desc' });
    expect(nextSort(desc, 'pr')).toBeNull();
  });

  it('starts a different column at asc', () => {
    expect(nextSort({ column: 'pr', dir: 'desc' }, 'author')).toEqual({ column: 'author', dir: 'asc' });
  });
});

describe('sortPrs', () => {
  it('leaves API order alone when unsorted', () => {
    const prs = [pr({ number: 2 }), pr({ number: 1 })];
    expect(sortPrs(prs, null, ctx)).toBe(prs);
  });

  it('sorts by PR number both ways', () => {
    const prs = [pr({ number: 2 }), pr({ number: 3 }), pr({ number: 1 })];
    expect(numbers(sortPrs(prs, { column: 'pr', dir: 'asc' }, ctx))).toEqual([1, 2, 3]);
    expect(numbers(sortPrs(prs, { column: 'pr', dir: 'desc' }, ctx))).toEqual([3, 2, 1]);
  });

  it('keeps API order for ties in both directions', () => {
    const prs = [pr({ number: 1, author: 'b' }), pr({ number: 2, author: 'a' }), pr({ number: 3, author: 'b' })];
    expect(numbers(sortPrs(prs, { column: 'author', dir: 'asc' }, ctx))).toEqual([2, 1, 3]);
    expect(numbers(sortPrs(prs, { column: 'author', dir: 'desc' }, ctx))).toEqual([1, 3, 2]);
  });

  it('sorts build state failing, unauthorized, pending, passing', () => {
    const prs = [
      pr({ number: 1, build_state: 'success' }),
      pr({ number: 2, build_state: null }),
      pr({ number: 3, build_state: 'failure' }),
      pr({ number: 4, build_state: 'building' }),
      pr({ number: 5, build_state: 'unauthorized' }),
    ];
    expect(numbers(sortPrs(prs, { column: 'build', dir: 'asc' }, ctx))).toEqual([3, 5, 2, 4, 1]);
  });

  it('sorts a GitHub status column by status, missing last', () => {
    const prs = [
      pr({ number: 1, gh_statuses: { Gate: 'success' } }),
      pr({ number: 2, gh_statuses: {} }),
      pr({ number: 3, gh_statuses: { Gate: 'pending' } }),
      pr({ number: 4, gh_statuses: { Gate: 'failure' } }),
    ];
    expect(numbers(sortPrs(prs, { column: 'status:Gate', dir: 'asc' }, ctx))).toEqual([4, 3, 1, 2]);
  });

  it('sorts labels by the most significant chip, no chips last', () => {
    const prs = [
      pr({ number: 1, labels: ['prio:high'] }),
      pr({ number: 2, labels: ['some-label'] }),
      pr({ number: 3, labels: ['WIP', 'do-not-test'] }),
      pr({ number: 4, labels: ['do-not-test'] }),
    ];
    expect(numbers(sortPrs(prs, { column: 'labels', dir: 'asc' }, ctx))).toEqual([3, 1, 4, 2]);
  });

  it('sorts review stages by needs-a-nudge: oldest pending, pending without a time, fail, pass, n/a', () => {
    const request = (reviewer: string, at: string | null) => ({ reviewer, is_team: false, as_code_owner: false, requested_at: at });
    const prs = [
      pr({ number: 1, reviews: [{ author: 'bob', state: 'APPROVED' }], assignees: ['bob'] }),
      pr({ number: 2, review_requests: [request('bob', '2026-11-03T00:00:00Z')] }),
      pr({ number: 3, reviews: [{ author: 'bob', state: 'CHANGES_REQUESTED' }] }),
      pr({ number: 4, assignees: ['bob'] }),
      pr({ number: 5, review_requests: [request('bob', '2026-11-01T00:00:00Z')] }),
    ];
    expect(numbers(sortPrs(prs, { column: 'review1', dir: 'asc' }, ctx))).toEqual([5, 2, 4, 3, 1]);
  });

  it('sorts ready merges by queue position', () => {
    const queued = (n: number) => ({ pr: n, source_sha: 'x', blockers: [], failures: 0, backoff_until_ms: null });
    const branch = {
      merge_queue: [queued(3), queued(1)],
      pending_merge: null,
      merge_paused_reason: null,
    } as unknown as DeveloperBranchStatus;
    const prs = [
      pr({ number: 1 }),
      pr({ number: 2, merge_blockers: ['label: WIP'] }),
      pr({ number: 3 }),
    ];
    expect(numbers(sortPrs(prs, { column: 'merge', dir: 'asc' }, { branch, nowMs: NOW }))).toEqual([2, 3, 1]);
  });
});
