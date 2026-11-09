import { describe, expect, it } from 'vitest';
import type { DeveloperBranchStatus, PrConfig, PrStack } from '../../shared/ciApi';
import { buildListItems, matchesFilter, nextSort, sortPrs, type ListItem, type ListQuery, type SortContext } from './prListView';
import { matchesSearch } from './prTableModel';

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

  it('sorts by merge journey progress, then queue position', () => {
    const queued = (n: number) => ({ pr: n, source_sha: 'x', blockers: [], failures: 0, backoff_until_ms: null });
    const branch = {
      merge_queue: [queued(3), queued(1)],
      pending_merge: null,
      merge_paused_reason: null,
    } as unknown as DeveloperBranchStatus;
    const prs = [
      pr({ number: 1 }),
      pr({ number: 2, merge_blockers: ['label: WIP'], labels: ['WIP'] }),
      pr({ number: 3 }),
      pr({ number: 4, merge_blockers: ['not built'], build_state: null }),
    ];
    expect(numbers(sortPrs(prs, { column: 'merge', dir: 'asc' }, { branch, nowMs: NOW }))).toEqual([3, 1, 2, 4]);
  });
});

describe('buildListItems', () => {
  // Stack 900: #10 merged, #11 targets the branch, #12 and #13 stacked above, #14 closed.
  const stack: PrStack = {
    number: 900,
    base_ref_name: 'main',
    size: 5,
    entries: [
      { position: 1, number: 10, state: 'MERGED', title: 'Groundwork' },
      { position: 2, number: 11, state: 'OPEN', title: 'Bottom' },
      { position: 3, number: 12, state: 'OPEN', title: 'Middle' },
      { position: 4, number: 13, state: 'OPEN', title: 'Top' },
      { position: 5, number: 14, state: 'CLOSED', title: 'Abandoned' },
    ],
    name: 'Widgets',
    doc_url: null,
    meta: {},
  };
  function member(number: number, position: number, overrides: Partial<PrConfig> = {}): PrConfig {
    return pr({
      number,
      title: stack.entries[position - 1].title,
      stack,
      stack_entry: { position },
      display_only: position > 2,
      merge_blockers: position > 2 ? null : [],
      ...overrides,
    });
  }
  const bottom = member(11, 2, { author: 'alice', source_branch_name: 'b' });
  const middle = member(12, 3, { author: 'bob', source_branch_name: 'c' });
  const top = member(13, 4, { author: 'alice', source_branch_name: 'a' });
  const loneA = pr({ number: 5, source_branch_name: 'aa' });
  const loneZ = pr({ number: 6, source_branch_name: 'zz' });
  // API order: branch PRs first, upper-stack PRs after.
  const prs = [loneZ, bottom, loneA, middle, top];

  function query(overrides: Partial<Omit<ListQuery, 'searchPr'>> = {}): ListQuery {
    const q = { filter: 'all' as const, viewer: 'alice', search: '', ...overrides };
    return { ...q, searchPr: (p) => matchesSearch(p, [], q.search) };
  }

  // Lone PRs as numbers, stacks as `s<number>:<row numbers>`.
  function shape(items: ListItem[]): (number | string)[] {
    return items.map((item) => (item.kind === 'pr'
      ? item.pr.number
      : `s${item.stack.number}:${item.rows.map((r) => (r.kind === 'pr' ? r.pr.number : r.entry.number)).join(',')}`));
  }

  it('groups a stack into one item with every entry by position, in place of its first member', () => {
    const items = buildListItems(prs, query(), null, ctx);
    expect(shape(items)).toEqual([6, 's900:10,11,12,13,14', 5]);
    const s = items[1];
    expect(s.kind === 'stack' && s.matchCount).toBe(null);
    expect(s.kind === 'stack' && s.rows.map((r) => r.kind)).toEqual(['entry', 'pr', 'pr', 'pr', 'entry']);
  });

  it('sorts a stack as a unit keyed by its bottom PR', () => {
    // Bottom PR's branch is `b`; the top member's `a` doesn't pull the stack first.
    expect(shape(buildListItems(prs, query(), { column: 'branch', dir: 'asc' }, ctx)))
      .toEqual([5, 's900:10,11,12,13,14', 6]);
    expect(shape(buildListItems(prs, query(), { column: 'branch', dir: 'desc' }, ctx)))
      .toEqual([6, 's900:10,11,12,13,14', 5]);
  });

  it('a filter hides non-matching members, keeps merged / closed entries, and counts matching members', () => {
    const items = buildListItems(prs, query({ filter: 'mine' }), null, ctx);
    expect(shape(items)).toEqual([6, 's900:10,11,13,14', 5]);
    expect(items[1].kind === 'stack' && items[1].matchCount).toBe(2);
  });

  it('drops a stack with no matching rows', () => {
    expect(shape(buildListItems(prs, query({ filter: 'mine', viewer: 'nobody' }), null, ctx))).toEqual([]);
  });

  it('search matches merged / closed entries by title', () => {
    const items = buildListItems(prs, query({ search: 'groundwork' }), null, ctx);
    expect(shape(items)).toEqual(['s900:10,14']);
    expect(items[0].kind === 'stack' && items[0].matchCount).toBe(1);
  });

  it('search matching the stack name keeps every row', () => {
    expect(shape(buildListItems(prs, query({ search: 'widgets' }), null, ctx))).toEqual(['s900:10,11,12,13,14']);
  });

  it('a stack whose bottom PR is filtered out still sorts by it', () => {
    const items = buildListItems(prs, query({ filter: 'mine', search: 'top' }), { column: 'branch', dir: 'asc' }, ctx);
    expect(shape(items)).toEqual(['s900:10,13,14']);
  });
});
