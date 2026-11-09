import { describe, expect, it } from 'vitest';
import type { DeveloperBranchStatus, MergeQueueEntry, PrConfig, ReviewRequest } from '../../shared/ciApi';
import {
  buildStateLabel,
  buildTally,
  formatAge,
  isBuildOutOfDate,
  journeyFill,
  journeyMarks,
  labelsCell,
  matchesSearch,
  mergeCell,
  reviewSlots,
  stageTally,
  statusTally,
  statusColumnNames,
  upToDateCell,
} from './prTableModel';

function basePr(overrides: Partial<PrConfig> = {}): PrConfig {
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
    review_requests: [],
    reviews: [{ author: 'carol', state: 'APPROVED' }],
    merge_blockers: [],
    stack_entry: null,
    stack: null,
    display_only: false,
    ...overrides,
  };
}

function branch(overrides: Partial<DeveloperBranchStatus> = {}): DeveloperBranchStatus {
  return {
    branch: 'main',
    sha: 'abc',
    deploy_state: 'success',
    deploy_batch_id: null,
    merge_candidate: null,
    n_prs: 2,
    last_update: null,
    merge_paused_reason: null,
    pending_merge: null,
    build_slots: { used: 0, max: 3, running: [], on_deck: null, queued: [] },
    merge_queue: [],
    ...overrides,
  };
}

function queued(prNumber: number, overrides: Partial<MergeQueueEntry> = {}): MergeQueueEntry {
  return { pr: prNumber, source_sha: 'def', blockers: [], failures: 0, backoff_until_ms: null, ...overrides };
}

const pr = basePr;

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

describe('labelsCell', () => {
  it('passes and lists labels when none blocks merging', () => {
    expect(labelsCell(pr({ labels: ['prio:high', 'bug'] }))).toEqual({ status: 'pass', text: 'prio:high, bug' });
    expect(labelsCell(pr({ labels: [] }))).toEqual({ status: 'pass', text: '' });
  });

  it('fails on a label that blocks merging', () => {
    const cell = labelsCell(pr({ labels: ['WIP', 'bug'], merge_blockers: ['not approved', 'label: WIP'] }));
    expect(cell).toEqual({ status: 'fail', text: 'WIP, bug', detail: 'not merged while labelled WIP' });
  });
});

describe('upToDateCell', () => {
  it('passes for a build against the current target', () => {
    expect(upToDateCell(pr())).toEqual({ status: 'pass', text: 'up to date' });
  });

  it('is pending for a build against an older target', () => {
    const cell = upToDateCell(pr({ build_state: 'building', out_of_date: true }));
    expect(cell.status).toBe('pending');
    expect(cell.text).toBe('out of date');
  });

  it('does not apply without a build', () => {
    for (const build_state of [null, 'unauthorized']) {
      expect(upToDateCell(pr({ build_state, out_of_date: true })).status).toBe('na');
    }
  });
});

describe('matchesSearch', () => {
  const names = ['ci-test', 'Gate'];

  it('matches everything on an empty or blank term', () => {
    expect(matchesSearch(pr(), names, '')).toBe(true);
    expect(matchesSearch(pr(), names, '   ')).toBe(true);
  });

  it('matches title, number, branch, author, labels and review text case-insensitively', () => {
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

function request(overrides: Partial<ReviewRequest> = {}): ReviewRequest {
  return { reviewer: 'appsec', is_team: true, as_code_owner: true, requested_at: '2026-10-18T12:00:00Z', ...overrides };
}

const approved = (author: string) => ({ author, state: 'APPROVED' });

const user = (reviewer: string, requested_at: string | null = '2026-10-18T12:00:00Z') =>
  request({ reviewer, is_team: false, as_code_owner: false, requested_at });

describe('reviewSlots', () => {
  // Not yet approved in CI's eyes, so waits show as pending.
  const pr = (overrides: Partial<PrConfig> = {}): PrConfig => basePr({ review_state: 'pending', ...overrides });

  it('fills both slots with distinct non-author approvals', () => {
    expect(reviewSlots(pr({ reviews: [approved('alice'), approved('carol'), approved('dave'), approved('erin')] }))).toEqual([
      { status: 'pass', text: 'approved by carol' },
      { status: 'pass', text: 'approved by dave' },
    ]);
  });

  it('fails slot 1 on any outstanding change request, regardless of approvals', () => {
    const reviews = [approved('carol'), { author: 'dave', state: 'CHANGES_REQUESTED' }];
    expect(reviewSlots(pr({ reviews }))).toEqual([
      { status: 'fail', text: 'changes requested by dave' },
      { status: 'pass', text: 'approved by carol' },
    ]);
  });

  it('ignores dismissed reviews', () => {
    expect(reviewSlots(pr({ reviews: [{ author: 'carol', state: 'DISMISSED' }] }))[0].status).toBe('pending');
  });

  it('splits two waited-on reviewers across the slots, oldest request first', () => {
    const cells = reviewSlots(pr({
      reviews: [],
      review_requests: [user('kush', '2026-10-14T00:00:00Z'), user('grohli', '2026-10-13T00:00:00Z')],
    }));
    expect(cells).toEqual([
      { status: 'pending', text: 'awaiting grohli', requestedAt: '2026-10-13T00:00:00Z' },
      { status: 'pending', text: 'awaiting kush', requestedAt: '2026-10-14T00:00:00Z' },
    ]);
  });

  it('waits on the remaining requested reviewer after one approval', () => {
    expect(reviewSlots(pr({ review_requests: [user('patrick')] }))[1]).toEqual({
      status: 'pending',
      text: 'awaiting patrick',
      requestedAt: '2026-10-18T12:00:00Z',
    });
  });

  it('counts assignees as waited on, but not once they have reviewed, and never the author', () => {
    const cells = reviewSlots(pr({ assignees: ['carol', 'alice', 'dave'] }));
    expect(cells[1]).toEqual({ status: 'pending', text: 'awaiting dave', requestedAt: undefined });
  });

  it('lists everyone left waiting in the last slot', () => {
    const cells = reviewSlots(pr({ reviews: [], review_requests: [user('a'), user('b'), user('c')] }));
    expect(cells.map((c) => c.text)).toEqual(['awaiting a', 'awaiting b, c']);
  });

  it('gives slot 2 to an open codeowner request, and slot 1 to everyone else', () => {
    const cells = reviewSlots(pr({ reviews: [], review_requests: [request(), user('a'), user('b')] }));
    expect(cells).toEqual([
      { status: 'pending', text: 'awaiting a, b', requestedAt: '2026-10-18T12:00:00Z' },
      { status: 'pending', text: 'awaiting appsec (codeowner)', requestedAt: '2026-10-18T12:00:00Z' },
    ]);
  });

  it('waits on the codeowner even with two internal approvals, aged by the oldest request', () => {
    const cells = reviewSlots(pr({
      reviews: [approved('carol'), approved('dave')],
      review_requests: [request(), request({ reviewer: 'compiler', requested_at: '2026-10-17T00:00:00Z' })],
    }));
    expect(cells[0]).toEqual({ status: 'pass', text: 'approved by carol' });
    expect(cells[1]).toEqual({
      status: 'pending',
      text: 'awaiting appsec, compiler (codeowner)',
      requestedAt: '2026-10-17T00:00:00Z',
    });
  });

  it('shows a codeowner whose reviewer is hidden from CI as just "codeowner"', () => {
    const cells = reviewSlots(pr({ reviews: [], review_requests: [request({ reviewer: null, is_team: false })] }));
    expect(cells).toEqual([
      { status: 'pending', text: 'no reviewer assigned' },
      { status: 'pending', text: 'awaiting codeowner', requestedAt: '2026-10-18T12:00:00Z' },
    ]);
  });

  it('needs a 2nd reviewer after a single approval with nobody else asked', () => {
    expect(reviewSlots(pr())[1]).toEqual({ status: 'pending', text: 'needs a 2nd reviewer' });
  });

  it('is n/a in slot 2 with nobody asked at all', () => {
    expect(reviewSlots(pr({ reviews: [] }))).toEqual([
      { status: 'pending', text: 'no reviewer assigned' },
      { status: 'na', text: '' },
    ]);
  });
  it('shows remaining waits as optional once CI counts the PR as approved', () => {
    const approvedPr = basePr({ review_state: 'approved', review_requests: [user('patrick')] });
    expect(reviewSlots(approvedPr)[1]).toEqual({ status: 'na', text: 'awaiting patrick', requestedAt: '2026-10-18T12:00:00Z' });
    expect(reviewSlots(basePr({ review_state: 'approved' }))[1]).toEqual({ status: 'na', text: '2nd reviewer optional' });
  });

  it('keeps an open codeowner request pending even once approved', () => {
    const codeowner = { ...user('appsec'), as_code_owner: true };
    expect(reviewSlots(basePr({ review_state: 'approved', review_requests: [codeowner] }))[1].status).toBe('pending');
  });
});


describe('formatAge', () => {
  const now = Date.parse('2026-10-21T12:00:00Z');

  it('uses minutes, then hours, then days', () => {
    expect(formatAge('2026-10-21T11:15:00Z', now)).toBe('45m');
    expect(formatAge('2026-10-20T07:00:00Z', now)).toBe('29h');
    expect(formatAge('2026-10-18T12:00:00Z', now)).toBe('3d');
  });

  it('clamps clock skew to zero', () => {
    expect(formatAge('2026-10-21T12:05:00Z', now)).toBe('0m');
  });
});

describe('journeyFill', () => {
  const now = Date.parse('2026-11-04T12:00:00Z');

  it('fills every step for a PR ready to merge', () => {
    expect(journeyFill(pr(), branch({ merge_queue: [queued(15837)] }), now)).toBe(6);
  });

  it('stops at the first step that is not done', () => {
    expect(journeyFill(pr({ merge_blockers: ['checks failing: ci-test'] }), branch(), now)).toBe(0);
    const unreviewed = pr({ review_state: 'pending', reviews: [], merge_blockers: ['not approved'] });
    expect(journeyFill(unreviewed, branch(), now)).toBe(1);
    expect(journeyFill(pr({ labels: ['WIP'], merge_blockers: ['label: WIP'] }), branch(), now)).toBe(3);
    expect(journeyFill(pr({ out_of_date: true, merge_blockers: ['not built against current main'] }), branch(), now)).toBe(4);
    expect(journeyFill(pr(), branch({ merge_paused_reason: 'deploying' }), now)).toBe(5);
  });

  it('counts an optional review slot as done', () => {
    expect(reviewSlots(pr())[1].status).toBe('na');
    expect(journeyFill(pr(), undefined, now)).toBe(6);
  });
});

describe('journeyMarks', () => {
  const now = Date.parse('2026-11-04T12:00:00Z');
  const names = ['Gate', 'ci-test'];
  const ready = branch({ merge_queue: [queued(15837)] });

  it('is all done for a PR ready to merge, skipping statuses it lacks', () => {
    expect(journeyMarks(pr(), names, ready, now)).toEqual(Array(8).fill('done'));
  });

  it('marks the first failed cell as fail and everything after it as todo', () => {
    const failing = pr({ gh_statuses: { Gate: 'failure', 'ci-test': 'success' }, merge_blockers: ['checks failing: Gate'] });
    expect(journeyMarks(failing, names, ready, now)).toEqual(['done', 'fail', 'todo', 'todo', 'todo', 'todo', 'todo', 'todo']);
  });

  it('marks a waiting cell as pending', () => {
    expect(journeyMarks(pr({ build_state: 'unauthorized' }), names, ready, now)[0]).toBe('pending');
    const unreviewed = pr({ review_state: 'pending', reviews: [], merge_blockers: ['not approved'] });
    expect(journeyMarks(unreviewed, names, ready, now).slice(2, 5)).toEqual(['done', 'pending', 'todo']);
  });
});

describe('stack summary tallies', () => {
  it('counts journey steps by status, stacked n/a as pending', () => {
    const cells = [
      { status: 'pass' as const, text: 'approved by carol' },
      { status: 'pending' as const, text: 'awaiting bob' },
      { status: 'pass' as const, text: 'approved by bob' },
      { status: 'na' as const, text: '', stacked: true },
      { status: 'na' as const, text: '' },
    ];
    expect(stageTally(cells)).toEqual([
      { kind: 'stage', state: 'pass', count: 2 },
      { kind: 'stage', state: 'pending', count: 2 },
    ]);
  });

  it('counts build states with upper-stack PRs as pending', () => {
    const prs = [pr(), pr({ build_state: null }), pr({ display_only: true, build_state: null }), pr({ display_only: true, build_state: null })];
    expect(buildTally(prs)).toEqual([
      { kind: 'build', state: 'success', count: 1 },
      { kind: 'build', state: 'pending', count: 3 },
    ]);
  });

  it('counts a GitHub status, skipping PRs without it', () => {
    const prs = [pr(), pr({ gh_statuses: { 'ci-test': 'failure' } }), pr({ gh_statuses: {} })];
    expect(statusTally(prs, 'ci-test')).toEqual([
      { kind: 'build', state: 'success', count: 1 },
      { kind: 'build', state: 'failure', count: 1 },
    ]);
  });
});

describe('mergeCell', () => {
  const now = Date.parse('2026-11-04T12:00:00Z');

  it('is n/a when blocked, with the blockers as detail', () => {
    const cell = mergeCell(pr({ merge_blockers: ['not approved', 'not built'] }), branch(), now);
    expect(cell).toEqual({ status: 'na', text: '', detail: 'not approved\nnot built' });
  });

  it('shows queue position alongside blockers', () => {
    const b = branch({ merge_queue: [queued(1), queued(15837)] });
    const cell = mergeCell(pr({ merge_blockers: ['checks pending: ci-test'] }), b, now);
    expect(cell.text).toBe('#2 in queue');
  });

  it('says an upper-stack PR is stacked rather than blocked', () => {
    const stacked = pr({ display_only: true, merge_blockers: null, build_state: null, out_of_date: null, batch_id: null });
    expect(mergeCell(stacked, branch(), now)).toEqual({ status: 'na', text: '', detail: 'stacked', stacked: true });
    expect(buildStateLabel(stacked)).toBe('stacked');
    expect(isBuildOutOfDate(stacked)).toBe(false);
    expect(upToDateCell(stacked).status).toBe('na');
    expect(labelsCell({ ...stacked, labels: ['WIP'] }).status).toBe('fail');
  });

  it('names the stack entry directly below, whatever its state', () => {
    const stack = {
      number: 900,
      base_ref_name: 'main',
      size: 3,
      entries: [
        { position: 1, number: 10, state: 'OPEN', title: 'Bottom' },
        { position: 2, number: 11, state: 'CLOSED', title: 'Abandoned' },
        { position: 3, number: 12, state: 'OPEN', title: 'Top' },
      ],
      name: null,
      doc_url: null,
      meta: {},
    };
    const top = pr({ display_only: true, merge_blockers: null, build_state: null, out_of_date: null, stack, stack_entry: { position: 3 } });
    expect(mergeCell(top, branch(), now)).toEqual({ status: 'na', text: '', detail: 'stacked on #11', stacked: true });
    expect(buildStateLabel(top)).toBe('stacked on #11');
  });

  it('is ready when nothing blocks it', () => {
    expect(mergeCell(pr(), branch({ merge_queue: [queued(15837)] }), now)).toEqual({ status: 'pass', text: '#1 in queue' });
  });

  it('falls back to the PR alone without branch status', () => {
    expect(mergeCell(pr(), undefined, now)).toEqual({ status: 'pass', text: 'ready' });
    expect(mergeCell(pr({ merge_blockers: ['label: WIP'] }), undefined, now).status).toBe('na');
  });

  it('shows an in-flight merge with its age', () => {
    const b = branch({ pending_merge: { pr: 15837, requested_at_ms: now - 5 * 60_000 } });
    expect(mergeCell(pr(), b, now)).toEqual({
      status: 'pending',
      text: 'merging',
      requestedAt: new Date(now - 5 * 60_000).toISOString(),
    });
  });

  it('shows a branch-wide pause only when the PR is otherwise ready', () => {
    const b = branch({ merge_paused_reason: 'frozen', merge_queue: [queued(15837)] });
    expect(mergeCell(pr(), b, now)).toEqual({ status: 'pending', text: '#1 in queue · merging paused (frozen)' });
    expect(mergeCell(pr({ merge_blockers: ['not built'] }), b, now).status).toBe('na');
  });

  it('shows merge retry backoff', () => {
    const b = branch({ merge_queue: [queued(15837, { failures: 2, backoff_until_ms: now + 10 * 60_000 })] });
    expect(mergeCell(pr(), b, now)).toEqual({ status: 'pending', text: '#1 in queue · merge failed ×2, retrying in 10m' });
  });

  it('ignores a backoff that has expired', () => {
    const b = branch({ merge_queue: [queued(15837, { failures: 1, backoff_until_ms: now - 1 })] });
    expect(mergeCell(pr(), b, now).status).toBe('pass');
  });

  it('is searchable', () => {
    expect(matchesSearch(pr({ merge_blockers: ['checks failing: ci-test'] }), [], 'checks failing')).toBe(true);
    expect(matchesSearch(pr(), [], '#1 in queue', branch({ merge_queue: [queued(15837)] }), now)).toBe(true);
  });
});
