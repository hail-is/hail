import { describe, expect, it } from 'vitest';
import type { PrConfig, ReviewRequest } from '../../shared/ciApi';
import {
  buildStateLabel,
  formatAge,
  isBuildOutOfDate,
  matchesSearch,
  reviewSlots,
  statusColumnNames,
} from './prTableModel';

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
    review_requests: [],
    reviews: [{ author: 'carol', state: 'APPROVED' }],
    stack_entry: null,
    stack: null,
    display_only: false,
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
