import { describe, expect, it } from 'vitest';
import type { PrConfig, ReviewRequest } from '../../shared/ciApi';
import {
  buildStateLabel,
  formatAge,
  isBuildOutOfDate,
  matchesSearch,
  reviewOne,
  reviewTwo,
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

describe('reviewOne', () => {
  it('passes on the first non-author approval', () => {
    expect(reviewOne(pr({ reviews: [approved('alice'), approved('carol')] }))).toEqual({
      status: 'pass',
      text: 'approved by carol',
    });
  });

  it('fails on any outstanding change request, regardless of approvals', () => {
    const reviews = [approved('carol'), { author: 'dave', state: 'CHANGES_REQUESTED' }];
    expect(reviewOne(pr({ reviews }))).toEqual({ status: 'fail', text: 'changes requested by dave' });
  });

  it('ignores dismissed reviews', () => {
    expect(reviewOne(pr({ reviews: [{ author: 'carol', state: 'DISMISSED' }] })).status).toBe('pending');
  });

  it('waits on assignees and non-codeowner requests, with the request age', () => {
    const cell = reviewOne(pr({
      reviews: [],
      assignees: ['carol', 'alice'],
      review_requests: [request({ reviewer: 'dave', is_team: false, as_code_owner: false }), request()],
    }));
    expect(cell).toEqual({ status: 'pending', text: 'awaiting carol, dave', requestedAt: '2026-10-18T12:00:00Z' });
  });

  it('says when nobody has been asked', () => {
    expect(reviewOne(pr({ reviews: [] }))).toEqual({ status: 'pending', text: 'no reviewer assigned' });
  });
});

describe('reviewTwo', () => {
  it('waits on an open codeowner request even with an internal approval', () => {
    expect(reviewTwo(pr({ review_requests: [request()] }))).toEqual({
      status: 'pending',
      text: 'awaiting appsec (codeowner)',
      requestedAt: '2026-10-18T12:00:00Z',
    });
  });

  it('waits on the codeowner even with two internal approvals, aged by the oldest request', () => {
    const cell = reviewTwo(pr({
      reviews: [approved('carol'), approved('dave')],
      review_requests: [request(), request({ reviewer: 'compiler', requested_at: '2026-10-17T00:00:00Z' })],
    }));
    expect(cell.text).toBe('awaiting appsec, compiler (codeowner)');
    expect(cell.requestedAt).toBe('2026-10-17T00:00:00Z');
  });

  it('needs one more review after a single approval (e.g. the codeowner approved first)', () => {
    expect(reviewTwo(pr())).toEqual({ status: 'pending', text: 'one more review' });
  });

  it('passes on a second distinct non-author approval', () => {
    expect(reviewTwo(pr({ reviews: [approved('carol'), approved('alice'), approved('dave')] }))).toEqual({
      status: 'pass',
      text: 'approved by dave',
    });
  });

  it('is n/a with no approvals and no codeowner request', () => {
    expect(reviewTwo(pr({ reviews: [] })).status).toBe('na');
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
