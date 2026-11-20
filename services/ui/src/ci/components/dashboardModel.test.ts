import { describe, expect, it } from 'vitest';
import type { DeveloperBranchStatus, MergeQueueEntry } from '../../shared/ciApi';
import {
  deployDisplay,
  failedDeploysKey,
  failedJobs,
  isRateLimitLow,
  isTraceback,
  mergeQueueStatus,
} from './dashboardModel';

function branch(overrides: Partial<DeveloperBranchStatus> = {}): DeveloperBranchStatus {
  return {
    branch: 'main',
    sha: 'abc123',
    deploy_state: 'success',
    deploy_batch_id: 7,
    merge_candidate: null,
    n_prs: 3,
    last_update: null,
    merge_paused_reason: null,
    pending_merge: null,
    build_slots: { used: 0, max: 3, running: [], on_deck: null, queued: [] },
    merge_queue: [],
    ...overrides,
  };
}

function entry(overrides: Partial<MergeQueueEntry> = {}): MergeQueueEntry {
  return { pr: 1, blockers: [], failures: 0, backoff_until_ms: null, ...overrides };
}

describe('deployDisplay', () => {
  it('treats a null state with a batch as deploying', () => {
    expect(deployDisplay({ deploy_state: null, deploy_batch_id: 7 })).toEqual({ label: 'deploying', tone: 'warn' });
  });

  it('treats a null state without a batch as no deploy yet', () => {
    expect(deployDisplay({ deploy_state: null, deploy_batch_id: null }).label).toBe('no deploy yet');
  });

  it('flags both failure states', () => {
    expect(deployDisplay({ deploy_state: 'failure', deploy_batch_id: 7 }).tone).toBe('bad');
    expect(deployDisplay({ deploy_state: 'checkout_failure', deploy_batch_id: null }).tone).toBe('bad');
  });
});

describe('mergeQueueStatus', () => {
  const now = 1_000_000;

  it('shows an active backoff ahead of blockers', () => {
    const s = mergeQueueStatus(entry({ failures: 2, backoff_until_ms: now + 90_000, blockers: ['not approved'] }), now);
    expect(s).toEqual({ text: 'merge failed 2 times; retrying in 2m', tone: 'bad' });
  });

  it('ignores an expired backoff', () => {
    expect(mergeQueueStatus(entry({ failures: 1, backoff_until_ms: now - 1 }), now).text).toBe('ready to merge');
  });

  it('joins blockers', () => {
    expect(mergeQueueStatus(entry({ blockers: ['not approved', 'not built'] }), now).text).toBe(
      'not approved · not built',
    );
  });
});

describe('isRateLimitLow', () => {
  it('is low below 10% remaining', () => {
    expect(isRateLimitLow({ remaining: 499, limit: 5000, reset_at: '' })).toBe(true);
    expect(isRateLimitLow({ remaining: 500, limit: 5000, reset_at: '' })).toBe(false);
  });
});

describe('failure information', () => {
  it('tells a traceback from failed jobs', () => {
    const tb = ['Traceback (most recent call last):\n', 'ValueError: boom\n'];
    const jobs = [{ batch_id: 1, job_id: 2, name: 'deploy', state: 'Failed' }];
    expect(isTraceback(tb)).toBe(true);
    expect(failedJobs(tb)).toEqual([]);
    expect(isTraceback(jobs)).toBe(false);
    expect(failedJobs(jobs)).toEqual(jobs);
    expect(failedJobs(null)).toEqual([]);
  });
});

describe('failedDeploysKey', () => {
  it('only changes when the set of failed deploys changes', () => {
    const ok = [branch(), branch({ branch: 'release', deploy_state: null })];
    expect(failedDeploysKey(ok)).toBe('');
    const failed = [branch({ deploy_state: 'failure', deploy_batch_id: 9 })];
    expect(failedDeploysKey(failed)).toBe('main:9:failure');
    expect(failedDeploysKey([branch({ deploy_state: 'failure', deploy_batch_id: 10 })])).not.toBe(
      failedDeploysKey(failed),
    );
  });
});
