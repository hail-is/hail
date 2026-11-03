import { describe, expect, it } from 'vitest';
import type { CIEvent } from '../../shared/ciApi';
import { collapseRepeats, eventDetails, isFailureEvent } from './eventLogModel';

function ev(id: number, overrides: Partial<CIEvent> = {}): CIEvent {
  return {
    id,
    time: `2026-11-03T12:00:${String(id).padStart(2, '0')}.000+00:00`,
    event: 'update_failed',
    target_branch: 'hail-is/hail:main',
    pr_number: null,
    username: null,
    batch_id: null,
    source_sha: null,
    target_sha: null,
    previous_sha: null,
    namespace: null,
    service: null,
    merge_uuid: null,
    message: 'GitHub 502',
    ...overrides,
  };
}

describe('collapseRepeats', () => {
  it('collapses consecutive equal events, keeping newest and oldest', () => {
    const groups = collapseRepeats([ev(3), ev(2), ev(1)]);
    expect(groups).toHaveLength(1);
    expect(groups[0].count).toBe(3);
    expect(groups[0].latest.id).toBe(3);
    expect(groups[0].earliest.id).toBe(1);
  });

  it('ignores fields outside the grouping key', () => {
    const groups = collapseRepeats([ev(2, { batch_id: 7, source_sha: 'a' }), ev(1, { batch_id: 8, source_sha: 'b' })]);
    expect(groups).toHaveLength(1);
  });

  it.each([
    ['event', { event: 'merge_failed' }],
    ['message', { message: 'GitHub 503' }],
    ['pr', { pr_number: 15900 }],
    ['branch', { target_branch: 'hail-is/hail:other' }],
  ])('splits on a different %s', (_name, overrides) => {
    expect(collapseRepeats([ev(2), ev(1, overrides)])).toHaveLength(2);
  });

  it('only collapses adjacent rows', () => {
    const groups = collapseRepeats([ev(3), ev(2, { event: 'ci_started', message: null }), ev(1)]);
    expect(groups.map((g) => g.count)).toEqual([1, 1, 1]);
  });

  it('handles no rows', () => {
    expect(collapseRepeats([])).toEqual([]);
  });
});

describe('isFailureEvent', () => {
  it('flags failures and expired merges', () => {
    expect(isFailureEvent('merge_failed')).toBe(true);
    expect(isFailureEvent('update_failed')).toBe(true);
    expect(isFailureEvent('merge_expired')).toBe(true);
    expect(isFailureEvent('merge_succeeded')).toBe(false);
    expect(isFailureEvent('deploy_failure_alerted')).toBe(false);
  });
});

describe('eventDetails', () => {
  it('shortens shas and shows target moves', () => {
    expect(eventDetails(ev(1, {
      source_sha: '0123456789abcdef',
      previous_sha: 'aaaaaaaaaaaa',
      target_sha: 'bbbbbbbbbbbb',
    }))).toEqual(['source 0123456', 'target aaaaaaa → bbbbbbb']);
  });

  it('shows namespace and service', () => {
    expect(eventDetails(ev(1, { namespace: 'ns', service: 'batch' }))).toEqual(['ns/batch']);
    expect(eventDetails(ev(1, { namespace: 'ns' }))).toEqual(['ns']);
  });

  it('is empty when nothing applies', () => {
    expect(eventDetails(ev(1))).toEqual([]);
  });
});
