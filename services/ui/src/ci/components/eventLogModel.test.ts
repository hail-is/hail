import { describe, expect, it } from 'vitest';
import type { CIEvent } from '../../shared/ciApi';
import { collapseRepeats, currentMergeFailures, eventDetails } from './eventLogModel';

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
    is_error: true,
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

describe('currentMergeFailures', () => {
  const branch = 'hail-is/hail:main';
  const failed = (id: number, overrides: Partial<CIEvent> = {}) => ev(id, {
    event: 'merge_failed', pr_number: 1, source_sha: 'src', target_sha: 'tgt', message: `fail ${id}`, ...overrides,
  });

  it('picks the newest failure for the current attempt', () => {
    const result = currentMergeFailures([failed(3), failed(2)], branch, 'tgt', [{ pr: 1, source_sha: 'src' }]);
    expect(result.get(1)?.id).toBe(3);
  });

  it('ignores failures from an older source or target sha', () => {
    const rows = [failed(3, { source_sha: 'old' }), failed(2, { target_sha: 'old' })];
    expect(currentMergeFailures(rows, branch, 'tgt', [{ pr: 1, source_sha: 'src' }]).size).toBe(0);
  });

  it('ignores other branches, PRs and events', () => {
    const rows = [
      failed(4, { target_branch: 'hail-is/hail:other' }),
      failed(3, { pr_number: 2 }),
      failed(2, { event: 'merge_requested' }),
    ];
    expect(currentMergeFailures(rows, branch, 'tgt', [{ pr: 1, source_sha: 'src' }]).size).toBe(0);
  });
});
