import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useJobDetails, REFRESH_INTERVAL_MS } from './useJobDetails';

const BASE_PATH = '/batch';
const BATCH_ID = '1';
const JOB_ID = '1';
const ATTEMPT_LATEST = 'attempt-1';
const ATTEMPT_OLDER = 'attempt-0';

const mockJob = { id: 1, batch_id: 1, state: 'Running', spec: {} };
const mockTerminalJob = { id: 1, batch_id: 1, state: 'Success', spec: {} };
const mockAttempts = [{ attempt_id: ATTEMPT_OLDER }, { attempt_id: ATTEMPT_LATEST }];
const LOG_V1 = 'log text v1';
const LOG_V2 = 'log text v2 — updated';
const JOB_URL = `${BASE_PATH}/api/v1alpha/batches/${BATCH_ID}/jobs/${JOB_ID}`;

type MockFetchOpts = { log?: string; job?: object };

function makeFetch({ log = LOG_V1, job = mockJob }: MockFetchOpts = {}) {
  return vi.fn((url: string) => {
    const u = url as string;
    if (u === `${JOB_URL}/attempts`) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(mockAttempts) });
    }
    if (u === JOB_URL) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(job) });
    }
    if (u.includes('/log/')) {
      return Promise.resolve({ ok: true, text: () => Promise.resolve(log) });
    }
    if (u.includes('/resource_usage')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(null) });
    }
    return Promise.resolve({ ok: false, status: 404 });
  });
}

// Flush all pending microtasks/promises inside act
async function flush() {
  await act(async () => {
    // Two rounds handle chained promise chains (fetch → json → setState)
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal('fetch', makeFetch());
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------
// T1 — Initial load populates job; attempt data is lazy (not yet loaded)
// ---------------------------------------------------------------------------
describe('T1 — initial load', () => {
  it('populates job and attempts; attempt data is empty until ensureAttemptLoaded', async () => {
    const { result } = renderHook(() => useJobDetails(BASE_PATH, BATCH_ID, JOB_ID));
    expect(result.current.loading).toBe(true);

    await flush();

    expect(result.current.loading).toBe(false);
    expect(result.current.job).not.toBeNull();
    expect(result.current.job?.state).toBe('Running');
    expect(result.current.attempts).toHaveLength(2);
    // Attempt data is lazy — not loaded yet
    expect(result.current.getAttemptData(ATTEMPT_LATEST).loading).toBe(false);
    expect(result.current.getAttemptData(ATTEMPT_LATEST).logs).toEqual({});
  });
});

// ---------------------------------------------------------------------------
// T2 — ensureAttemptLoaded fetches and caches
// ---------------------------------------------------------------------------
describe('T2 — ensureAttemptLoaded fetches and caches', () => {
  it('fetches on cache miss and stores result; second call is a no-op', async () => {
    const fetchSpy = makeFetch({ log: LOG_V1 });
    vi.stubGlobal('fetch', fetchSpy);

    const { result } = renderHook(() => useJobDetails(BASE_PATH, BATCH_ID, JOB_ID));
    await flush(); // initial job/attempts load

    act(() => result.current.ensureAttemptLoaded(ATTEMPT_LATEST));
    await flush();

    expect(result.current.getAttemptData(ATTEMPT_LATEST).logs.main).toBe(LOG_V1);
    expect(result.current.getAttemptData(ATTEMPT_LATEST).loading).toBe(false);

    const callsBefore = fetchSpy.mock.calls.filter(([u]) => (u as string).includes('/log/')).length;

    // Second call — should be a cache hit, no new fetch
    act(() => result.current.ensureAttemptLoaded(ATTEMPT_LATEST));
    await flush();

    const callsAfter = fetchSpy.mock.calls.filter(([u]) => (u as string).includes('/log/')).length;
    expect(callsAfter).toBe(callsBefore);
  });
});

// ---------------------------------------------------------------------------
// T3 — Cache hit: re-calling ensureAttemptLoaded never double-fetches
// ---------------------------------------------------------------------------
describe('T3 — no double-fetch on cache hit', () => {
  it('fires the log endpoint exactly once even with repeated calls', async () => {
    const fetchSpy = makeFetch();
    vi.stubGlobal('fetch', fetchSpy);

    const { result } = renderHook(() => useJobDetails(BASE_PATH, BATCH_ID, JOB_ID));
    await flush();

    act(() => result.current.ensureAttemptLoaded(ATTEMPT_LATEST));
    await flush();

    const logCalls = () => fetchSpy.mock.calls.filter(([u]) => (u as string).includes('/log/')).length;
    const after1 = logCalls();

    act(() => result.current.ensureAttemptLoaded(ATTEMPT_LATEST));
    await flush();
    act(() => result.current.ensureAttemptLoaded(ATTEMPT_LATEST));
    await flush();

    expect(logCalls()).toBe(after1);
  });
});

// ---------------------------------------------------------------------------
// T4 — Refresh busts only the latest attempt
// ---------------------------------------------------------------------------
describe('T4 — refresh only re-fetches the latest attempt', () => {
  it('re-fetches latest attempt log on timer tick but not the older attempt', async () => {
    const fetchSpy = makeFetch();
    vi.stubGlobal('fetch', fetchSpy);

    const { result } = renderHook(() => useJobDetails(BASE_PATH, BATCH_ID, JOB_ID));
    await flush();

    // Prime both attempts into the cache
    act(() => result.current.ensureAttemptLoaded(ATTEMPT_LATEST));
    act(() => result.current.ensureAttemptLoaded(ATTEMPT_OLDER));
    await flush();

    const logUrlFor = (id: string) => `${JOB_URL}/log/main?attempt_id=${id}`;
    const countCalls = (url: string) => fetchSpy.mock.calls.filter(([u]) => u === url).length;

    const latestBefore = countCalls(logUrlFor(ATTEMPT_LATEST));
    const olderBefore = countCalls(logUrlFor(ATTEMPT_OLDER));

    // Advance timer to trigger auto-refresh
    await act(async () => {
      vi.advanceTimersByTime(REFRESH_INTERVAL_MS + 100);
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(countCalls(logUrlFor(ATTEMPT_LATEST))).toBeGreaterThan(latestBefore);
    expect(countCalls(logUrlFor(ATTEMPT_OLDER))).toBe(olderBefore);
  });
});

// ---------------------------------------------------------------------------
// T5 — Pending-logs banner when refresh returns new content
// ---------------------------------------------------------------------------
describe('T5 — pending logs banner on changed content', () => {
  it('shows hasPendingLogs when refresh changes logs; commit clears it', async () => {
    let currentLog = LOG_V1;
    const fetchSpy = vi.fn((url: string) => {
      const u = url as string;
      if (u === `${JOB_URL}/attempts`) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(mockAttempts) });
      }
      if (u === JOB_URL) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(mockJob) });
      }
      if (u.includes('/log/')) {
        return Promise.resolve({ ok: true, text: () => Promise.resolve(currentLog) });
      }
      if (u.includes('/resource_usage')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(null) });
      }
      return Promise.resolve({ ok: false, status: 404 });
    });
    vi.stubGlobal('fetch', fetchSpy);

    const { result } = renderHook(() => useJobDetails(BASE_PATH, BATCH_ID, JOB_ID));
    await flush();

    act(() => result.current.ensureAttemptLoaded(ATTEMPT_LATEST));
    await flush();

    expect(result.current.getAttemptData(ATTEMPT_LATEST).committedLogs.main).toBe(LOG_V1);
    expect(result.current.getAttemptData(ATTEMPT_LATEST).hasPendingLogs).toBe(false);

    // Change logs so the next refresh returns different content
    currentLog = LOG_V2;

    await act(async () => {
      vi.advanceTimersByTime(REFRESH_INTERVAL_MS + 100);
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(result.current.getAttemptData(ATTEMPT_LATEST).hasPendingLogs).toBe(true);
    expect(result.current.getAttemptData(ATTEMPT_LATEST).committedLogs.main).toBe(LOG_V1);

    act(() => result.current.commitAttemptLogs(ATTEMPT_LATEST));

    expect(result.current.getAttemptData(ATTEMPT_LATEST).hasPendingLogs).toBe(false);
    expect(result.current.getAttemptData(ATTEMPT_LATEST).committedLogs.main).toBe(LOG_V2);
  });
});

// ---------------------------------------------------------------------------
// T6 — No pending banner when logs unchanged on refresh
// ---------------------------------------------------------------------------
describe('T6 — no pending logs banner when content unchanged', () => {
  it('hasPendingLogs stays false when refresh returns same log text', async () => {
    const fetchSpy = makeFetch({ log: LOG_V1 });
    vi.stubGlobal('fetch', fetchSpy);

    const { result } = renderHook(() => useJobDetails(BASE_PATH, BATCH_ID, JOB_ID));
    await flush();

    act(() => result.current.ensureAttemptLoaded(ATTEMPT_LATEST));
    await flush();

    await act(async () => {
      vi.advanceTimersByTime(REFRESH_INTERVAL_MS + 100);
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(result.current.getAttemptData(ATTEMPT_LATEST).hasPendingLogs).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// T7 — Terminal job stops auto-refresh
// ---------------------------------------------------------------------------
describe('T7 — terminal job stops auto-refresh', () => {
  it('does not re-fetch after terminal state is received', async () => {
    const fetchSpy = makeFetch({ job: mockTerminalJob });
    vi.stubGlobal('fetch', fetchSpy);

    const { result } = renderHook(() => useJobDetails(BASE_PATH, BATCH_ID, JOB_ID));
    await flush();

    expect(result.current.job?.state).toBe('Success');

    const jobCallsBefore = fetchSpy.mock.calls.filter(([u]) => u === JOB_URL).length;

    await act(async () => {
      vi.advanceTimersByTime(REFRESH_INTERVAL_MS * 3);
      await Promise.resolve();
      await Promise.resolve();
    });

    const jobCallsAfter = fetchSpy.mock.calls.filter(([u]) => u === JOB_URL).length;
    expect(jobCallsAfter).toBe(jobCallsBefore);
  });
});

// ---------------------------------------------------------------------------
// T8 — Proactive prefetch: attempt data ready without ensureAttemptLoaded
// ---------------------------------------------------------------------------
describe('T8 — proactive prefetch on refresh', () => {
  it('populates latest attempt data after a refresh without explicit ensureAttemptLoaded', async () => {
    const fetchSpy = makeFetch({ log: LOG_V1 });
    vi.stubGlobal('fetch', fetchSpy);

    const { result } = renderHook(() => useJobDetails(BASE_PATH, BATCH_ID, JOB_ID));
    await flush(); // initial load — attempt data still empty (lazy)

    expect(result.current.getAttemptData(ATTEMPT_LATEST).logs).toEqual({});

    // Advance timer to trigger auto-refresh (which proactively fetches latest attempt)
    await act(async () => {
      vi.advanceTimersByTime(REFRESH_INTERVAL_MS + 100);
      // Extra flushes: fetchData awaits job+attempts, then fetchAttemptData awaits logs
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    // Data should now be populated without ever calling ensureAttemptLoaded
    expect(result.current.getAttemptData(ATTEMPT_LATEST).loading).toBe(false);
    expect(result.current.getAttemptData(ATTEMPT_LATEST).logs.main).toBe(LOG_V1);
  });
});
