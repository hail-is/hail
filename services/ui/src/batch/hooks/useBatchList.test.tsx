import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useBatchList, REFRESH_INTERVAL_MS } from './useBatchList';

const BASE_PATH = '/batch';

function batch(id: number, complete = true) {
  return {
    id, user: 'alice', billing_project: 'proj', state: complete ? 'success' : 'running', complete,
    n_jobs: 1, n_completed: complete ? 1 : 0, n_succeeded: complete ? 1 : 0, n_failed: 0, n_cancelled: 0, cost: 0,
  };
}

// Cursor -> page served. Pages are deliberately sparse so a cursor can't be guessed by arithmetic.
const PAGES = new Map<string | null, object>([
  [null, { batches: [batch(900), batch(870)], last_batch_id: 870 }],
  ['870', { batches: [batch(500), batch(420)], last_batch_id: 420 }],
  ['420', { batches: [batch(7)] }],
]);

let fetchMock: ReturnType<typeof vi.fn>;

function requestedParams(callIndex: number): URLSearchParams {
  const url = fetchMock.mock.calls[callIndex][0] as string;
  return new URL(url, 'http://localhost').searchParams;
}

async function flush() {
  await act(async () => {
    for (let i = 0; i < 5; i++) await Promise.resolve();
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  window.history.replaceState(null, '', '/batches');
  fetchMock = vi.fn((url: string) => {
    const params = new URL(url, 'http://localhost').searchParams;
    const page = PAGES.get(params.get('last_batch_id'));
    return Promise.resolve({ ok: true, text: () => Promise.resolve(JSON.stringify(page)) });
  });
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('query defaults', () => {
  it('defaults to the current user when the URL has no q', async () => {
    const { result } = renderHook(() => useBatchList(BASE_PATH, 'alice'));
    await flush();
    expect(result.current.q).toBe('user = alice');
    expect(requestedParams(0).get('q')).toBe('user = alice');
  });

  it('sends an explicitly empty q (rather than omitting it) so the server does not apply its default', async () => {
    window.history.replaceState(null, '', '/batches?q=');
    const { result } = renderHook(() => useBatchList(BASE_PATH, 'alice'));
    await flush();
    expect(result.current.q).toBe('');
    expect(requestedParams(0).has('q')).toBe(true);
    expect(requestedParams(0).get('q')).toBe('');
  });
});

describe('pagination', () => {
  it('steps forward and back through the exact cursors it was served', async () => {
    const { result } = renderHook(() => useBatchList(BASE_PATH, 'alice'));
    await flush();
    expect(result.current.isFirstPage).toBe(true);
    expect(result.current.hasPreviousPage).toBe(false);

    act(() => { result.current.goToNextPage(); });
    await flush();
    expect(result.current.batches?.map((b) => b.id)).toEqual([500, 420]);

    act(() => { result.current.goToNextPage(); });
    await flush();
    expect(result.current.batches?.map((b) => b.id)).toEqual([7]);
    expect(result.current.hasNextPage).toBe(false);

    act(() => { result.current.goToPreviousPage(); });
    await flush();
    expect(result.current.batches?.map((b) => b.id)).toEqual([500, 420]);

    act(() => { result.current.goToPreviousPage(); });
    await flush();
    expect(result.current.batches?.map((b) => b.id)).toEqual([900, 870]);
    expect(result.current.isFirstPage).toBe(true);
  });

  it('a deep link has no previous page, only a first page', async () => {
    window.history.replaceState(null, '', '/batches?q=&last_batch_id=870');
    const { result } = renderHook(() => useBatchList(BASE_PATH, 'alice'));
    await flush();
    expect(result.current.isFirstPage).toBe(false);
    expect(result.current.hasPreviousPage).toBe(false);
  });

  it('a new search resets to the first page', async () => {
    const { result } = renderHook(() => useBatchList(BASE_PATH, 'alice'));
    await flush();
    act(() => { result.current.goToNextPage(); });
    await flush();
    act(() => { result.current.setSearch('billing_project = proj'); });
    await flush();
    expect(result.current.isFirstPage).toBe(true);
    expect(new URLSearchParams(window.location.search).get('last_batch_id')).toBeNull();
    expect(new URLSearchParams(window.location.search).get('q')).toBe('billing_project = proj');
  });
});

describe('auto-refresh', () => {
  it('stops refreshing a later page once every batch on it is complete', async () => {
    window.history.replaceState(null, '', '/batches?q=&last_batch_id=870');
    const { result } = renderHook(() => useBatchList(BASE_PATH, 'alice'));
    await flush();
    expect(result.current.refreshActive).toBe(false);
    const calls = fetchMock.mock.calls.length;
    await act(async () => { vi.advanceTimersByTime(REFRESH_INTERVAL_MS * 2); });
    expect(fetchMock.mock.calls.length).toBe(calls);
  });

  it('keeps refreshing the first page, where new batches appear', async () => {
    const { result } = renderHook(() => useBatchList(BASE_PATH, 'alice'));
    await flush();
    expect(result.current.refreshActive).toBe(true);
    const calls = fetchMock.mock.calls.length;
    await act(async () => { vi.advanceTimersByTime(REFRESH_INTERVAL_MS); });
    expect(fetchMock.mock.calls.length).toBe(calls + 1);
  });
});
