import { useState, useEffect, useRef, useCallback } from 'react';
import type { Batch } from '../../shared/batchApi';
import { createHailApi } from '../../shared/batchApi';

export interface UseBatchListResult {
  batches: Batch[] | null;
  error: string | null;
  loading: boolean;
  batchesLoading: boolean;
  refreshing: boolean;
  autoRefresh: boolean;
  setAutoRefresh: (_v: boolean) => void;
  // False once nothing on the current page can change: not the first page (where new batches
  // appear) and every visible batch is complete.
  refreshActive: boolean;
  countdownKey: number;
  refreshIntervalMs: number;
  q: string;
  hasPreviousPage: boolean;
  isFirstPage: boolean;
  hasNextPage: boolean;
  setSearch: (_q: string) => void;
  goToNextPage: () => void;
  goToPreviousPage: () => void;
  goToFirstPage: () => void;
}

export const REFRESH_INTERVAL_MS = 30_000;

interface Page {
  q: string;
  lastBatchId: number | undefined;
  // Cursors of the pages before this one, so "Previous" can step back exactly. Batch ids are
  // sparse per-user, so a cursor can't be derived by arithmetic as the jobs list does.
  previousCursors: (number | undefined)[];
}

export function defaultBatchQuery(username: string): string {
  return username ? `user = ${username}` : '';
}

// Unlike the jobs list, an absent `q` and an empty `q` differ here: absent means "the default
// (your own batches)", empty means "no filters" — so an empty q is kept in the URL explicitly.
function syncUrl(newQ: string, newLastBatchId: number | undefined): void {
  const params = new URLSearchParams(window.location.search);
  params.set('q', newQ);
  if (newLastBatchId != null) params.set('last_batch_id', String(newLastBatchId)); else params.delete('last_batch_id');
  window.history.replaceState(null, '', `?${params.toString()}`);
}

export function useBatchList(basePath: string, username: string): UseBatchListResult {
  const api = useRef(createHailApi(basePath)).current;

  const getInitialPage = (): Page => {
    const params = new URLSearchParams(window.location.search);
    const rawLast = params.get('last_batch_id');
    return {
      q: params.get('q') ?? defaultBatchQuery(username),
      lastBatchId: rawLast != null ? parseInt(rawLast, 10) : undefined,
      previousCursors: [],
    };
  };

  const [page, setPage] = useState<Page>(getInitialPage);
  const { q, lastBatchId: pageLastBatchId } = page;

  const [batches, setBatches] = useState<Batch[] | null>(null);
  const [nextLastBatchId, setNextLastBatchId] = useState<number | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [batchesLoading, setBatchesLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [countdownKey, setCountdownKey] = useState(0);
  const [autoRefresh, setAutoRefreshState] = useState<boolean>(() => {
    try {
      return localStorage.getItem('batch.batchesPage.autoRefresh') !== 'false';
    } catch {
      return true;
    }
  });

  const setAutoRefresh = useCallback((v: boolean) => {
    setAutoRefreshState(v);
    try {
      localStorage.setItem('batch.batchesPage.autoRefresh', String(v));
    } catch { /* ignore */ }
  }, []);

  // Responses to a superseded request (e.g. a slow refresh landing after the user has already
  // searched again) are dropped rather than overwriting the newer page.
  const requestSeq = useRef(0);

  const fetchData = useCallback(async (
    kind: 'search' | 'refresh', currentQ: string, currentLastBatchId: number | undefined,
  ) => {
    const seq = ++requestSeq.current;
    if (kind === 'refresh') setRefreshing(true); else setBatchesLoading(true);
    try {
      const result = await api.getBatches({ q: currentQ, lastBatchId: currentLastBatchId });
      if (seq !== requestSeq.current) return;
      setBatches(result.batches);
      setNextLastBatchId(result.last_batch_id);
      setError(null);
    } catch (e) {
      if (seq !== requestSeq.current) return;
      setError(String(e));
    } finally {
      if (seq === requestSeq.current) {
        setLoading(false);
        setRefreshing(false);
        setBatchesLoading(false);
        setCountdownKey((k) => k + 1);
      }
    }
  }, [api]);

  useEffect(() => {
    void fetchData('search', q, pageLastBatchId);
  }, [q, pageLastBatchId, fetchData]);

  const isFirstPage = pageLastBatchId == null;
  const refreshActive = batches != null && (isFirstPage || batches.some((b) => !b.complete));

  useEffect(() => {
    if (!autoRefresh || !refreshActive) return;
    const id = setInterval(() => { void fetchData('refresh', q, pageLastBatchId); }, REFRESH_INTERVAL_MS);
    return () => { clearInterval(id); };
  }, [autoRefresh, refreshActive, fetchData, q, pageLastBatchId]);

  const setSearch = useCallback((newQ: string) => {
    setPage({ q: newQ, lastBatchId: undefined, previousCursors: [] });
    syncUrl(newQ, undefined);
  }, []);

  const goToNextPage = useCallback(() => {
    if (nextLastBatchId == null) return;
    setPage((p) => ({ q: p.q, lastBatchId: nextLastBatchId, previousCursors: [...p.previousCursors, p.lastBatchId] }));
    syncUrl(q, nextLastBatchId);
  }, [q, nextLastBatchId]);

  const goToPreviousPage = useCallback(() => {
    const prev = page.previousCursors.at(-1);
    if (page.previousCursors.length === 0) return;
    setPage({ q, lastBatchId: prev, previousCursors: page.previousCursors.slice(0, -1) });
    syncUrl(q, prev);
  }, [q, page.previousCursors]);

  const goToFirstPage = useCallback(() => {
    setPage({ q, lastBatchId: undefined, previousCursors: [] });
    syncUrl(q, undefined);
  }, [q]);

  return {
    batches,
    error,
    loading,
    batchesLoading,
    refreshing,
    autoRefresh,
    setAutoRefresh,
    refreshActive,
    countdownKey,
    refreshIntervalMs: REFRESH_INTERVAL_MS,
    q,
    hasPreviousPage: page.previousCursors.length > 0,
    isFirstPage,
    hasNextPage: nextLastBatchId != null,
    setSearch,
    goToNextPage,
    goToPreviousPage,
    goToFirstPage,
  };
}
