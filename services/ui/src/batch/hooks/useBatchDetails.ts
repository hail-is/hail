import { useState, useEffect, useRef, useCallback } from 'react';
import type { Batch, BatchJob, BillingProjectInfo } from '../../shared/batchApi';
import { createHailApi } from '../../shared/batchApi';

export interface UseBatchDetailsResult {
  batch: Batch | null;
  jobs: BatchJob[] | null;
  lastJobId: number | undefined;
  billingProjectInfo: BillingProjectInfo | null;
  error: string | null;
  loading: boolean;
  autoRefresh: boolean;
  setAutoRefresh: (_v: boolean) => void;
  countdownKey: number;
  refreshIntervalMs: number;
  batchRefreshing: boolean;
  jobsLoading: boolean;
  q: string;
  hasPreviousPage: boolean;
  setSearch: (q: string) => void;
  goToNextPage: () => void;
  goToPreviousPage: () => void;
  cancelBatch: () => Promise<void>;
  deleteBatch: () => Promise<void>;
}

export const REFRESH_INTERVAL_MS = 30_000;

export function useBatchDetails(basePath: string, batchId: string): UseBatchDetailsResult {
  const api = useRef(createHailApi(basePath)).current;

  const getInitialPage = () => {
    const params = new URLSearchParams(window.location.search);
    return {
      q: params.get('q') ?? '',
      lastJobId: params.get('last_job_id') != null ? Number(params.get('last_job_id')) : undefined,
    };
  };

  const [{ q, lastJobId: pageLastJobId }, setPage] = useState(getInitialPage);

  const [batch, setBatch] = useState<Batch | null>(null);
  const [jobs, setJobs] = useState<BatchJob[] | null>(null);
  const [lastJobId, setLastJobId] = useState<number | undefined>(undefined);
  const [billingProjectInfo, setBillingProjectInfo] = useState<BillingProjectInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [batchRefreshing, setBatchRefreshing] = useState(false);
  const [jobsLoading, setJobsLoading] = useState(false);
  const isInitialMount = useRef(true);
  const [countdownKey, setCountdownKey] = useState(0);
  const [autoRefresh, setAutoRefreshState] = useState<boolean>(() => {
    try {
      return localStorage.getItem('batch.batchPage.autoRefresh') !== 'false';
    } catch {
      return true;
    }
  });

  const setAutoRefresh = useCallback((v: boolean) => {
    setAutoRefreshState(v);
    try {
      localStorage.setItem('batch.batchPage.autoRefresh', String(v));
    } catch { /* ignore */ }
  }, []);

  // 'initial' clears `loading` as soon as the batch summary resolves, independent of the jobs
  // list (which shows its own spinner via `jobsLoading`). 'search'/'refresh' must not touch
  // `loading`, or the whole page would remount on every search/pagination click.
  const fetchData = useCallback(async (
    kind: 'initial' | 'refresh' | 'search', currentQ: string, currentLastJobId: number | undefined,
  ) => {
    if (kind === 'refresh') setBatchRefreshing(true);
    if (kind === 'initial' || kind === 'search') setJobsLoading(true);

    const batchPromise = api.getBatch(batchId).then((batchData) => {
      setBatch(batchData);
      if (kind === 'initial') {
        setLoading(false);
        api.getBillingProject(batchData.billing_project)
          .then(setBillingProjectInfo)
          .catch(() => { setBillingProjectInfo(null); });
      }
      setCountdownKey((k) => k + 1);
    });

    const jobsPromise = api.getBatchJobs(batchId, { q: currentQ, lastJobId: currentLastJobId }).then((jobsPage) => {
      setJobs(jobsPage.jobs);
      setLastJobId(jobsPage.last_job_id);
    });

    try {
      await Promise.all([batchPromise, jobsPromise]);
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
      if (kind === 'refresh') setBatchRefreshing(false);
      if (kind === 'initial' || kind === 'search') setJobsLoading(false);
    }
  }, [api, batchId]);

  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false;
      setLoading(true);
      void fetchData('initial', q, pageLastJobId);
    } else {
      void fetchData('search', q, pageLastJobId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q, pageLastJobId]);

  // Auto-refresh: re-fetch the current page for as long as the batch isn't complete.
  useEffect(() => {
    if (!batch || batch.complete || !autoRefresh) return;
    const id = setInterval(() => { void fetchData('refresh', q, pageLastJobId); }, REFRESH_INTERVAL_MS);
    return () => { clearInterval(id); };
  }, [batch, fetchData, autoRefresh, q, pageLastJobId]);

  const syncUrl = (newQ: string, newLastJobId: number | undefined) => {
    const params = new URLSearchParams(window.location.search);
    if (newQ) params.set('q', newQ); else params.delete('q');
    if (newLastJobId != null) params.set('last_job_id', String(newLastJobId)); else params.delete('last_job_id');
    window.history.replaceState(null, '', `?${params.toString()}`);
  };

  const setSearch = useCallback((newQ: string) => {
    setPage({ q: newQ, lastJobId: undefined });
    syncUrl(newQ, undefined);
  }, []);

  const goToNextPage = useCallback(() => {
    if (lastJobId == null) return;
    setPage({ q, lastJobId });
    syncUrl(q, lastJobId);
  }, [q, lastJobId]);

  // Step the cursor that fetched the current page back by 50, rather than deriving it from the
  // lowest job_id actually returned — job_id isn't dense (job-group structure), so that drifts.
  const hasPreviousPage = pageLastJobId != null && pageLastJobId > 0;

  const goToPreviousPage = useCallback(() => {
    if (pageLastJobId == null) return;
    const prevLastJobId = Math.max(0, pageLastJobId - 50);
    setPage({ q, lastJobId: prevLastJobId > 0 ? prevLastJobId : undefined });
    syncUrl(q, prevLastJobId > 0 ? prevLastJobId : undefined);
  }, [q, pageLastJobId]);

  const cancelBatch = useCallback(async () => {
    await api.cancelBatch(batchId);
    await fetchData('refresh', q, pageLastJobId);
  }, [api, batchId, fetchData, q, pageLastJobId]);

  const deleteBatch = useCallback(async () => {
    await api.deleteBatch(batchId);
  }, [api, batchId]);

  return {
    batch,
    jobs,
    lastJobId,
    billingProjectInfo,
    error,
    loading,
    autoRefresh,
    setAutoRefresh,
    countdownKey,
    refreshIntervalMs: REFRESH_INTERVAL_MS,
    batchRefreshing,
    jobsLoading,
    q,
    hasPreviousPage,
    setSearch,
    goToNextPage,
    goToPreviousPage,
    cancelBatch,
    deleteBatch,
  };
}
