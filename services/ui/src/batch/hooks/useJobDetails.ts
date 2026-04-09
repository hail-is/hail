import { useState, useEffect, useRef, useCallback } from 'react';
import { Job, Attempt, TERMINAL_STATES } from '../components/jobModels';
import { ResourceUsageData } from '../components/ResourceCharts';

export type LogMap = Record<string, string | null>;

export type AttemptCache = {
  logs: LogMap;
  resourceUsage: ResourceUsageData | null;
  loading: boolean;
  error: string | null;
  committedLogs: LogMap;
  hasPendingLogs: boolean;
  isRefreshing: boolean;  // background re-fetch in flight (not initial load)
};

export interface UseJobDetailsResult {
  job: Job | null;
  attempts: Attempt[] | null;
  error: string | null;
  loading: boolean;
  autoRefresh: boolean;
  setAutoRefresh: (_v: boolean) => void;
  countdownKey: number;
  refreshIntervalMs: number;
  jobRefreshing: boolean;
  // eslint-disable-next-line no-unused-vars
  getAttemptData: (_attemptId: string) => AttemptCache;
  // eslint-disable-next-line no-unused-vars
  ensureAttemptLoaded: (_attemptId: string) => void;
  commitAttemptLogs: (attemptId: string) => void;
}

export const REFRESH_INTERVAL_MS = 30_000;

export const DEFAULT_EMPTY_CACHE: AttemptCache = {
  logs: {},
  resourceUsage: null,
  loading: false,
  error: null,
  committedLogs: {},
  hasPendingLogs: false,
  isRefreshing: false,
};

async function fetchText(url: string): Promise<string> {
  const resp = await fetch(url, { credentials: 'same-origin' }); // nosemgrep: rules.lgpl.javascript.ssrf.rule-node-ssrf
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.text();
}

async function fetchJson<T>(url: string): Promise<T> {
  const resp = await fetch(url, { credentials: 'same-origin' }); // nosemgrep: rules.lgpl.javascript.ssrf.rule-node-ssrf
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json() as Promise<T>;
}

export function useJobDetails(basePath: string, batchId: string, jobId: string): UseJobDetailsResult {
  const [job, setJob] = useState<Job | null>(null);
  const [attempts, setAttempts] = useState<Attempt[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [jobRefreshing, setJobRefreshing] = useState(false);
  const [countdownKey, setCountdownKey] = useState(0);
  const [autoRefresh, setAutoRefreshState] = useState<boolean>(() => {
    try {
      return localStorage.getItem('batch.jobPage.autoRefresh') !== 'false';
    } catch {
      return true;
    }
  });

  const cache = useRef<Partial<Record<string, AttemptCache>>>({});
  const inFlight = useRef<Set<string>>(new Set());
  const jobRef = useRef<Job | null>(null);
  const [, setCacheTick] = useState(0);

  const bumpCache = useCallback(() => { setCacheTick((t) => t + 1); }, []);

  const setAutoRefresh = useCallback((v: boolean) => {
    setAutoRefreshState(v);
    try {
      localStorage.setItem('batch.jobPage.autoRefresh', String(v));
    } catch { /* ignore */ }
  }, []);

  const fetchAttemptData = useCallback((attemptId: string, isRefresh: boolean) => {
    if (inFlight.current.has(attemptId)) return;
    inFlight.current.add(attemptId);

    if (!isRefresh) {
      cache.current[attemptId] = { ...DEFAULT_EMPTY_CACHE, loading: true };
    } else {
      // eslint-disable-next-line security/detect-object-injection
      const prior = cache.current[attemptId];
      if (prior) {
        // eslint-disable-next-line security/detect-object-injection
        cache.current[attemptId] = { ...prior, isRefreshing: true };
      }
    }
    bumpCache();

    const currentJob = jobRef.current;
    const hasInput = (currentJob?.spec?.input_files ?? []).length > 0;
    const hasOutput = (currentJob?.spec?.output_files ?? []).length > 0;

    const attemptParam = `?attempt_id=${attemptId}`;
    const apiBase = `${basePath}/api/v1alpha/batches/${batchId}/jobs/${jobId}`;

    Promise.all([
      hasInput ? fetchText(`${apiBase}/log/input${attemptParam}`).catch(() => null) : Promise.resolve(null),
      fetchText(`${apiBase}/log/main${attemptParam}`).catch(() => null),
      hasOutput ? fetchText(`${apiBase}/log/output${attemptParam}`).catch(() => null) : Promise.resolve(null),
      fetchJson<ResourceUsageData>(`${apiBase}/resource_usage${attemptParam}`).catch(() => null),
    ]).then(([inputLog, mainLog, outputLog, resourceUsage]) => {
      const logs: LogMap = { input: inputLog, main: mainLog, output: outputLog };
      const existing = cache.current[attemptId];

      if (isRefresh && existing) {
        const changed = (['input', 'main', 'output'] as const).some(
          // eslint-disable-next-line security/detect-object-injection
          (k) => logs[k] !== existing.committedLogs[k]
        );
        cache.current[attemptId] = {
          ...existing,
          logs,
          resourceUsage,
          loading: false,
          error: null,
          hasPendingLogs: changed,
          isRefreshing: false,
        };
      } else {
        cache.current[attemptId] = {
          logs,
          resourceUsage,
          loading: false,
          error: null,
          committedLogs: logs,
          hasPendingLogs: false,
          isRefreshing: false,
        };
      }

      inFlight.current.delete(attemptId);
      bumpCache();
    }).catch(() => {
      // eslint-disable-next-line security/detect-object-injection
      const prior = cache.current[attemptId];
      if (prior) {
        // eslint-disable-next-line security/detect-object-injection
        cache.current[attemptId] = { ...prior, isRefreshing: false };
      }
      inFlight.current.delete(attemptId);
      bumpCache();
    });
  }, [basePath, batchId, jobId, bumpCache]);

  const fetchData = useCallback(async (isRefresh = false) => {
    const apiBase = `${basePath}/api/v1alpha/batches/${batchId}/jobs/${jobId}`;
    if (isRefresh) setJobRefreshing(true);
    try {
      const [jobResp, attemptsResp] = await Promise.all([
        fetch(apiBase, { credentials: 'same-origin' }), // nosemgrep: rules.lgpl.javascript.ssrf.rule-node-ssrf
        fetch(`${apiBase}/attempts`, { credentials: 'same-origin' }), // nosemgrep: rules.lgpl.javascript.ssrf.rule-node-ssrf
      ]);
      if (!jobResp.ok) throw new Error(`Job fetch: HTTP ${jobResp.status}`);
      const [jobData, attemptsData] = await Promise.all([
        jobResp.json() as Promise<Job>,
        attemptsResp.ok ? (attemptsResp.json() as Promise<Attempt[]>) : Promise.resolve(null),
      ]);

      jobRef.current = jobData;
      setJob(jobData);
      setAttempts(attemptsData);
      setError(null);

      // On refresh, proactively re-fetch the latest attempt's logs so data is
      // ready regardless of which tab the user is currently viewing.
      if (isRefresh && attemptsData && attemptsData.length > 0) {
        const latestAttempt = attemptsData[attemptsData.length - 1];
        fetchAttemptData(latestAttempt.attempt_id, true);
      }

      setCountdownKey((k) => k + 1);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
      if (isRefresh) setJobRefreshing(false);
    }
  }, [basePath, batchId, jobId, fetchAttemptData]);

  useEffect(() => {
    void fetchData(false);
  }, [fetchData]);

  // Auto-refresh for non-terminal jobs
  useEffect(() => {
    if (!job || TERMINAL_STATES.has(job.state) || !autoRefresh) return;
    const id = setInterval(() => { void fetchData(true); }, REFRESH_INTERVAL_MS);
    return () => { clearInterval(id); };
  }, [job, fetchData, autoRefresh]);

  const ensureAttemptLoaded = useCallback((attemptId: string) => {
    // eslint-disable-next-line security/detect-object-injection
    const entry = cache.current[attemptId];
    if (entry && !entry.loading) return;
    if (inFlight.current.has(attemptId)) return;
    fetchAttemptData(attemptId, false);
  }, [fetchAttemptData]);

  const commitAttemptLogs = useCallback((attemptId: string) => {
    const entry = cache.current[attemptId];
    if (!entry) return;
    // eslint-disable-next-line security/detect-object-injection
    cache.current[attemptId] = { ...entry, committedLogs: entry.logs, hasPendingLogs: false };
    bumpCache();
  }, [bumpCache]);

  const getAttemptData = useCallback((attemptId: string): AttemptCache => {
    // eslint-disable-next-line security/detect-object-injection
    return cache.current[attemptId] ?? DEFAULT_EMPTY_CACHE;
  }, []);

  return {
    job,
    attempts,
    error,
    loading,
    autoRefresh,
    setAutoRefresh,
    countdownKey,
    refreshIntervalMs: REFRESH_INTERVAL_MS,
    jobRefreshing,
    getAttemptData,
    ensureAttemptLoaded,
    commitAttemptLogs,
  };
}
