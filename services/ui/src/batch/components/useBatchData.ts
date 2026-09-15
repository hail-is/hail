import { useCallback, useRef, useState } from 'react';
import { hailApiFetch as apiFetch } from '../../shared/hailApiFetch';

// Generic Batch-domain data for a single batch: its own status, its full (recursive) job list,
// and its job-group hierarchy. Deliberately has no CI-specific (or any other service-specific)
// knowledge — any page showing a batch's jobs/job-groups (the CI PR page today, a future React
// batch-details page tomorrow) should use this instead of writing its own fetch/cache logic.
// See dev-docs/services/ui/README.md's "Data fetching for a batch" section for the pattern this
// is meant to establish.

export type JobState = 'Pending' | 'Ready' | 'Creating' | 'Running' | 'Failed' | 'Cancelled' | 'Error' | 'Success';

export interface JobListEntry {
  job_id: number;
  job_group_id: number;
  name: string | null;
  state: JobState;
  exit_code: number | null;
}

// Mirrors job_group_record_to_dict (batch/batch/batch.py) — this is what
// GET .../job-groups/{id} and GET .../job-groups/{id}/job-groups actually return. Note this does
// NOT match the openapi.yaml JobGroupDetailResponse schema, which documents a `children` field
// and a `parent_job_group_id` field that the real handler never sets — that schema is
// aspirational/stale, not a contract the backend honors today.
export interface JobGroupSummary {
  job_group_id: number;
  n_jobs: number;
  n_completed: number;
  n_succeeded: number;
  n_failed: number;
  n_cancelled: number;
  attributes?: Record<string, string>;
}

export interface BatchStatus {
  id: number;
  state: string;
  complete: boolean;
  cost: number | null;
  n_jobs: number;
  n_completed: number;
  n_succeeded: number;
  n_failed: number;
  n_cancelled: number;
  time_created: string | null;
  time_completed: string | null;
  attributes?: Record<string, string>;
}

// A batch always has an implicit root job group (id 0); every job lands there unless the
// submitter explicitly nests it in a sub-group via the hailtop.batch job-group API.
export const ROOT_JOB_GROUP_ID = 0;

async function fetchAllJobs(batchBaseUrl: string, batchId: number): Promise<JobListEntry[]> {
  const all: JobListEntry[] = [];
  let lastId: number | undefined;
  for (;;) {
    const url = new URL(`${batchBaseUrl}/api/v1alpha/batches/${batchId}/jobs`, window.location.origin);
    if (lastId !== undefined) url.searchParams.set('last_job_id', String(lastId));
    const page = await apiFetch<{ jobs: JobListEntry[]; last_job_id?: number }>(url.toString());
    all.push(...page.jobs);
    if (page.last_job_id === undefined) break;
    lastId = page.last_job_id;
  }
  return all;
}

async function fetchJobGroups(batchBaseUrl: string, batchId: number, parentJobGroupId: number): Promise<JobGroupSummary[]> {
  const all: JobGroupSummary[] = [];
  let lastId: number | undefined;
  for (;;) {
    const url = new URL(`${batchBaseUrl}/api/v1alpha/batches/${batchId}/job-groups/${parentJobGroupId}/job-groups`, window.location.origin);
    if (lastId !== undefined) url.searchParams.set('last_job_group_id', String(lastId));
    const page = await apiFetch<{ job_groups: JobGroupSummary[]; last_job_group_id?: number }>(url.toString());
    all.push(...page.job_groups);
    if (page.last_job_group_id === undefined) break;
    lastId = page.last_job_group_id;
  }
  return all;
}

export interface UseBatchDataResult {
  batchStatus: BatchStatus | null;
  jobs: JobListEntry[] | null;
  jobsError: string | null;
  refreshing: boolean;
  // Re-fetches batchStatus + the full job list. Call once on mount and on whatever interval/
  // toggle drives the page's auto-refresh UI — this hook has no polling opinion of its own,
  // since when/whether to poll is page-level UI policy, not a data-fetching concern.
  refresh: (isRefresh: boolean) => Promise<void>;
  // The root job group's rollup counts are the batch's own aggregate counts (root job group IS
  // the batch, not a separate fetched entity) — derived from batchStatus, no extra request.
  rootJobGroup: JobGroupSummary | null;
  // Jobs directly owned by a given job group — synchronous, filtered from the already-fetched
  // recursive job list. Never triggers a request.
  getJobs: (parentJobGroupId: number) => JobListEntry[];
  // Sub-job-groups of a given job group. Returns undefined until fetchJobGroups(parentJobGroupId)
  // has been called and resolved at least once.
  getJobGroups: (parentJobGroupId: number) => JobGroupSummary[] | undefined;
  getJobGroupsError: (parentJobGroupId: number) => string | undefined;
  // Fetches (once) and caches a job group's children, keyed by parentJobGroupId, for the
  // lifetime of this hook instance — i.e. for as long as the page stays mounted. Safe to call
  // repeatedly (e.g. on every row expand): a second call while a fetch is in flight or after one
  // has already succeeded is a no-op. A failed fetch is not cached, so a later call retries.
  fetchJobGroups: (parentJobGroupId: number) => void;
}

export function useBatchData(batchBaseUrl: string, batchId: number | undefined): UseBatchDataResult {
  const [batchStatus, setBatchStatus] = useState<BatchStatus | null>(null);
  const [jobs, setJobs] = useState<JobListEntry[] | null>(null);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const [jobGroupsByParent, setJobGroupsByParent] = useState<Map<number, JobGroupSummary[]>>(new Map());
  const [jobGroupErrorsByParent, setJobGroupErrorsByParent] = useState<Map<number, string>>(new Map());
  // Tracks in-flight/succeeded fetches so fetchJobGroups is idempotent without waiting for a
  // state update to land — a ref (not state) because it's bookkeeping, not something that
  // should itself trigger a re-render.
  const jobGroupFetchesStarted = useRef<Set<number>>(new Set());

  const refresh = useCallback(async (isRefresh: boolean) => {
    if (batchId === undefined) return;
    if (isRefresh) setRefreshing(true);
    try {
      const [status, allJobs] = await Promise.all([
        apiFetch<BatchStatus>(`${batchBaseUrl}/api/v1alpha/batches/${batchId}`),
        fetchAllJobs(batchBaseUrl, batchId),
      ]);
      setBatchStatus(status);
      setJobs(allJobs);
      setJobsError(null);
    } catch (e: unknown) {
      setJobsError(e instanceof Error ? e.message : String(e));
    } finally {
      if (isRefresh) setRefreshing(false);
    }
  }, [batchBaseUrl, batchId]);

  const fetchJobGroupsFor = useCallback((parentJobGroupId: number) => {
    if (batchId === undefined) return;
    if (jobGroupFetchesStarted.current.has(parentJobGroupId)) return;
    jobGroupFetchesStarted.current.add(parentJobGroupId);
    fetchJobGroups(batchBaseUrl, batchId, parentJobGroupId)
      .then((groups) => {
        setJobGroupsByParent((prev) => new Map(prev).set(parentJobGroupId, groups));
      })
      .catch((e: unknown) => {
        jobGroupFetchesStarted.current.delete(parentJobGroupId); // allow a retry
        setJobGroupErrorsByParent((prev) => new Map(prev).set(parentJobGroupId, e instanceof Error ? e.message : String(e)));
      });
  }, [batchBaseUrl, batchId]);

  const getJobs = useCallback(
    (parentJobGroupId: number): JobListEntry[] => (jobs ?? []).filter((j) => j.job_group_id === parentJobGroupId),
    [jobs],
  );

  const getJobGroups = useCallback(
    (parentJobGroupId: number): JobGroupSummary[] | undefined => jobGroupsByParent.get(parentJobGroupId),
    [jobGroupsByParent],
  );

  const getJobGroupsError = useCallback(
    (parentJobGroupId: number): string | undefined => jobGroupErrorsByParent.get(parentJobGroupId),
    [jobGroupErrorsByParent],
  );

  const rootJobGroup: JobGroupSummary | null = batchStatus
    ? {
        job_group_id: ROOT_JOB_GROUP_ID,
        n_jobs: batchStatus.n_jobs,
        n_completed: batchStatus.n_completed,
        n_succeeded: batchStatus.n_succeeded,
        n_failed: batchStatus.n_failed,
        n_cancelled: batchStatus.n_cancelled,
        attributes: batchStatus.attributes,
      }
    : null;

  return {
    batchStatus,
    jobs,
    jobsError,
    refreshing,
    refresh,
    rootJobGroup,
    getJobs,
    getJobGroups,
    getJobGroupsError,
    fetchJobGroups: fetchJobGroupsFor,
  };
}
