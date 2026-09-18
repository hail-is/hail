import { hailApiFetch } from './hailApiFetch';

// Wire types + client for the batch service's API. Colocated in shared/ rather than
// batch/components/ because the frontend treats the backend as one API surface — any service
// UI may need to call another service's API (e.g. monitoring's billing dashboard already calls
// batch's billing_breakdown endpoint), so a service's API contract lives in shared/, not with
// whichever page happens to call it first. See dev-docs/services/ui/README.md.

export interface CostBreakdownEntry {
  resource: string;
  cost: number;
}

export interface Batch {
  id: number;
  user: string;
  billing_project: string;
  state: 'open' | 'running' | 'success' | 'failure' | 'cancelled';
  complete: boolean;
  n_jobs: number;
  n_completed: number;
  n_succeeded: number;
  n_failed: number;
  n_cancelled: number;
  time_created?: string | null;
  time_completed?: string | null;
  duration?: string | null;
  cost: number;
  cost_breakdown?: CostBreakdownEntry[] | null;
  attributes?: Record<string, string>;
}

export interface BatchJob {
  batch_id: number;
  job_id: number;
  name: string | null;
  state: string;
  exit_code?: number | null;
  duration?: number | null;
  cost: number;
  always_run: boolean;
}

export interface BatchJobsPage {
  jobs: BatchJob[];
  last_job_id?: number;
}

export interface BillingProjectInfo {
  billing_project: string;
  accrued_cost: number;
  limit?: number | null;
}

// A small, hand-written mirror of just the endpoints callers actually use — not the batch
// service's full API surface. Add to this as more pages need more of it.
export function createHailApi(basePath: string) {
  return {
    getBatch: (batchId: string) =>
      hailApiFetch<Batch>(`${basePath}/api/v1alpha/batches/${batchId}`),

    getBatchJobs: (batchId: string, params: { q?: string; lastJobId?: number }) => {
      const search = new URLSearchParams();
      if (params.q) search.set('q', params.q);
      if (params.lastJobId != null) search.set('last_job_id', String(params.lastJobId));
      // Otherwise jobs nested in child job groups (e.g. CI's per-shard test groups) are omitted.
      search.set('recursive', 'true');
      const qs = search.toString();
      return hailApiFetch<BatchJobsPage>(`${basePath}/api/v2alpha/batches/${batchId}/jobs?${qs}`);
    },

    getBillingProject: (billingProject: string) =>
      hailApiFetch<BillingProjectInfo>(`${basePath}/api/v1alpha/billing_projects/${encodeURIComponent(billingProject)}`),

    cancelBatch: (batchId: string): Promise<void> =>
      hailApiFetch(`${basePath}/api/v1alpha/batches/${batchId}/cancel`, { method: 'PATCH' }),

    deleteBatch: (batchId: string): Promise<void> =>
      hailApiFetch(`${basePath}/api/v1alpha/batches/${batchId}`, { method: 'DELETE' }),
  };
}

export type HailApi = ReturnType<typeof createHailApi>;
