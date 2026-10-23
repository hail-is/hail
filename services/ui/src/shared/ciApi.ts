import { hailApiFetch, hailApiFetchVoid } from './hailApiFetch';

// Wire types + client for the ci service's API. See batchApi.ts for why this lives in shared/.
// Field shapes follow ci/ci/templates/openapi.yaml.

export interface WatchedBranchSummary {
  branch: string;
  branch_fqn: string;
  repo: string;
}

export interface WatchedBranchesResponse {
  branches: WatchedBranchSummary[];
}

export interface ReviewRequest {
  reviewer: string; // user login, or team slug when is_team
  is_team: boolean;
  as_code_owner: boolean;
  requested_at: string | null; // ISO 8601
}

export interface Review {
  author: string;
  state: string; // GitHub PullRequestReviewState: APPROVED, CHANGES_REQUESTED, DISMISSED, ...
}

export interface PrConfig {
  number: number;
  title: string;
  author: string;
  assignees: string[];
  reviewers: string[];
  source_branch_name: string;
  build_state: string | null;
  review_state: string | null;
  labels: string[];
  out_of_date: boolean;
  gh_statuses: Record<string, string>;
  batch_id: number | null;
  review_requests: ReviewRequest[];
  reviews: Review[];
}

// Status endpoints: in-memory state that resets when CI restarts. Times are epoch ms.

export interface QueuedBuild {
  pr: number;
  reason: string;
}

export interface MergeQueueEntry {
  pr: number;
  blockers: string[];
  failures: number;
  backoff_until_ms: number | null;
}

export interface DeveloperBranchStatus {
  branch: string;
  sha: string | null;
  deploy_state: string | null; // success / failure / checkout_failure; null while deploying
  deploy_batch_id: number | null;
  merge_candidate: number | null;
  n_prs: number;
  last_update: { started_ms: number; finished_ms: number; duration_s: number } | null;
  merge_paused_reason: 'frozen' | 'deploying' | null;
  pending_merge: { pr: number; requested_at_ms: number } | null;
  build_slots: {
    used: number; // can exceed max: the merge candidate bypasses the cap
    max: number;
    running: number[];
    on_deck: number | null;
    queued: QueuedBuild[];
  };
  merge_queue: MergeQueueEntry[];
}

export interface CiDeveloperStatus {
  started_at_ms: number;
  ci_unhealthy: boolean;
  frozen: boolean;
  branches: DeveloperBranchStatus[];
}

export interface RateLimit {
  remaining: number;
  limit: number;
  reset_at: string; // ISO 8601
}

export interface GraphQLRateLimit extends RateLimit {
  last_update_cost: number;
}

export interface OperatorBranchStatus {
  branch: string;
  pip_version: string | null;
  updating: boolean;
  update_started_ms: number | null;
  dirty: string[];
  last_update_error: string | null;
  rate_limit: { rest: RateLimit | null; graphql: GraphQLRateLimit | null };
  deploy_exception: string | null;
}

export interface GitHubApiError {
  branch: string;
  at_ms: number;
  operation: string;
  status: number | null;
  message: string;
}

export interface CiOperatorStatus {
  started_at_ms: number;
  health_problems: string[];
  github: {
    last_webhook_ms: Record<string, number>; // keyed by GitHub event type
    last_api_error: GitHubApiError | null;
  };
  loops: Record<string, { last_success_ms: number | null }>; // keyed by loop name
  branches: OperatorBranchStatus[];
}

export interface FailedDeployJob {
  batch_id: number;
  job_id: number;
  name?: string;
  state: string;
}

export interface DeployStatus {
  branch: string;
  sha: string | null;
  deploy_batch_id: number | null;
  deploy_state: string | null;
  repo: string;
  // Failed jobs for a batch failure, or traceback lines when the deploy failed before a batch
  // existed; null unless deploy_state is failure / checkout_failure.
  failure_information: FailedDeployJob[] | string[] | null;
}

// A small, hand-written mirror of just the endpoints callers actually use — not the ci
// service's full API surface. Add to this as more pages need more of it.
export function createCiApi(basePath: string) {
  const jsonPost = (body: unknown): RequestInit => ({
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  return {
    getWatchedBranches: () =>
      hailApiFetch<WatchedBranchesResponse>(`${basePath}/api/v1alpha/watched_branches`),

    getWatchedBranchPrs: (branch: string) =>
      hailApiFetch<PrConfig[]>(`${basePath}/api/v1alpha/watched_branches/${encodeURIComponent(branch)}/prs`),

    getDeveloperStatus: () =>
      hailApiFetch<CiDeveloperStatus>(`${basePath}/api/v1alpha/ci_developer_status`),

    getOperatorStatus: () =>
      hailApiFetch<CiOperatorStatus>(`${basePath}/api/v1alpha/ci_operator_status`),

    // Fetches failed job details from Batch, so only call it when a deploy has failed.
    getDeployStatus: () =>
      hailApiFetch<DeployStatus[]>(`${basePath}/api/v1alpha/deploy_status`),

    freeze: () => hailApiFetchVoid(`${basePath}/api/v1alpha/freeze`, { method: 'POST' }),

    unfreeze: () => hailApiFetchVoid(`${basePath}/api/v1alpha/unfreeze`, { method: 'POST' }),

    authorizeSha: (sha: string) =>
      hailApiFetchVoid(`${basePath}/api/v1alpha/authorize_sha`, jsonPost({ sha })),
  };
}

export type CiApi = ReturnType<typeof createCiApi>;
