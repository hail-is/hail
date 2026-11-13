import { hailApiFetch, hailApiFetchVoid } from './hailApiFetch';

// Wire types + client for the ci service's API. See batchApi.ts for why this lives in shared/.
// Field shapes follow ci/ci/templates/openapi.yaml.

// Branch identity and config; CI state is on CiDeveloperStatus.
export interface WatchedBranchSummary {
  index: number; // the {watched_branch_index} in UI page URLs
  branch: string;
  branch_fqn: string;
  repo: string;
  sha: string | null;
  deployable: boolean;
  mergeable: boolean;
}

export interface WatchedBranchesResponse {
  branches: WatchedBranchSummary[];
}

export interface ReviewRequest {
  // User login, or team slug when is_team. Null for a codeowner request whose reviewer CI's GitHub
  // token can't read (a team, without the org Members permission).
  reviewer: string | null;
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
  out_of_date: boolean | null; // null when display_only
  gh_statuses: Record<string, string>;
  batch_id: number | null;
  review_requests: ReviewRequest[];
  reviews: Review[];
  merge_blockers: string[] | null; // empty if nothing blocks the merge; null when display_only
  stack_entry: { position: number } | null; // 1 is nearest the stack's base branch
  stack: PrStack | null;
  // An upper-stack PR: in a native stack based on the branch, targeting another PR in it. Listed,
  // but CI never builds or merges it.
  display_only: boolean;
}

export interface PrStackEntry {
  position: number;
  number: number;
  state: string; // GitHub PullRequestState: OPEN, CLOSED, MERGED
  title: string;
}

// A native GitHub stack (gh stack); every member carries the same object.
export interface PrStack {
  number: number; // a group key, not a PR number
  base_ref_name: string;
  size: number;
  entries: PrStackEntry[]; // by position, merged and closed included
  name: string | null; // from the position-1 PR's Stack Meta, else the lowest open entry's title
  doc_url: string | null;
  meta: Record<string, string>;
}

// Status endpoints: in-memory state that resets when CI restarts. Times are epoch ms.

export interface QueuedBuild {
  pr: number;
  reason: string;
}

export interface MergeQueueEntry {
  pr: number;
  source_sha: string;
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

// One row of the CI event log (ci_events). Columns are sparse: each event type fills the ones
// that apply to it.
export interface CIEvent {
  id: number; // increases with time
  time: string; // ISO 8601
  event: string;
  target_branch: string | null; // fully-qualified, e.g. hail-is/hail:main
  pr_number: number | null;
  username: string | null; // null when CI acted on its own
  batch_id: number | null;
  source_sha: string | null;
  target_sha: string | null;
  previous_sha: string | null;
  namespace: string | null;
  service: string | null;
  merge_uuid: string | null;
  message: string | null;
  is_error: boolean; // a failure event, or a failed deploy
}

export interface CIEventsResponse {
  rows: CIEvent[]; // newest first
  cursor: number | null; // pass back as `cursor` for the next (older) page
  has_more: boolean;
}

export interface CIEventsQuery {
  targetBranch?: string;
  pr?: number;
  event?: string;
  sinceMs?: number;
  errorsOnly?: boolean;
  includePrEvents?: boolean; // false hides PR-scoped events that aren't errors
  cursor?: number;
  limit?: number;
}

// A small, hand-written mirror of just the endpoints callers actually use — not the ci
// service's full API surface. Add to this as more pages need more of it.
export function createCiApi(basePath: string) {
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

    getEvents: (q: CIEventsQuery) => {
      const params = new URLSearchParams();
      if (q.targetBranch !== undefined) params.set('target_branch', q.targetBranch);
      if (q.pr !== undefined) params.set('pr', String(q.pr));
      if (q.event !== undefined) params.set('event', q.event);
      if (q.sinceMs !== undefined) params.set('since_ms', String(q.sinceMs));
      if (q.errorsOnly !== undefined) params.set('errors_only', String(q.errorsOnly));
      if (q.includePrEvents !== undefined) params.set('include_pr_events', String(q.includePrEvents));
      if (q.cursor !== undefined) params.set('cursor', String(q.cursor));
      if (q.limit !== undefined) params.set('limit', String(q.limit));
      return hailApiFetch<CIEventsResponse>(`${basePath}/api/v1alpha/events?${params.toString()}`);
    },
  };
}

export type CiApi = ReturnType<typeof createCiApi>;
