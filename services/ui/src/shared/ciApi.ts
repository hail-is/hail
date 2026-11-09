import { hailApiFetch } from './hailApiFetch';

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

// A small, hand-written mirror of just the endpoints callers actually use — not the ci
// service's full API surface. Add to this as more pages need more of it.
export function createCiApi(basePath: string) {
  return {
    getWatchedBranches: () =>
      hailApiFetch<WatchedBranchesResponse>(`${basePath}/api/v1alpha/watched_branches`),

    getWatchedBranchPrs: (branch: string) =>
      hailApiFetch<PrConfig[]>(`${basePath}/api/v1alpha/watched_branches/${encodeURIComponent(branch)}/prs`),
  };
}

export type CiApi = ReturnType<typeof createCiApi>;
