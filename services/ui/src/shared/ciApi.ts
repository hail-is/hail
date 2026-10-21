import { hailApiFetch } from './hailApiFetch';

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
