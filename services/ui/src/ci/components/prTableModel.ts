import type { PrConfig, ReviewRequest } from '../../shared/ciApi';

export function ghStatuses(pr: PrConfig): Map<string, string> {
  return new Map(Object.entries(pr.gh_statuses));
}

// The API doesn't list status names, so the columns are the union of names across all PRs.
// Sorted so the column order doesn't shift between page loads.
export function statusColumnNames(prs: PrConfig[]): string[] {
  const names = new Set<string>();
  for (const pr of prs) {
    for (const name of Object.keys(pr.gh_statuses)) names.add(name);
  }
  return [...names].sort();
}

// A build that ran (or is running) against an older target SHA; its label gets a `*`. Only
// applies when there is a build: `null` (pending) and `unauthorized` mean there isn't one, even
// though the API's `out_of_date` can still be set for them.
const BUILD_STATES_WITH_A_BUILD = new Set(['success', 'failure', 'building']);

export function isBuildOutOfDate(pr: PrConfig): boolean {
  return pr.out_of_date && pr.build_state !== null && BUILD_STATES_WITH_A_BUILD.has(pr.build_state);
}

export function buildStateLabel(pr: PrConfig): string {
  if (pr.build_state === null) return 'pending';
  return isBuildOutOfDate(pr) ? `${pr.build_state} *` : pr.build_state;
}

// One PR-flow stage column (Review 1, Review 2, ...). `requestedAt` is the oldest open request
// the stage is waiting on, for showing its age.
export type StageStatus = 'pass' | 'fail' | 'pending' | 'na';

export interface StageCell {
  status: StageStatus;
  text: string;
  requestedAt?: string;
}

function oldestRequestedAt(requests: ReviewRequest[]): string | undefined {
  const times = requests.map((r) => r.requested_at).filter((t): t is string => t !== null).sort();
  return times[0];
}

function nonAuthorReviewers(pr: PrConfig, state: string): string[] {
  return pr.reviews.filter((r) => r.author !== pr.author && r.state === state).map((r) => r.author);
}

// First non-author approval. Any outstanding change request fails it, whatever the approvals.
// Hail asks internal reviewers via Assignees, so those count as who it's waiting on.
export function reviewOne(pr: PrConfig): StageCell {
  const blockers = nonAuthorReviewers(pr, 'CHANGES_REQUESTED');
  if (blockers.length > 0) return { status: 'fail', text: `changes requested by ${blockers.join(', ')}` };
  const approvers = nonAuthorReviewers(pr, 'APPROVED');
  if (approvers.length > 0) return { status: 'pass', text: `approved by ${approvers[0]}` };
  const requests = pr.review_requests.filter((r) => !r.as_code_owner);
  const waitingOn = [...new Set([...pr.assignees, ...requests.map((r) => r.reviewer)])].filter((n) => n !== pr.author);
  if (waitingOn.length === 0) return { status: 'pending', text: 'no reviewer assigned' };
  return { status: 'pending', text: `awaiting ${waitingOn.join(', ')}`, requestedAt: oldestRequestedAt(requests) };
}

// While a codeowner request is open, waiting on that codeowner; otherwise a second distinct
// non-author approval. (A codeowner's approval clears their request, so it then just counts.)
export function reviewTwo(pr: PrConfig): StageCell {
  const codeowner = pr.review_requests.filter((r) => r.as_code_owner);
  if (codeowner.length > 0) {
    return {
      status: 'pending',
      text: `awaiting ${codeowner.map((r) => r.reviewer).join(', ')} (codeowner)`,
      requestedAt: oldestRequestedAt(codeowner),
    };
  }
  const approvers = nonAuthorReviewers(pr, 'APPROVED');
  if (approvers.length >= 2) return { status: 'pass', text: `approved by ${approvers[1]}` };
  if (approvers.length === 1) return { status: 'pending', text: 'one more review' };
  return { status: 'na', text: '' };
}

// Coarse age for "waiting since": 45m, 5h, 3d.
export function formatAge(iso: string, nowMs: number): string {
  const minutes = Math.max(0, Math.floor((nowMs - Date.parse(iso)) / 60_000));
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours}h`;
  return `${Math.floor(hours / 24)}d`;
}

// Case-insensitive substring match against the text of every cell in the row.
export function matchesSearch(pr: PrConfig, statusNames: string[], term: string): boolean {
  const needle = term.trim().toLowerCase();
  if (needle === '') return true;
  const statuses = ghStatuses(pr);
  const cells = [
    `${pr.title} #${pr.number}`,
    buildStateLabel(pr),
    ...statusNames.map((name) => statuses.get(name) ?? ''),
    pr.labels.join(', '),
    pr.source_branch_name,
    reviewOne(pr).text,
    reviewTwo(pr).text,
    pr.author,
  ];
  return cells.some((cell) => cell.toLowerCase().includes(needle));
}
