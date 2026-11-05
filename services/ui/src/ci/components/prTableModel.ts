import type { DeveloperBranchStatus, PrConfig, ReviewRequest } from '../../shared/ciApi';

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
  return pr.out_of_date === true && pr.build_state !== null && BUILD_STATES_WITH_A_BUILD.has(pr.build_state);
}

export function buildStateLabel(pr: PrConfig): string {
  if (pr.display_only) return 'not built';
  if (pr.build_state === null) return 'pending';
  return isBuildOutOfDate(pr) ? `${pr.build_state} *` : pr.build_state;
}

// One PR-flow stage column (Review 1, Review 2, ...). `requestedAt` is the oldest open request
// the stage is waiting on, for showing its age. `labels` are shown as chips after the text.
export type StageStatus = 'pass' | 'fail' | 'pending' | 'na';

export interface StageCell {
  status: StageStatus;
  text: string;
  requestedAt?: string;
  labels?: string[];
}

function oldestRequestedAt(requests: ReviewRequest[]): string | undefined {
  const times = requests.map((r) => r.requested_at).filter((t): t is string => t !== null).sort();
  return times[0];
}

function nonAuthorReviewers(pr: PrConfig, state: string): string[] {
  return pr.reviews.filter((r) => r.author !== pr.author && r.state === state).map((r) => r.author);
}

function awaiting(names: string[], requests: ReviewRequest[]): StageCell {
  const asked = requests.filter((r) => r.reviewer !== null && names.includes(r.reviewer));
  return { status: 'pending', text: `awaiting ${names.join(', ')}`, requestedAt: oldestRequestedAt(asked) };
}

function codeownerCell(requests: ReviewRequest[]): StageCell {
  // A null reviewer is one CI's GitHub token can't read (a team, without the org Members permission)
  const known = requests.map((r) => r.reviewer).filter((n): n is string => n !== null);
  const text = known.length > 0 ? `awaiting ${known.join(', ')} (codeowner)` : 'awaiting codeowner';
  return { status: 'pending', text, requestedAt: oldestRequestedAt(requests) };
}

// Who a PR's review is waiting on: non-codeowner requests (oldest first) and assignees who haven't
// approved or requested changes yet, excluding the author. Hail asks internal reviewers via
// Assignees, so those count as waited on too. `codeowner` holds any open codeowner requests.
export interface ReviewWait {
  waiting: string[];
  requests: ReviewRequest[];
  codeowner: ReviewRequest[];
}

export function reviewWait(pr: PrConfig): ReviewWait {
  const codeowner = pr.review_requests.filter((r) => r.as_code_owner);
  const requests = pr.review_requests
    .filter((r) => !r.as_code_owner)
    .sort((a, b) => (a.requested_at ?? '\uffff').localeCompare(b.requested_at ?? '\uffff'));
  const reviewed = new Set([
    ...nonAuthorReviewers(pr, 'CHANGES_REQUESTED'),
    ...nonAuthorReviewers(pr, 'APPROVED'),
    pr.author,
  ]);
  const requested = requests.map((r) => r.reviewer).filter((n): n is string => n !== null);
  const waiting = [...new Set([...requested, ...pr.assignees])].filter((n) => !reviewed.has(n));
  return { waiting, requests, codeowner };
}

// Review 1 / Review 2 as two reviewer slots. An open codeowner request always takes slot 2. The
// other slots are filled in order: outstanding change requests (which fail slot 1, whatever the
// approvals), then approvals, then everyone still waited on (`reviewWait`). The last slot lists
// everyone left waiting, so nobody drops off the page.
export function reviewSlots(pr: PrConfig): [StageCell, StageCell] {
  const { waiting, requests, codeowner } = reviewWait(pr);
  const blockers = nonAuthorReviewers(pr, 'CHANGES_REQUESTED');
  const approvers = nonAuthorReviewers(pr, 'APPROVED');

  const filled: StageCell[] = [];
  if (blockers.length > 0) filled.push({ status: 'fail', text: `changes requested by ${blockers.join(', ')}` });
  for (const a of approvers) filled.push({ status: 'pass', text: `approved by ${a}` });

  const nSlots = codeowner.length > 0 ? 1 : 2;
  const slots = filled.slice(0, nSlots);
  for (let i = 0; slots.length < nSlots && i < waiting.length; i++) {
    const isLast = slots.length === nSlots - 1;
    slots.push(awaiting(isLast ? waiting.slice(i) : [waiting[i]], requests));
    if (isLast) break;
  }

  const one = slots[0] ?? { status: 'pending', text: 'no reviewer assigned' };
  let two: StageCell;
  if (codeowner.length > 0) two = codeownerCell(codeowner);
  else if (slots[1]) two = slots[1];
  else if (one.status === 'pass') two = { status: 'pending', text: 'needs a 2nd reviewer' };
  else two = { status: 'na', text: '' };
  return [one, two];
}

// The Merge column. Blockers show as pending rather than failed: whatever is failing (checks,
// reviews) is already red in its own column, so this column says why CI isn't merging. Queue
// position, retries and pauses come from the branch's developer status; without it, only the
// PR's own blockers are shown. Label blockers (`label: WIP`) come back as chips, not text.
const LABEL_BLOCKER = 'label: ';

export function mergeCell(pr: PrConfig, branch: DeveloperBranchStatus | undefined, nowMs: number): StageCell {
  if (pr.display_only || pr.merge_blockers === null) return { status: 'na', text: 'stacked' };
  if (branch?.pending_merge?.pr === pr.number) {
    return { status: 'pending', text: 'merging', requestedAt: new Date(branch.pending_merge.requested_at_ms).toISOString() };
  }
  const queueIndex = branch?.merge_queue.findIndex((e) => e.pr === pr.number) ?? -1;
  const entry = queueIndex >= 0 ? branch?.merge_queue[queueIndex] : undefined;
  const parts = entry ? [`#${queueIndex + 1} in queue`] : [];

  if (pr.merge_blockers.length > 0) {
    const labels = pr.merge_blockers.filter((b) => b.startsWith(LABEL_BLOCKER)).map((b) => b.slice(LABEL_BLOCKER.length));
    const others = pr.merge_blockers.filter((b) => !b.startsWith(LABEL_BLOCKER));
    const text = [...parts, ...others].join(' · ');
    return labels.length > 0 ? { status: 'pending', text, labels } : { status: 'pending', text };
  }
  if (branch?.merge_paused_reason) {
    return { status: 'pending', text: [...parts, `merging paused (${branch.merge_paused_reason})`].join(' · ') };
  }
  if (entry?.backoff_until_ms != null && entry.backoff_until_ms > nowMs) {
    const retryIn = formatAge(new Date(nowMs).toISOString(), entry.backoff_until_ms);
    const failed = `merge failed${entry.failures > 1 ? ` ×${entry.failures}` : ''}, retrying in ${retryIn}`;
    return { status: 'pending', text: [...parts, failed].join(' · ') };
  }
  return { status: 'pass', text: ['ready', ...parts].join(' · ') };
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
export function matchesSearch(
  pr: PrConfig,
  statusNames: string[],
  term: string,
  branch?: DeveloperBranchStatus,
  nowMs: number = Date.now(),
): boolean {
  const needle = term.trim().toLowerCase();
  if (needle === '') return true;
  const statuses = ghStatuses(pr);
  const merge = mergeCell(pr, branch, nowMs);
  const cells = [
    `${pr.title} #${pr.number}`,
    buildStateLabel(pr),
    ...statusNames.map((name) => statuses.get(name) ?? ''),
    pr.labels.join(', '),
    pr.source_branch_name,
    ...reviewSlots(pr).map((cell) => cell.text),
    merge.text,
    ...(merge.labels ?? []),
    pr.author,
  ];
  return cells.some((cell) => cell.toLowerCase().includes(needle));
}
