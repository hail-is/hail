import type { DeveloperBranchStatus, PrConfig, ReviewRequest } from '../../shared/ciApi';
import { labelEffect } from './labelModel';

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

// A build that ran (or is running) against an older target SHA. Only applies when there is a
// build: `null` (pending) and `unauthorized` mean there isn't one, even though the API's
// `out_of_date` can still be set for them.
const BUILD_STATES_WITH_A_BUILD = new Set(['success', 'failure', 'building']);

export function isBuildOutOfDate(pr: PrConfig): boolean {
  return pr.out_of_date === true && pr.build_state !== null && BUILD_STATES_WITH_A_BUILD.has(pr.build_state);
}

export function buildStateLabel(pr: PrConfig): string {
  if (pr.display_only) return 'not built';
  return pr.build_state ?? 'pending';
}

// One step of the merge journey (Review 1, Review 2, Labels, ...). `requestedAt` is the oldest
// open request the step is waiting on, for showing its age; `detail` is a tooltip.
export type StageStatus = 'pass' | 'fail' | 'pending' | 'na';

export interface StageCell {
  status: StageStatus;
  text: string;
  requestedAt?: string;
  detail?: string;
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

// Review 1 / Review 2 as two reviewer slots. An open codeowner request always takes slot 2. The
// other slots are filled in order: outstanding change requests (which fail slot 1, whatever the
// approvals), then approvals, then everyone still waited on, oldest request first. Hail asks
// internal reviewers via Assignees, so those count as waited on too. The last slot lists everyone
// left waiting, so nobody drops off the page.
export function reviewSlots(pr: PrConfig): [StageCell, StageCell] {
  const codeowner = pr.review_requests.filter((r) => r.as_code_owner);
  const requests = pr.review_requests
    .filter((r) => !r.as_code_owner)
    .sort((a, b) => (a.requested_at ?? '\uffff').localeCompare(b.requested_at ?? '\uffff'));

  const blockers = nonAuthorReviewers(pr, 'CHANGES_REQUESTED');
  const approvers = nonAuthorReviewers(pr, 'APPROVED');
  const reviewed = new Set([...blockers, ...approvers, pr.author]);
  const requested = requests.map((r) => r.reviewer).filter((n): n is string => n !== null);
  const waiting = [...new Set([...requested, ...pr.assignees])].filter((n) => !reviewed.has(n));

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

  let one: StageCell = slots[0] ?? { status: 'pending', text: 'no reviewer assigned' };
  let two: StageCell;
  if (codeowner.length > 0) two = codeownerCell(codeowner);
  else if (slots[1]) two = slots[1];
  else if (one.status === 'pass') two = { status: 'pending', text: 'needs a 2nd reviewer' };
  else two = { status: 'na', text: '' };
  // CI merges on its own review state, so once it's approved anyone still waited on (other than a
  // codeowner) is optional: shown muted rather than as a pending step.
  if (pr.review_state === 'approved') {
    if (one.status === 'pending') one = { ...one, status: 'na' };
    if (two.status === 'pending' && codeowner.length === 0) {
      two = two.text === 'needs a 2nd reviewer' ? { status: 'na', text: '2nd reviewer optional' } : { ...two, status: 'na' };
    }
  }
  return [one, two];
}

const LABEL_BLOCKER = 'label: ';

// The Labels step: fails on a label that stops CI merging, passes otherwise. Rendered as the
// labels themselves (as chips), with no status icon. An upper-stack PR has no merge blockers, so
// its labels are judged directly.
export function labelsCell(pr: PrConfig): StageCell {
  const blocking = pr.merge_blockers === null
    ? pr.labels.filter((l) => labelEffect(l)?.effect === 'blocks-merge')
    : pr.merge_blockers.filter((b) => b.startsWith(LABEL_BLOCKER)).map((b) => b.slice(LABEL_BLOCKER.length));
  const text = pr.labels.join(', ');
  if (blocking.length > 0) return { status: 'fail', text, detail: `not merged while labelled ${blocking.join(', ')}` };
  return { status: 'pass', text };
}

// The Up to date step: whether the build ran against the target branch's current commit.
export function upToDateCell(pr: PrConfig): StageCell {
  if (pr.build_state === null || !BUILD_STATES_WITH_A_BUILD.has(pr.build_state)) return { status: 'na', text: '' };
  if (pr.out_of_date) {
    return {
      status: 'pending',
      text: 'out of date',
      detail: 'Built or building against an older commit of the target branch. CI rebuilds it against the latest commit when it becomes the merge candidate.',
    };
  }
  return { status: 'pass', text: 'up to date' };
}

// The Merge step, last in the journey. It only says what the rest of the row can't: queue
// position, an in-flight merge, a branch-wide pause, retry backoff. A blocked PR is n/a (the
// earlier steps show why; the blockers are in the tooltip). Queue position, retries and pauses come
// from the branch's developer status; without it, a PR with no blockers is just "ready".
export function mergeCell(pr: PrConfig, branch: DeveloperBranchStatus | undefined, nowMs: number): StageCell {
  if (pr.display_only || pr.merge_blockers === null) return { status: 'na', text: 'stacked' };
  if (branch?.pending_merge?.pr === pr.number) {
    return { status: 'pending', text: 'merging', requestedAt: new Date(branch.pending_merge.requested_at_ms).toISOString() };
  }
  const queueIndex = branch?.merge_queue.findIndex((e) => e.pr === pr.number) ?? -1;
  const entry = queueIndex >= 0 ? branch?.merge_queue[queueIndex] : undefined;
  const parts = entry ? [`#${queueIndex + 1} in queue`] : [];

  if (pr.merge_blockers.length > 0) {
    return { status: 'na', text: parts.join(' · '), detail: pr.merge_blockers.join('\n') };
  }
  if (branch?.merge_paused_reason) {
    return { status: 'pending', text: [...parts, `merging paused (${branch.merge_paused_reason})`].join(' · ') };
  }
  if (entry?.backoff_until_ms != null && entry.backoff_until_ms > nowMs) {
    const retryIn = formatAge(new Date(nowMs).toISOString(), entry.backoff_until_ms);
    const failed = `merge failed${entry.failures > 1 ? ` ×${entry.failures}` : ''}, retrying in ${retryIn}`;
    return { status: 'pending', text: [...parts, failed].join(' · ') };
  }
  return { status: 'pass', text: parts.length > 0 ? parts.join(' · ') : 'ready' };
}

// Blockers the Build step covers (CI's build and the GitHub status checks).
function blocksBuild(blocker: string): boolean {
  return blocker === 'not built' || blocker === 'no status checks reported' || blocker.startsWith('checks ');
}

// How many journey steps, from the left, are done: the row fills green up to its first step
// that isn't, which is what's holding the PR up. A review slot that's n/a (optional) counts as done.
export function journeyFill(pr: PrConfig, branch: DeveloperBranchStatus | undefined, nowMs: number): number {
  const done = (cell: StageCell): boolean => cell.status === 'pass' || cell.status === 'na';
  const steps = [
    !pr.display_only && !(pr.merge_blockers ?? []).some(blocksBuild),
    ...reviewSlots(pr).map(done),
    labelsCell(pr).status === 'pass',
    upToDateCell(pr).status === 'pass',
    mergeCell(pr, branch, nowMs).status === 'pass',
  ];
  const first = steps.indexOf(false);
  return first < 0 ? steps.length : first;
}

// How each merge journey cell is drawn: `done` cells fill green from the left; the first cell
// that isn't done is the blocker, tinted by why (`fail` or `pending`); cells after it are `todo`.
export type JourneyMark = 'done' | 'fail' | 'pending' | 'todo';

const FAILED_STATES = new Set(['failure', 'error']);

function stateMark(state: string | null): JourneyMark {
  if (state === 'success') return 'done';
  return state !== null && FAILED_STATES.has(state) ? 'fail' : 'pending';
}

function stageMark(cell: StageCell): JourneyMark {
  if (cell.status === 'pass') return 'done';
  return cell.status === 'fail' ? 'fail' : 'pending';
}

// One mark per journey cell, left to right: CI, each GitHub status in `statusNames` (a PR
// without that status skips it), Review 1, Review 2, Labels, Up to date, Merge.
export function journeyMarks(
  pr: PrConfig,
  statusNames: string[],
  branch: DeveloperBranchStatus | undefined,
  nowMs: number,
): JourneyMark[] {
  const statuses = ghStatuses(pr);
  const optional = (cell: StageCell): StageCell => (cell.status === 'na' ? { ...cell, status: 'pass' } : cell);
  const cells: JourneyMark[] = [
    stateMark(pr.build_state),
    ...statusNames.map((name) => {
      const state = statuses.get(name) ?? '';
      return state === '' ? 'done' : stateMark(state);
    }),
    ...reviewSlots(pr).map((cell) => stageMark(optional(cell))),
    stageMark(labelsCell(pr)),
    stageMark(upToDateCell(pr)),
    stageMark(mergeCell(pr, branch, nowMs)),
  ];
  const first = cells.findIndex((m) => m !== 'done');
  return first < 0 ? cells : cells.map((m, i) => (i <= first ? m : 'todo'));
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
  const cells = [
    `${pr.title} #${pr.number}`,
    buildStateLabel(pr),
    ...statusNames.map((name) => statuses.get(name) ?? ''),
    ...reviewSlots(pr).map((cell) => cell.text),
    pr.labels.join(', '),
    upToDateCell(pr).text,
    mergeCell(pr, branch, nowMs).text,
    ...(pr.merge_blockers ?? []),
    pr.source_branch_name,
    pr.author,
  ];
  return cells.some((cell) => cell.toLowerCase().includes(needle));
}
