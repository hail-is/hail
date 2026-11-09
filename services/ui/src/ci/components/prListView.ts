import type { DeveloperBranchStatus, PrConfig } from '../../shared/ciApi';
import { labelEffect, visibleLabels } from './labelModel';
import { ghStatuses, journeyFill, mergeCell, reviewSlots, reviewWait, upToDateCell, type StageCell } from './prTableModel';

// Author / reviewer filters. `viewer` is the signed-in user's GitHub username; the filters that
// need it are hidden when it's unknown.
export type PrFilter = 'all' | 'mine' | 'waiting-on-me' | 'my-requests' | 'open-requests';

export const PR_FILTERS: { value: PrFilter; label: string; needsViewer: boolean }[] = [
  { value: 'all', label: 'All', needsViewer: false },
  { value: 'mine', label: 'Mine', needsViewer: true },
  { value: 'waiting-on-me', label: 'Waiting on me', needsViewer: true },
  { value: 'my-requests', label: 'My open requests', needsViewer: true },
  { value: 'open-requests', label: 'Any open request', needsViewer: false },
];

// A review is outstanding while someone is waited on or a codeowner request is open.
function hasOpenRequest(pr: PrConfig): boolean {
  const { waiting, codeowner } = reviewWait(pr);
  return waiting.length > 0 || codeowner.length > 0;
}

export function matchesFilter(pr: PrConfig, filter: PrFilter, viewer: string | null): boolean {
  switch (filter) {
    case 'all':
      return true;
    case 'mine':
      return pr.author === viewer;
    case 'waiting-on-me':
      return viewer !== null && reviewWait(pr).waiting.includes(viewer);
    case 'my-requests':
      return pr.author === viewer && hasOpenRequest(pr);
    case 'open-requests':
      return hasOpenRequest(pr);
  }
}

// Sortable columns. A GitHub status column is `status:<name>`.
export type SortColumn = 'pr' | 'build' | 'labels' | 'branch' | 'review1' | 'review2' | 'uptodate' | 'merge' | 'author' | `status:${string}`;

export interface SortState {
  column: SortColumn;
  dir: 'asc' | 'desc';
}

// Header clicks cycle asc → desc → unsorted (API order); a new column starts at asc.
export function nextSort(current: SortState | null, column: SortColumn): SortState | null {
  if (current?.column !== column) return { column, dir: 'asc' };
  return current.dir === 'asc' ? { column, dir: 'desc' } : null;
}

// Build states and GitHub statuses, most in need of attention first. Unknown states sort with
// pending; a missing status sorts last.
function statusRank(state: string | null | undefined): number {
  switch (state) {
    case 'failure':
    case 'error':
    case 'cancelled':
      return 0;
    case 'unauthorized':
      return 1;
    case 'success':
      return 3;
    case undefined:
    case '':
      return 4;
    default:
      return 2;
  }
}

// "Needs a nudge": pending longest-waiting first, then pending with no known request time, then
// fail, pass, n/a.
const STAGE_RANK = new Map([['fail', 2], ['pass', 3], ['na', 4]]);

function stageKey(cell: StageCell): SortKey {
  if (cell.status === 'pending') {
    return cell.requestedAt === undefined ? [1, ''] : [0, cell.requestedAt];
  }
  return [STAGE_RANK.get(cell.status) ?? 4, ''];
}

const LABEL_EFFECT_RANK = new Map([['blocks-merge', 0], ['priority', 1], ['test-control', 2]]);

type SortKey = (number | string)[];

export interface SortContext {
  branch: DeveloperBranchStatus | undefined;
  nowMs: number;
}

function sortKey(pr: PrConfig, column: SortColumn, ctx: SortContext): SortKey {
  switch (column) {
    case 'pr':
      return [pr.number];
    case 'build':
      return [statusRank(pr.build_state ?? 'pending')];
    case 'labels': {
      // Most significant visible chip first; PRs with no chips last.
      const first = visibleLabels(pr.labels).at(0);
      if (first === undefined) return [LABEL_EFFECT_RANK.size, ''];
      return [LABEL_EFFECT_RANK.get(labelEffect(first)?.effect ?? '') ?? LABEL_EFFECT_RANK.size, first];
    }
    case 'branch':
      return [pr.source_branch_name.toLowerCase()];
    case 'review1':
      return stageKey(reviewSlots(pr)[0]);
    case 'review2':
      return stageKey(reviewSlots(pr)[1]);
    case 'uptodate':
      return stageKey(upToDateCell(pr));
    case 'merge': {
      // Furthest along the merge journey first; ties in merge-queue order.
      const queueIndex = ctx.branch?.merge_queue.findIndex((e) => e.pr === pr.number) ?? -1;
      return [-journeyFill(pr, ctx.branch, ctx.nowMs), queueIndex >= 0 ? queueIndex : Number.MAX_SAFE_INTEGER];
    }
    case 'author':
      return [pr.author.toLowerCase()];
    default:
      return [statusRank(ghStatuses(pr).get(column.slice('status:'.length)))];
  }
}

function compareKeys(a: SortKey, b: SortKey): number {
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const x = a.at(i);
    const y = b.at(i);
    if (x === y) continue;
    if (typeof x === 'number' && typeof y === 'number') return x - y;
    return String(x).localeCompare(String(y));
  }
  return a.length - b.length;
}

// Sorted copy of `prs`; ties (and an unset sort) keep API order, in both directions.
export function sortPrs(prs: PrConfig[], sort: SortState | null, ctx: SortContext): PrConfig[] {
  if (sort === null) return prs;
  const sign = sort.dir === 'asc' ? 1 : -1;
  const keyed = prs.map((pr, index) => ({ pr, index, key: sortKey(pr, sort.column, ctx) }));
  keyed.sort((a, b) => sign * compareKeys(a.key, b.key) || a.index - b.index);
  return keyed.map((k) => k.pr);
}
