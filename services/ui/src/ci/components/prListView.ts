import type { DeveloperBranchStatus, PrConfig, PrStack, PrStackEntry } from '../../shared/ciApi';
import { labelEffect, visibleLabels } from './labelModel';
import {
  ghStatuses,
  journeyFill,
  matchesSearchTerm,
  mergeCell,
  reviewSlots,
  reviewWait,
  type StageCell,
  upToDateCell,
} from './prTableModel';

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

// Sorted copy of `items`, each keyed by the PR `prOf` picks; ties (and an unset sort) keep the
// given order, in both directions.
function sortBy<T>(items: T[], prOf: (_item: T) => PrConfig, sort: SortState | null, ctx: SortContext): T[] {
  if (sort === null) return items;
  const sign = sort.dir === 'asc' ? 1 : -1;
  const keyed = items.map((item, index) => ({ item, index, key: sortKey(prOf(item), sort.column, ctx) }));
  keyed.sort((a, b) => sign * compareKeys(a.key, b.key) || a.index - b.index);
  return keyed.map((k) => k.item);
}

export function sortPrs(prs: PrConfig[], sort: SortState | null, ctx: SortContext): PrConfig[] {
  return sortBy(prs, (pr) => pr, sort, ctx);
}

// Rows of the PR list. A stack is one item: a summary row plus its entries by position. Entries
// the API lists are full PR rows; merged, closed and unlisted open entries come from
// `stack.entries` and render muted.
export type StackRow =
  | { kind: 'pr'; pr: PrConfig; position: number }
  | { kind: 'entry'; entry: PrStackEntry };

export type ListItem =
  | { kind: 'pr'; pr: PrConfig }
  | {
    kind: 'stack';
    stack: PrStack;
    rows: StackRow[]; // the rows left after filtering, by position
    // Rows matching the active filter / search; null when neither is active. Merged / closed
    // entries only count when the search matches their title.
    matchCount: number | null;
  };

export interface ListQuery {
  filter: PrFilter;
  viewer: string | null;
  search: string;
  // Search over a listed PR's cells (`matchesSearch` with the page's columns).
  searchPr: (_pr: PrConfig) => boolean;
}

// Groups `prs` into stacks, applies the filter and search, and sorts. A stack sorts as a unit
// keyed by its bottom PR (the one targeting the branch) and stays in the list while any of its
// rows match. Merged / closed rows are always shown as context once the stack is listed; they have
// no author or review data, so only search (by title) can match them. A search matching the
// stack's name matches every member.
export function buildListItems(prs: PrConfig[], query: ListQuery, sort: SortState | null, ctx: SortContext): ListItem[] {
  const filtering = query.filter !== 'all';
  const searching = query.search.trim() !== '';
  const units: { item: ListItem; keyPr: PrConfig }[] = [];
  const stacks = new Map<number, PrConfig[]>();
  for (const pr of prs) {
    if (pr.stack === null) {
      if (matchesFilter(pr, query.filter, query.viewer) && query.searchPr(pr)) units.push({ item: { kind: 'pr', pr }, keyPr: pr });
      continue;
    }
    const members = stacks.get(pr.stack.number);
    if (members === undefined) {
      stacks.set(pr.stack.number, [pr]);
      // Placeholder keeps the stack where its first member was in API order.
      units.push({ item: { kind: 'stack', stack: pr.stack, rows: [], matchCount: null }, keyPr: pr });
    } else {
      members.push(pr);
    }
  }

  const result: { item: ListItem; keyPr: PrConfig }[] = [];
  for (const unit of units) {
    if (unit.item.kind === 'pr') {
      result.push(unit);
      continue;
    }
    const stack = unit.item.stack;
    const members = stacks.get(stack.number) ?? [];
    const nameMatches = stack.name !== null && matchesSearchTerm(stack.name, query.search);
    const rows = stackRows(stack, members);
    const matches = (row: StackRow): boolean => {
      if (row.kind === 'pr') {
        return matchesFilter(row.pr, query.filter, query.viewer) && (nameMatches || query.searchPr(row.pr));
      }
      return !filtering && (nameMatches || matchesSearchTerm(`${row.entry.title} #${row.entry.number}`, query.search));
    };
    const matchCount = rows.filter(matches).length;
    if (matchCount === 0) continue;
    const shown = rows.filter((row) => row.kind === 'entry' || matches(row));
    const keyPr = members.find((m) => !m.display_only)
      ?? [...members].sort((a, b) => (a.stack_entry?.position ?? 0) - (b.stack_entry?.position ?? 0))[0];
    result.push({
      item: { kind: 'stack', stack, rows: shown, matchCount: filtering || searching ? matchCount : null },
      keyPr,
    });
  }
  return sortBy(result, (u) => u.keyPr, sort, ctx).map((u) => u.item);
}

// Every entry of `stack` by position, using the listed PR where there is one. Listed members
// missing from `entries` (it's capped) are added from their own `stack_entry`.
function stackRows(stack: PrStack, members: PrConfig[]): StackRow[] {
  const byNumber = new Map(members.map((m) => [m.number, m]));
  const rows: StackRow[] = stack.entries.map((entry) => {
    const pr = byNumber.get(entry.number);
    return pr === undefined ? { kind: 'entry', entry } : { kind: 'pr', pr, position: entry.position };
  });
  const inEntries = new Set(stack.entries.map((e) => e.number));
  for (const m of members) {
    if (!inEntries.has(m.number)) rows.push({ kind: 'pr', pr: m, position: m.stack_entry?.position ?? 0 });
  }
  const position = (row: StackRow): number => (row.kind === 'pr' ? row.position : row.entry.position);
  return rows.sort((a, b) => position(a) - position(b));
}
