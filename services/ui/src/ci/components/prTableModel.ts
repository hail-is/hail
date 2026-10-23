import type { PrConfig } from '../../shared/ciApi';

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
    pr.review_state ?? '',
    pr.author,
  ];
  return cells.some((cell) => cell.toLowerCase().includes(needle));
}
