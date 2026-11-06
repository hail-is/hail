import type { CIEvent } from '../../shared/ciApi';

// A run of consecutive events that are the same apart from time (and ids, batches, shas):
// e.g. update_failed every 5 minutes during a GitHub outage.
export interface EventGroup {
  latest: CIEvent;
  earliest: CIEvent;
  count: number;
}

function sameGroup(a: CIEvent, b: CIEvent): boolean {
  return a.event === b.event
    && a.target_branch === b.target_branch
    && a.pr_number === b.pr_number
    && a.message === b.message;
}

// Rows arrive newest first; groups keep that order.
export function collapseRepeats(rows: CIEvent[]): EventGroup[] {
  const groups: EventGroup[] = [];
  for (const row of rows) {
    const last = groups.at(-1);
    if (last && sameGroup(last.earliest, row)) {
      last.earliest = row;
      last.count += 1;
    } else {
      groups.push({ latest: row, earliest: row, count: 1 });
    }
  }
  return groups;
}

// The newest merge_failed event for each PR's current merge attempt (its source SHA against the
// branch's current SHA), so a failure from before a new commit or a main update isn't shown.
// Rows are newest first.
export function currentMergeFailures(
  rows: CIEvent[],
  branchFqn: string,
  targetSha: string | null,
  queue: { pr: number; source_sha: string }[],
): Map<number, CIEvent> {
  const failures = new Map<number, CIEvent>();
  for (const { pr, source_sha } of queue) {
    const latest = rows.find((e) => e.event === 'merge_failed'
      && e.target_branch === branchFqn
      && e.pr_number === pr
      && e.source_sha === source_sha
      && e.target_sha === targetSha);
    if (latest) failures.set(pr, latest);
  }
  return failures;
}

export function isFailureEvent(event: string): boolean {
  return event.endsWith('_failed') || event === 'merge_expired';
}

function shortSha(sha: string): string {
  return sha.slice(0, 7);
}

// The structured columns worth showing, other than those with their own column.
export function eventDetails(e: CIEvent): string[] {
  const details: string[] = [];
  if (e.source_sha) details.push(`source ${shortSha(e.source_sha)}`);
  if (e.previous_sha && e.target_sha) {
    details.push(`target ${shortSha(e.previous_sha)} → ${shortSha(e.target_sha)}`);
  } else if (e.target_sha) {
    details.push(`target ${shortSha(e.target_sha)}`);
  }
  if (e.namespace) details.push(e.service ? `${e.namespace}/${e.service}` : e.namespace);
  if (e.merge_uuid) details.push(`merge ${e.merge_uuid}`);
  return details;
}
