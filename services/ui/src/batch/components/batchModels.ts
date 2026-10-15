import type { Batch, BatchJob, CostBreakdownEntry } from '../../shared/batchApi';

export function jobDisplayState(job: BatchJob): string {
  return job.always_run && job.state !== 'Success' && job.state !== 'Failed' && job.state !== 'Error'
    ? `${job.state} (always run)`
    : job.state;
}

export function formatDurationMs(ms: number | null | undefined): string {
  if (ms == null) return '';
  const totalSec = Math.round(ms / 1000);
  const days = Math.floor(totalSec / 86_400);
  const hours = Math.floor((totalSec % 86_400) / 3_600);
  const minutes = Math.floor((totalSec % 3_600) / 60);
  const seconds = totalSec % 60;
  if (days > 0) return `${days}d${hours}h`;
  if (hours > 0) return `${hours}h${minutes}m`;
  if (minutes > 0) return `${minutes}m${seconds}s`;
  return `${seconds}s`;
}

const BILLING_PROJECT_TERM = /^\s*billing_project\s*(==|!=|=)/;

// Narrows a v2 batch query to one billing project, replacing any billing_project term already in
// it (rather than ANDing a second one, which could only ever match nothing) and keeping the rest.
export function withBillingProjectFilter(q: string, billingProject: string): string {
  const kept = q.split('\n').filter((line) => line.trim().length > 0 && !BILLING_PROJECT_TERM.test(line));
  return [...kept, `billing_project = ${billingProject}`].join('\n');
}

export function isFilteredToBillingProject(q: string, billingProject: string): boolean {
  return q.split('\n').some((line) => /^\s*billing_project\s*==?\s*(\S+)\s*$/.exec(line)?.[1] === billingProject);
}

// Resource names look like 'compute/n1-preemptible/1' or 'service-fee/1'; the leading segment is
// the human-meaningful category.
export function costByCategory(breakdown: readonly CostBreakdownEntry[]): CostBreakdownEntry[] {
  const totals = new Map<string, number>();
  for (const { resource, cost } of breakdown) {
    const category = resource.split('/')[0];
    totals.set(category, (totals.get(category) ?? 0) + cost);
  }
  return [...totals].map(([resource, cost]) => ({ resource, cost })).sort((a, b) => b.cost - a.cost);
}

export function jobCountsSummary(batch: Batch): string {
  const parts: string[] = [];
  const pending = batch.n_jobs - batch.n_completed;
  if (pending !== 0) parts.push(`${pending} pending`);
  if (batch.n_succeeded !== 0) parts.push(`${batch.n_succeeded} succeeded`);
  if (batch.n_cancelled !== 0) parts.push(`${batch.n_cancelled} cancelled`);
  if (batch.n_failed !== 0) parts.push(`${batch.n_failed} failed`);
  return parts.length > 0 ? parts.join(', ') : 'no jobs';
}
