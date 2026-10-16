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

function fieldTermPattern(field: string): RegExp {
  // Field names are fixed identifiers from this codebase (never user input), so no escaping needed.
  return new RegExp(`^\\s*${field}\\s*(==|!=|=)\\s*(\\S*)\\s*$`);
}

// Sets a v2 query's term for `field` to `field = value` (or removes it, if `value` is null),
// replacing any term for that field already there rather than ANDing a second one (which for a
// single-valued field could only ever match nothing), and keeping the rest of the query.
//
// With op '!=', excludes `value` instead: any `=` term for the field goes (it would contradict or
// make redundant the exclusion), but other `!=` terms for it stay, so exclusions accumulate.
export function withFieldTerm(q: string, field: string, value: string | null, op: '=' | '!=' = '='): string {
  const pattern = fieldTermPattern(field);
  const kept = q.split('\n').filter((line) => {
    if (line.trim().length === 0) return false;
    const m = pattern.exec(line);
    if (m == null) return true;
    return op === '!=' && m[1] === '!=' && m[2] !== value;
  });
  return (value == null ? kept : [...kept, `${field} ${op} ${value}`]).join('\n');
}

export function hasFieldTerm(q: string, field: string, value: string): boolean {
  const pattern = fieldTermPattern(field);
  return q.split('\n').some((line) => {
    const m = pattern.exec(line);
    return m != null && m[1] !== '!=' && m[2] === value;
  });
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

export interface JobCount {
  label: string;
  count: number;
}

// The nonzero job-state counts for a batch, in display order.
export function jobCountBreakdown(batch: Batch): JobCount[] {
  return [
    { label: 'pending', count: batch.n_jobs - batch.n_completed },
    { label: 'succeeded', count: batch.n_succeeded },
    { label: 'failed', count: batch.n_failed },
    { label: 'cancelled', count: batch.n_cancelled },
  ].filter(({ count }) => count !== 0);
}
