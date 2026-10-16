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
export function withFieldTerm(q: string, field: string, value: string | null): string {
  const pattern = fieldTermPattern(field);
  const kept = q.split('\n').filter((line) => line.trim().length > 0 && !pattern.test(line));
  return (value == null ? kept : [...kept, `${field} = ${value}`]).join('\n');
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

export function jobCountsSummary(batch: Batch): string {
  const parts: string[] = [];
  const pending = batch.n_jobs - batch.n_completed;
  if (pending !== 0) parts.push(`${pending} pending`);
  if (batch.n_succeeded !== 0) parts.push(`${batch.n_succeeded} succeeded`);
  if (batch.n_cancelled !== 0) parts.push(`${batch.n_cancelled} cancelled`);
  if (batch.n_failed !== 0) parts.push(`${batch.n_failed} failed`);
  return parts.length > 0 ? parts.join(', ') : 'no jobs';
}

const MSEC_MCPU_PER_CPU_HOUR = 1000 * 3_600_000;

export function formatCpuHours(msecMcpu: number | null | undefined): string {
  if (msecMcpu == null) return '';
  const hours = msecMcpu / MSEC_MCPU_PER_CPU_HOUR;
  if (hours > 0 && hours < 0.01) return '< 0.01';
  return hours.toLocaleString(undefined, { maximumFractionDigits: hours < 100 ? 2 : 0 });
}
