import { SegmentedBar } from '../../shared/SegmentedBar';
import type { Segment } from '../../shared/SegmentedBar';
import { CollapsibleItem } from '../../batch/components/CollapsibleItem';
import { ROOT_JOB_GROUP_ID } from '../../batch/components/useBatchData';
import type { JobGroupSummary, UseBatchDataResult } from '../../batch/components/useBatchData';
import { JobList } from './JobList';

export { ROOT_JOB_GROUP_ID };
export type { JobGroupSummary };

function jobGroupSegments(g: JobGroupSummary): Segment[] {
  const inProgress = g.n_jobs - g.n_completed;
  return [
    [g.n_succeeded, 'bg-green-500', `Succeeded: ${g.n_succeeded}`],
    [inProgress, 'bg-sky-400', `In progress: ${inProgress}`],
    [g.n_failed, 'bg-red-500', `Failed: ${g.n_failed}`],
    [g.n_cancelled, 'bg-zinc-400', `Cancelled: ${g.n_cancelled}`],
  ];
}

function groupLabel(g: JobGroupSummary): string {
  return g.attributes?.name ?? (g.job_group_id === ROOT_JOB_GROUP_ID ? 'root' : `job group ${g.job_group_id}`);
}

function JobGroupRow({ batchBaseUrl, batchId, summary, batchData }: {
  batchBaseUrl: string;
  batchId: number;
  summary: JobGroupSummary;
  batchData: UseBatchDataResult;
}): JSX.Element {
  const childGroups = batchData.getJobGroups(summary.job_group_id);
  const error = batchData.getJobGroupsError(summary.job_group_id);
  // Already fully known client-side (getJobs filters the already-fetched recursive job list) —
  // doesn't need to wait on whether this group *has* sub-groups, which is the only part that
  // genuinely requires a network round trip. Rendering it unconditionally, rather than gated
  // behind childGroups resolving, avoids an artificial "Loading…" for data that was never
  // actually loading.
  const ownJobs = batchData.getJobs(summary.job_group_id);

  return (
    <CollapsibleItem
      title={groupLabel(summary)}
      summary={
        <div className="flex items-center gap-2">
          <SegmentedBar segments={jobGroupSegments(summary)} total={summary.n_jobs} className="h-3 w-32" />
          <span>{summary.n_completed}/{summary.n_jobs} jobs</span>
        </div>
      }
      onExpand={() => { batchData.fetchJobGroups(summary.job_group_id); }}
    >
      <div className="pl-4">
        {error ? (
          <p className="text-xs text-red-600">{error}</p>
        ) : childGroups === undefined ? (
          <p className="text-xs text-zinc-400">Checking for sub-groups&hellip;</p>
        ) : childGroups.length > 0 ? (
          <ul className="border-l border-zinc-200">
            {childGroups.map((child) => (
              <JobGroupRow key={child.job_group_id} batchBaseUrl={batchBaseUrl} batchId={batchId} summary={child} batchData={batchData} />
            ))}
          </ul>
        ) : (
          <p className="text-xs text-zinc-400 mb-1">No sub-groups</p>
        )}

        {ownJobs.length > 0 && (
          <div className="mt-2">
            <p className="text-xs text-zinc-400 mb-1">Jobs directly in this group</p>
            <JobList jobs={ownJobs} batchBaseUrl={batchBaseUrl} batchId={batchId} />
          </div>
        )}
      </div>
    </CollapsibleItem>
  );
}

// Generic job-group hierarchy viewer: nests recursively via job-groups/{id}/job-groups, lazily
// fetching a row's children only on its first expand (via CollapsibleItem's onExpand). All
// fetching, caching, and the "already-fetched jobs, filtered client-side" trick live in
// useBatchData (services/ui/src/batch/components/useBatchData.ts), keyed by job_group_id rather
// than tied to any one row's mount lifecycle — so the cache survives a row unmounting/
// remounting, not just staying mounted-but-collapsed. Works against any batch, CI or otherwise —
// CI's own build batches don't currently nest job groups, so this renders as a single root row
// against them today, but the component makes no CI-specific assumptions.
export function JobGroupTree({ batchBaseUrl, batchId, batchData }: {
  batchBaseUrl: string;
  batchId: number;
  batchData: UseBatchDataResult;
}): JSX.Element | null {
  if (!batchData.rootJobGroup) return null;
  return (
    <ul className="border border-zinc-200 rounded divide-y divide-zinc-100">
      <JobGroupRow batchBaseUrl={batchBaseUrl} batchId={batchId} summary={batchData.rootJobGroup} batchData={batchData} />
    </ul>
  );
}
