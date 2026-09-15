import { StateIcon } from '../../batch/components/StateIcon';
import type { JobListEntry } from '../../batch/components/useBatchData';

export type { JobState, JobListEntry } from '../../batch/components/useBatchData';

export function JobList({ jobs, batchBaseUrl, batchId }: {
  jobs: JobListEntry[];
  batchBaseUrl: string;
  batchId: number;
}): JSX.Element | null {
  if (jobs.length === 0) return null;
  return (
    <ul className="divide-y divide-zinc-100 border border-zinc-200 rounded">
      {jobs.map((job) => (
        <li key={job.job_id} className="flex items-center gap-2 px-3 py-1.5 text-sm">
          <StateIcon state={job.state} />
          <a href={`${batchBaseUrl}/batches/${batchId}/jobs/${job.job_id}`} className="text-sky-600 hover:underline">
            {job.job_id}
          </a>
          <span className="text-zinc-600">{job.name ?? ''}</span>
          {job.exit_code !== null && <span className="text-zinc-400">(exit {job.exit_code})</span>}
        </li>
      ))}
    </ul>
  );
}
