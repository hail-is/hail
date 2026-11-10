import type { BatchJob } from '../../shared/batchApi';

export function jobDisplayState(job: BatchJob): string {
  return job.always_run && job.state !== 'Success' && job.state !== 'Failed' && job.state !== 'Error'
    ? `${job.state} (always run)`
    : job.state;
}
