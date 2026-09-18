import { BatchJob } from '../../shared/batchApi';

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
