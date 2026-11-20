import type { DeployStatus, DeveloperBranchStatus, FailedDeployJob, MergeQueueEntry, RateLimit } from '../../shared/ciApi';

// Matches RATE_LIMIT_LOW_FRACTION in ci/ci/status.py, which feeds ci_unhealthy.
const RATE_LIMIT_LOW_FRACTION = 0.1;

export type Tone = 'good' | 'bad' | 'warn' | 'neutral';

export interface DeployDisplay {
  label: string;
  tone: Tone;
}

export function isDeployFailed(state: string | null): boolean {
  return state === 'failure' || state === 'checkout_failure';
}

export function deployDisplay(branch: Pick<DeveloperBranchStatus, 'deploy_state' | 'deploy_batch_id'>): DeployDisplay {
  switch (branch.deploy_state) {
    case 'success':
      return { label: 'deployed', tone: 'good' };
    case 'failure':
      return { label: 'deploy failed', tone: 'bad' };
    case 'checkout_failure':
      return { label: 'checkout failed', tone: 'bad' };
    case null:
      // CI clears the state when it starts a new deploy batch, so null with a batch means running.
      return branch.deploy_batch_id === null
        ? { label: 'no deploy yet', tone: 'neutral' }
        : { label: 'deploying', tone: 'warn' };
    default:
      return { label: branch.deploy_state, tone: 'neutral' };
  }
}

export function mergePausedLabel(reason: DeveloperBranchStatus['merge_paused_reason']): string | null {
  switch (reason) {
    case 'frozen':
      return 'Merging paused: merges and deploys are frozen';
    case 'deploying':
      return 'Merging paused: waiting for the current deploy to finish';
    default:
      return null;
  }
}

// One line saying why a merge-queue PR isn't merging, or that it's next.
export function mergeQueueStatus(entry: MergeQueueEntry, nowMs: number): { text: string; tone: Tone } {
  if (entry.backoff_until_ms !== null && entry.backoff_until_ms > nowMs) {
    const mins = Math.ceil((entry.backoff_until_ms - nowMs) / 60_000);
    const plural = entry.failures === 1 ? '' : 's';
    return { text: `merge failed ${entry.failures} time${plural}; retrying in ${mins}m`, tone: 'bad' };
  }
  if (entry.blockers.length > 0) {
    return { text: entry.blockers.join(' · '), tone: 'warn' };
  }
  return { text: 'ready to merge', tone: 'good' };
}

export function isRateLimitLow(rl: RateLimit): boolean {
  return rl.remaining < rl.limit * RATE_LIMIT_LOW_FRACTION;
}

export function isTraceback(info: DeployStatus['failure_information']): info is string[] {
  return Array.isArray(info) && info.length > 0 && typeof info[0] === 'string';
}

export function failedJobs(info: DeployStatus['failure_information']): FailedDeployJob[] {
  if (!Array.isArray(info) || isTraceback(info)) return [];
  return info as FailedDeployJob[];
}

// Identifies the set of failed deploys, so the page re-fetches /deploy_status (which does Batch
// I/O) only when that set changes rather than on every poll.
export function failedDeploysKey(branches: DeveloperBranchStatus[]): string {
  return branches
    .filter((b) => isDeployFailed(b.deploy_state))
    .map((b) => `${b.branch}:${b.deploy_batch_id ?? ''}:${b.deploy_state ?? ''}`)
    .join(',');
}
