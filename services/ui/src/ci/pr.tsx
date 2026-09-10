import { useEffect, useState, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import { hailApiFetch as apiFetch, hailApiFetchVoid as apiFetchVoid } from '../shared/hailApiFetch';
import { hasPermission } from '../shared/authUtils';
import { SegmentedBar, Segment } from '../shared/SegmentedBar';
import { StateIcon } from '../batch/components/StateIcon';
import { BatchStateIcon } from '../shared/BatchStateIcon';
import { useTip, FloatingTip } from '../shared/useTip';
import { AutoRefreshBar } from '../shared/AutoRefreshBar';

const REFRESH_INTERVAL_MS = 30_000;

type JobState = 'Pending' | 'Ready' | 'Creating' | 'Running' | 'Failed' | 'Cancelled' | 'Error' | 'Success';

interface JobListEntry {
  job_id: number;
  name: string | null;
  state: JobState;
  exit_code: number | null;
}

interface BatchSummary {
  id: number;
  state: string | null;
  cost: number | null;
  artifacts_uri: string;
}

interface PrsAheadEntry {
  number: number;
  title: string;
  is_merge_candidate: boolean;
}

interface WatchedBranchPr {
  number: number;
  title: string;
  labels: string[];
  // Active-PR fields — present only when the PR is currently open and tracked.
  review_approved?: boolean;
  checks_all_pass?: boolean;
  is_up_to_date?: boolean;
  blocking_labels?: string[];
  is_mergeable?: boolean;
  prs_ahead_in_queue?: PrsAheadEntry[];
  is_merge_candidate?: boolean;
  blocking_deploy_batch_id?: number | null;
  batch?: BatchSummary | null;
  review_state?: string | null;
  build_state?: string | null;
  // New/extended fields — optional so the page degrades gracefully against an
  // unmodified backend that doesn't return them yet.
  pr_authorized?: boolean;
  exception?: string | null;
  source_sha?: string;
  pending_build_reason?: string | null;
  // Historical-PR fields (GitHub fallback branch).
  merged?: boolean | null;
  merge_commit_sha?: string | null;
}

interface BatchStatus {
  id: number;
  state: string;
  complete: boolean;
  cost: number | null;
  n_jobs: number;
  n_completed: number;
  n_succeeded: number;
  n_failed: number;
  n_cancelled: number;
  time_created: string | null;
  time_completed: string | null;
  attributes?: Record<string, string>;
}

interface BatchHistoryEntry {
  id: number;
  state?: string;
  time_created?: string;
  attributes?: Record<string, string>;
}

function isActivePr(pr: WatchedBranchPr): boolean {
  return pr.review_approved !== undefined;
}

const JOB_STATE_ORDER: JobState[] = ['Failed', 'Error', 'Cancelled', 'Running', 'Pending', 'Ready', 'Creating', 'Success'];

function bucketJobs(jobs: JobListEntry[]): Record<JobState, JobListEntry[]> {
  const buckets = Object.fromEntries(JOB_STATE_ORDER.map((s) => [s, [] as JobListEntry[]])) as Record<JobState, JobListEntry[]>;
  for (const job of jobs) {
    buckets[job.state]?.push(job);
  }
  return buckets;
}

function gcpLoggingQueries(namespace: string, startTime: string, endTime: string | null): Record<string, string> {
  const severityQuery = (severities: string[]) => severities.map((s) => `severity=${s}`).join(' OR ');
  const timestampQuery = (start: string, end: string | null) =>
    `;startTime=${start}${end ? `;endTime=${end}` : ''}`;
  const serviceQuery = (services: string[]) =>
    services
      .map(
        (s) => `
(
resource.type="k8s_container"
resource.labels.namespace_name="${namespace}"
resource.labels.container_name="${s}"
)
`
      )
      .join(' OR ');
  const workerQuery = `
(
resource.type="gce_instance"
logName:"worker"
labels.namespace="${namespace}"
)
`;
  const url = (query: string, severities: string[]) =>
    `https://console.cloud.google.com/logs/query;query=${encodeURIComponent(
      query + severityQuery(severities)
    )};${encodeURIComponent(timestampQuery(startTime, endTime))}`;

  return {
    'batch-k8s-error-warning': url(serviceQuery(['batch', 'batch-driver']), ['ERROR', 'WARNING']),
    'batch-workers-error-warning': url(workerQuery, ['ERROR', 'WARNING']),
    'ci-k8s-error-warning': url(serviceQuery(['ci']), ['ERROR', 'WARNING']),
    'auth-k8s-error-warning': url(serviceQuery(['auth', 'auth-driver']), ['ERROR', 'WARNING']),
  };
}

async function fetchAllBatches(batchBaseUrl: string, q: string): Promise<BatchHistoryEntry[]> {
  const all: BatchHistoryEntry[] = [];
  let lastBatchId: number | undefined;
  for (;;) {
    const params = new URLSearchParams({ q });
    if (lastBatchId !== undefined) params.set('last_batch_id', String(lastBatchId));
    const page = await apiFetch<{ batches: BatchHistoryEntry[]; last_batch_id?: number }>(
      `${batchBaseUrl}/api/v1alpha/batches?${params.toString()}`
    );
    all.push(...page.batches);
    if (page.last_batch_id === undefined) break;
    lastBatchId = page.last_batch_id;
  }
  return all;
}

async function fetchAllJobs(batchBaseUrl: string, batchId: number): Promise<JobListEntry[]> {
  const all: JobListEntry[] = [];
  let lastJobId: number | undefined;
  for (;;) {
    const url = new URL(`${batchBaseUrl}/api/v1alpha/batches/${batchId}/jobs`, window.location.origin);
    if (lastJobId !== undefined) url.searchParams.set('last_job_id', String(lastJobId));
    const page = await apiFetch<{ jobs: JobListEntry[]; last_job_id?: number }>(url.toString());
    all.push(...page.jobs);
    if (page.last_job_id === undefined) break;
    lastJobId = page.last_job_id;
  }
  return all;
}

const FAILED_JOBS_DISPLAY_LIMIT = 10;

// Fetches up to FAILED_JOBS_DISPLAY_LIMIT + 1 failed/errored ("bad" state) job names for a
// batch, so callers can show the first 10 and know whether there are more without having to
// paginate through the whole batch.
async function fetchFirstBadJobs(batchBaseUrl: string, batchId: number): Promise<JobListEntry[]> {
  const found: JobListEntry[] = [];
  let lastJobId: number | undefined;
  while (found.length <= FAILED_JOBS_DISPLAY_LIMIT) {
    const url = new URL(`${batchBaseUrl}/api/v1alpha/batches/${batchId}/jobs`, window.location.origin);
    url.searchParams.set('q', 'bad');
    if (lastJobId !== undefined) url.searchParams.set('last_job_id', String(lastJobId));
    const page = await apiFetch<{ jobs: JobListEntry[]; last_job_id?: number }>(url.toString());
    found.push(...page.jobs);
    if (page.last_job_id === undefined) break;
    lastJobId = page.last_job_id;
  }
  return found.slice(0, FAILED_JOBS_DISPLAY_LIMIT + 1);
}

function storageUriToUrl(uri: string): string {
  if (uri.startsWith('gs://')) {
    return `https://console.cloud.google.com/storage/browser/${uri.slice('gs://'.length)}`;
  }
  return uri;
}

function Icon({ name, className }: { name: string; className: string }): JSX.Element {
  return <span className={`material-symbols-outlined align-middle text-lg ${className}`}>{name}</span>;
}

function MergeEligibility({ pr, basePath, batchBaseUrl, wbIndex }: {
  pr: WatchedBranchPr;
  basePath: string;
  batchBaseUrl: string;
  wbIndex: string;
}): JSX.Element {
  return (
    <div className="mt-4">
      <h2 className="text-lg font-semibold text-zinc-700 mb-2">Merge Eligibility</h2>
      <ul className="space-y-1 text-sm">
        <li>
          {pr.review_approved ? (
            <><Icon name="check_circle" className="text-green-600" /> PR approved</>
          ) : pr.review_state === 'changes_requested' ? (
            <><Icon name="block" className="text-red-600" /> Changes requested</>
          ) : (
            <><Icon name="pending" className="text-zinc-400" /> Awaiting review</>
          )}
        </li>
        <li>
          {pr.checks_all_pass ? (
            <><Icon name="check_circle" className="text-green-600" /> All checks passing</>
          ) : pr.build_state == null || pr.build_state === 'building' ? (
            <>
              <span className="material-symbols-outlined align-middle text-lg animate-spin text-zinc-400" style={{ animationDuration: '1s' }}>
                progress_activity
              </span>{' '}
              Build in progress
            </>
          ) : (
            <><Icon name="cancel" className="text-red-600" /> Checks not passing ({pr.build_state})</>
          )}
        </li>
        <li>
          {pr.is_up_to_date ? (
            <><Icon name="check_circle" className="text-green-600" /> Target SHA matches HEAD</>
          ) : pr.is_merge_candidate ? (
            <><Icon name="warning" className="text-orange-500" /> Target SHA does not match HEAD - rebuild starting shortly</>
          ) : (
            <><Icon name="warning" className="text-orange-500" /> Target SHA does not match HEAD - will rebuild once merge candidate</>
          )}
        </li>
        {pr.blocking_labels && pr.blocking_labels.length > 0 && (
          <li>
            <Icon name="cancel" className="text-red-600" /> Blocked by label: {pr.blocking_labels.join(', ')}
          </li>
        )}
      </ul>

      <h2 className="text-lg font-semibold text-zinc-700 mt-4 mb-2">Merge Queue</h2>
      {(() => {
        const buildFailing = pr.build_state === 'failure';
        if (buildFailing || !pr.review_approved) {
          return (
            <p className="text-sm text-zinc-600">
              Not in the merge queue -{' '}
              {buildFailing && !pr.review_approved
                ? 'needs approval and build is failing.'
                : !pr.review_approved
                ? 'needs approval.'
                : 'build is failing.'}
            </p>
          );
        }
        const aheadCount = pr.prs_ahead_in_queue?.length ?? 0;
        return (
          <>
            {aheadCount > 0 ? (
              <>
                <p className="text-sm text-zinc-600">{aheadCount} PR(s) ahead in queue:</p>
                <ul className="list-disc pl-6 text-sm">
                  {pr.prs_ahead_in_queue!.map((ahead) => (
                    <li key={ahead.number}>
                      <a href={`${basePath}/watched_branches/${wbIndex}/pr/${ahead.number}`} className="text-sky-600 hover:underline">
                        #{ahead.number}
                      </a>{' '}
                      - {ahead.title}
                      {ahead.is_merge_candidate ? <em> (merge candidate)</em> : null}
                    </li>
                  ))}
                </ul>
              </>
            ) : pr.is_mergeable ? (
              <p className="text-sm text-zinc-600">No approved PRs ahead - this PR is next to merge.</p>
            ) : pr.is_merge_candidate ? (
              <p className="text-sm text-zinc-600">This PR is the current merge candidate.</p>
            ) : (
              <p className="text-sm text-zinc-600">Not yet in the merge queue - awaiting build results.</p>
            )}
            {pr.blocking_deploy_batch_id && (
              <p className="text-sm text-orange-600 mt-1">
                <Icon name="hourglass_top" className="text-orange-500" /> Deploy batch{' '}
                <a href={`${batchBaseUrl}/batches/${pr.blocking_deploy_batch_id}`} className="text-sky-600 hover:underline">
                  {pr.blocking_deploy_batch_id}
                </a>{' '}
                is running - merge blocked until it completes.
              </p>
            )}
          </>
        );
      })()}
    </div>
  );
}

type BadJobsState =
  | { status: 'pending' }
  | { status: 'loading' }
  | { status: 'loaded'; jobs: JobListEntry[]; truncated: boolean }
  | { status: 'error'; message: string };

function BadJobsCell({
  state,
  repeatedNames,
  alwaysFailingNames,
  batchBaseUrl,
  batchId,
}: {
  state: BadJobsState | undefined;
  repeatedNames: Set<string>;
  alwaysFailingNames: Set<string>;
  batchBaseUrl: string;
  batchId: number;
}): JSX.Element | null {
  if (state === undefined) return null;
  if (state.status === 'pending') {
    return <span className="text-zinc-400">pending</span>;
  }
  if (state.status === 'loading') {
    return (
      <span className="text-zinc-400 inline-flex items-center gap-1">
        <span className="material-symbols-outlined text-sm animate-spin" style={{ animationDuration: '1s' }}>
          progress_activity
        </span>
        loading
      </span>
    );
  }
  if (state.status === 'error') {
    return <span className="text-red-600">{state.message}</span>;
  }
  if (state.jobs.length === 0) return null;
  return (
    <span className="font-mono text-xs">
      {state.jobs.map((j, i) => {
        const name = j.name ?? '';
        const always = alwaysFailingNames.has(name);
        const repeated = always || repeatedNames.has(name);
        const className = always ? 'font-bold text-sm' : repeated ? 'font-bold' : undefined;
        return (
          <span key={j.job_id} className={className}>
            {i > 0 && ', '}
            <a href={`${batchBaseUrl}/batches/${batchId}/jobs/${j.job_id}`} className="text-sky-600 hover:underline">
              {j.name}
            </a>
          </span>
        );
      })}
      {state.truncated && <span className="text-zinc-400"> &hellip; and more</span>}
    </span>
  );
}

function BatchHistoryTable({ batches, batchBaseUrl }: { batches: BatchHistoryEntry[]; batchBaseUrl: string }): JSX.Element {
  const [badJobs, setBadJobs] = useState<Map<number, BadJobsState> | null>(null);

  const loadBadJobs = useCallback(async () => {
    setBadJobs(new Map(batches.map((b) => [b.id, { status: 'pending' } as BadJobsState])));
    for (const b of batches) {
      setBadJobs((prev) => (prev ? new Map(prev).set(b.id, { status: 'loading' }) : prev));
      try {
        const jobs = await fetchFirstBadJobs(batchBaseUrl, b.id);
        const truncated = jobs.length > FAILED_JOBS_DISPLAY_LIMIT;
        setBadJobs((prev) =>
          prev ? new Map(prev).set(b.id, { status: 'loaded', jobs: jobs.slice(0, FAILED_JOBS_DISPLAY_LIMIT), truncated }) : prev
        );
      } catch (e) {
        setBadJobs((prev) => (prev ? new Map(prev).set(b.id, { status: 'error', message: 'failed to load' }) : prev));
      }
    }
  }, [batches, batchBaseUrl]);

  if (batches.length === 0) return <p className="text-sm text-zinc-500">No builds.</p>;

  // Job names that showed up as a failure/error in more than one loaded row are bolded; ones
  // that showed up in *every* loaded row are also sized up, so a solid vertical line of matches
  // down the column pops out.
  const loadedNameSets = badJobs
    ? [...badJobs.values()]
        .filter((s): s is Extract<BadJobsState, { status: 'loaded' }> => s.status === 'loaded')
        .map((s) => new Set(s.jobs.map((j) => j.name ?? '').filter((name) => name !== '')))
    : [];
  const nameCounts = new Map<string, number>();
  for (const set of loadedNameSets) {
    for (const name of set) {
      nameCounts.set(name, (nameCounts.get(name) ?? 0) + 1);
    }
  }
  const repeatedNames = new Set([...nameCounts].filter(([, count]) => count >= 2).map(([name]) => name));
  const alwaysFailingNames =
    loadedNameSets.length >= 2
      ? new Set([...nameCounts].filter(([, count]) => count === loadedNameSets.length).map(([name]) => name))
      : new Set<string>();

  return (
    <table className="w-auto text-sm border border-zinc-200 rounded overflow-hidden">
      <thead>
        <tr className="bg-zinc-100 text-left text-xs uppercase text-zinc-500">
          <th className="px-3 py-0.5">id</th>
          <th className="px-3 py-0.5">reason</th>
          <th className="px-3 py-0.5">started</th>
          <th className="px-3 py-0.5">state</th>
          <th className="px-3 py-0.5">
            {badJobs === null ? (
              <button type="button" onClick={loadBadJobs} className="normal-case text-sky-600 hover:underline">
                Load failed job names
              </button>
            ) : (
              'failed jobs'
            )}
          </th>
        </tr>
      </thead>
      <tbody className="divide-y divide-zinc-100">
        {batches.map((b) => (
          <tr key={b.id}>
            <td className="px-3 py-0.5">
              <a href={`${batchBaseUrl}/batches/${b.id}`} className="text-sky-600 hover:underline">{b.id}</a>
            </td>
            <td className="px-3 py-0.5">{b.attributes?.reason ?? ''}</td>
            <td className="px-3 py-0.5">{b.time_created ?? ''}</td>
            <td className="px-3 py-0.5 whitespace-nowrap">{b.state ? <BatchStateIcon state={b.state} /> : null} {b.state}</td>
            <td className="px-3 py-0.5">
              <BadJobsCell
                state={badJobs?.get(b.id)}
                repeatedNames={repeatedNames}
                alwaysFailingNames={alwaysFailingNames}
                batchBaseUrl={batchBaseUrl}
                batchId={b.id}
              />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function JobList({ jobs, batchBaseUrl, batchId }: { jobs: JobListEntry[]; batchBaseUrl: string; batchId: number }): JSX.Element | null {
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

// These fields are only ever `undefined` when the deployed `ci` backend hasn't picked up
// the API extension yet (it always sets them explicitly once it has — `null` is a real
// value, not an absence). This should never fire once frontend and backend are deployed
// from the same commit; if it does in a steady-state deploy, something is wrong.
function missingActivePrFields(pr: WatchedBranchPr): string[] {
  const missing: string[] = [];
  if (pr.pr_authorized === undefined) missing.push('pr_authorized');
  if (pr.exception === undefined) missing.push('exception');
  if (pr.source_sha === undefined) missing.push('source_sha');
  if (pr.pending_build_reason === undefined) missing.push('pending_build_reason');
  return missing;
}

function MissingApiFieldsWarning({ fields }: { fields: string[] }): JSX.Element | null {
  if (fields.length === 0) return null;
  return (
    <div className="mb-3 px-3 py-2 text-xs text-amber-800 bg-amber-50 border border-dashed border-amber-300 rounded">
      <Icon name="warning" className="text-amber-600" /> API field{fields.length > 1 ? 's' : ''} missing:{' '}
      <span className="font-mono">{fields.join(', ')}</span>
    </div>
  );
}

function BuildPanel({ pr, basePath, batchBaseUrl, wbBranchName, prNumber, batchStatus, jobs, jobsError }: {
  pr: WatchedBranchPr;
  basePath: string;
  batchBaseUrl: string;
  wbBranchName: string;
  prNumber: string;
  batchStatus: BatchStatus | null;
  jobs: JobListEntry[] | null;
  jobsError: string | null;
}): JSX.Element {
  const [retrying, setRetrying] = useState(false);
  const [authorizing, setAuthorizing] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const canManage = hasPermission('manage_ci');
  const missingFields = missingActivePrFields(pr);
  const warning = <MissingApiFieldsWarning fields={missingFields} />;
  const [tip, onTipEnter, onTipLeave] = useTip();

  if (pr.batch) {
    const jobBuckets = jobs ? bucketJobs(jobs) : null;
    const segments: Segment[] = jobBuckets
      ? [
          ['Success', 'bg-green-500'] as const,
          ['Running', 'bg-sky-500'] as const,
          ['Creating', 'bg-sky-300'] as const,
          ['Ready', 'bg-zinc-300'] as const,
          ['Pending', 'bg-zinc-200'] as const,
          ['Failed', 'bg-red-500'] as const,
          ['Error', 'bg-orange-500'] as const,
          ['Cancelled', 'bg-zinc-400'] as const,
        ].map(([state, color]) => [jobBuckets[state as JobState].length, color, `${state}: ${jobBuckets[state as JobState].length}`] as Segment)
      : [];
    const totalJobs = jobs?.length ?? 0;
    const runningJobs = jobBuckets ? jobBuckets.Running : [];
    const failedErroredJobs = jobBuckets ? [...jobBuckets.Failed, ...jobBuckets.Error] : [];

    return (
      <div className="mt-4">
        <h2 className="text-lg font-semibold text-zinc-700 mb-2">Current Build</h2>
        {warning}
        <div className="text-sm space-y-1">
          <div>
            batch:{' '}
            <a href={`${batchBaseUrl}/batches/${pr.batch.id}`} className="text-sky-600 hover:underline">{pr.batch.id}</a>
          </div>
          <div>
            artifacts:{' '}
            <a
              href={storageUriToUrl(pr.batch.artifacts_uri)}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sky-600 hover:underline"
            >
              {pr.batch.artifacts_uri}
            </a>
          </div>
          <div>cost: {pr.batch.cost ?? ''}</div>
          <div>labels: {pr.labels.join(', ')}</div>
        </div>

        {jobBuckets && totalJobs > 0 && (
          <div className="mt-3">
            <h3 className="text-sm font-semibold text-zinc-600 mb-1">Batch Status:</h3>
            <SegmentedBar
              segments={segments}
              total={totalJobs}
              className="h-4 w-full max-w-md"
              onSegmentEnter={onTipEnter}
              onSegmentLeave={onTipLeave}
            />
            <FloatingTip tip={tip} />
            <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-xs text-zinc-600">
              {segments
                .filter(([count]) => count > 0)
                .map(([count, color, label]) => (
                  <div key={label} className="flex items-center gap-1.5">
                    <span className={`inline-block w-2.5 h-2.5 rounded-sm ${color}`} />
                    <span>{label}</span>
                  </div>
                ))}
            </div>
          </div>
        )}

        {canManage && batchStatus?.complete && (
          <div className="mt-3 flex items-center gap-3">
            <RetryButton basePath={basePath} wbBranchName={wbBranchName} prNumber={prNumber} tactical={false} setError={setActionError} disabled={retrying} setRetrying={setRetrying} />
            {pr.is_up_to_date && (pr.build_state === 'failure' || pr.build_state === 'error') && (
              <RetryButton basePath={basePath} wbBranchName={wbBranchName} prNumber={prNumber} tactical={true} setError={setActionError} disabled={retrying} setRetrying={setRetrying} />
            )}
          </div>
        )}
        {actionError && <p className="text-sm text-red-600 mt-1">{actionError}</p>}

        {jobsError ? (
          <p className="text-sm text-red-600 mt-4">{jobsError}</p>
        ) : !jobBuckets ? (
          <p className="text-sm text-zinc-500 mt-4">Loading jobs&hellip;</p>
        ) : (
          <>
            {runningJobs.length > 0 && (
              <div className="mt-4">
                <h3 className="text-sm font-semibold text-zinc-600 mb-1">Running Jobs</h3>
                <JobList jobs={runningJobs} batchBaseUrl={batchBaseUrl} batchId={pr.batch.id} />
              </div>
            )}
            {failedErroredJobs.length > 0 && (
              <div className="mt-4">
                <h3 className="text-sm font-semibold text-zinc-600 mb-1">Failed / Errored Jobs</h3>
                <JobList jobs={failedErroredJobs} batchBaseUrl={batchBaseUrl} batchId={pr.batch.id} />
              </div>
            )}
          </>
        )}

        {batchStatus?.attributes?.namespace && batchStatus.time_created && (
          <div className="mt-4">
            <h3 className="text-sm font-semibold text-zinc-600 mb-1">Logging Queries</h3>
            <div className="text-sm space-y-0.5">
              {Object.entries(
                gcpLoggingQueries(batchStatus.attributes.namespace, batchStatus.time_created, batchStatus.time_completed)
              ).map(([name, link]) => (
                <div key={name}>
                  <a href={link} target="_blank" rel="noopener noreferrer" className="text-sky-600 hover:underline">{name}</a>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  if (pr.exception) {
    return (
      <div className="mt-4">
        <h2 className="text-lg font-semibold text-zinc-700 mb-2">Current Build</h2>
        {warning}
        <p className="text-red-600">Build error:</p>
        <pre className="bg-red-50 border border-red-200 rounded p-3 text-sm whitespace-pre-wrap overflow-x-auto">{pr.exception}</pre>
      </div>
    );
  }

  if (pr.pr_authorized === true) {
    return (
      <div className="mt-4">
        <h2 className="text-lg font-semibold text-zinc-700 mb-2">Current Build</h2>
        {warning}
        <p className="text-sm text-zinc-600">
          {pr.pending_build_reason
            ? `Waiting for a build slot (${pr.pending_build_reason})…`
            : 'No current build — waiting for a build slot (or a retry is being processed).'}
        </p>
      </div>
    );
  }

  if (pr.pr_authorized === false) {
    return (
      <div className="mt-4">
        <h2 className="text-lg font-semibold text-zinc-700 mb-2">Current Build</h2>
        {warning}
        <p className="text-sm text-zinc-600">
          Build blocked: this PR&apos;s latest commit is not authorized. It must be reviewed and approved manually by a developer before authorizing.
        </p>
        {canManage && (
          <AuthorizeButton
            basePath={basePath}
            sha={pr.source_sha ?? ''}
            authorizing={authorizing}
            setAuthorizing={setAuthorizing}
            setError={setActionError}
          />
        )}
        {actionError && <p className="text-sm text-red-600 mt-1">{actionError}</p>}
      </div>
    );
  }

  return (
    <div className="mt-4">
      <h2 className="text-lg font-semibold text-zinc-700 mb-2">Current Build</h2>
      {warning}
      <p className="text-sm text-zinc-600">No current build.</p>
    </div>
  );
}

function RetryButton({ basePath, wbBranchName, prNumber, tactical, setError, disabled, setRetrying }: {
  basePath: string;
  wbBranchName: string;
  prNumber: string;
  tactical: boolean;
  setError: (e: string | null) => void;
  disabled: boolean;
  setRetrying: (b: boolean) => void;
}): JSX.Element {
  return (
    <button
      type="button"
      disabled={disabled}
      className="px-3 py-1 text-sm rounded bg-sky-600 text-white hover:bg-sky-700 disabled:opacity-50"
      onClick={() => {
        setError(null);
        setRetrying(true);
        apiFetchVoid(`${basePath}/api/v1alpha/watched_branches/${encodeURIComponent(wbBranchName)}/prs/${prNumber}/retry`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ tactical }),
        })
          .then(() => { window.location.reload(); })
          .catch((e: unknown) => { setError(e instanceof Error ? e.message : String(e)); })
          .finally(() => { setRetrying(false); });
      }}
    >
      {tactical ? 'Tactical Retry' : 'Retry'}
    </button>
  );
}

function AuthorizeButton({ basePath, sha, authorizing, setAuthorizing, setError }: {
  basePath: string;
  sha: string;
  authorizing: boolean;
  setAuthorizing: (b: boolean) => void;
  setError: (e: string | null) => void;
}): JSX.Element {
  return (
    <button
      type="button"
      disabled={authorizing || !sha}
      className="mt-2 px-3 py-1 text-sm rounded bg-sky-600 text-white hover:bg-sky-700 disabled:opacity-50"
      onClick={() => {
        setError(null);
        setAuthorizing(true);
        apiFetchVoid(`${basePath}/api/v1alpha/authorize_sha`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sha }),
        })
          .then(() => { window.location.reload(); })
          .catch((e: unknown) => { setError(e instanceof Error ? e.message : String(e)); })
          .finally(() => { setAuthorizing(false); });
      }}
    >
      Authorize now
    </button>
  );
}

interface WatchedBranchListEntry {
  branch: string;
  branch_fqn: string;
  repo: string;
}

function PrPage({ basePath, batchBaseUrl, wbIndex, prNumber }: {
  basePath: string;
  batchBaseUrl: string;
  wbIndex: string;
  prNumber: string;
}): JSX.Element {
  // The page shell (rendered either by the real backend or, in dev, entirely locally by
  // dev_proxy with no upstream call) only knows the watched-branch index from the URL.
  // The branch's name/fqn require server-side config only the real backend has, so we
  // derive them here from the same list endpoint the watched-branches index page already
  // uses — this also keeps the shell itself upstream-free in dev.
  const [wbBranchName, setWbBranchName] = useState<string | null>(null);
  const [wbBranch, setWbBranch] = useState<string | null>(null);
  const [repo, setRepo] = useState<string | null>(null);
  const [wbError, setWbError] = useState<string | null>(null);

  const [pr, setPr] = useState<WatchedBranchPr | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [history, setHistory] = useState<BatchHistoryEntry[]>([]);
  const [deployHistory, setDeployHistory] = useState<BatchHistoryEntry[]>([]);

  const [batchStatus, setBatchStatus] = useState<BatchStatus | null>(null);
  const [jobs, setJobs] = useState<JobListEntry[] | null>(null);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [countdownKey, setCountdownKey] = useState(0);
  const [autoRefresh, setAutoRefreshState] = useState<boolean>(() => {
    try {
      return localStorage.getItem('ci.prPage.autoRefresh') !== 'false';
    } catch {
      return true;
    }
  });
  const setAutoRefresh = useCallback((v: boolean) => {
    setAutoRefreshState(v);
    try {
      localStorage.setItem('ci.prPage.autoRefresh', String(v));
    } catch { /* ignore */ }
  }, []);

  const batchId = pr?.batch?.id;

  const refreshBuild = useCallback(async (isRefresh: boolean) => {
    if (batchId === undefined) return;
    if (isRefresh) setRefreshing(true);
    try {
      const [status, allJobs] = await Promise.all([
        apiFetch<BatchStatus>(`${batchBaseUrl}/api/v1alpha/batches/${batchId}`),
        fetchAllJobs(batchBaseUrl, batchId),
      ]);
      setBatchStatus(status);
      setJobs(allJobs);
      setJobsError(null);
    } catch (e: unknown) {
      setJobsError(e instanceof Error ? e.message : String(e));
    } finally {
      setCountdownKey((k) => k + 1);
      if (isRefresh) setRefreshing(false);
    }
  }, [batchBaseUrl, batchId]);

  useEffect(() => { void refreshBuild(false); }, [refreshBuild]);

  // Keep polling only while the current batch hasn't finished.
  useEffect(() => {
    if (batchStatus === null || batchStatus.complete || !autoRefresh) return;
    const id = setInterval(() => { void refreshBuild(true); }, REFRESH_INTERVAL_MS);
    return () => { clearInterval(id); };
  }, [batchStatus, refreshBuild, autoRefresh]);

  useEffect(() => {
    apiFetch<{ branches: WatchedBranchListEntry[] }>(`${basePath}/api/v1alpha/watched_branches`)
      .then((r) => {
        const entry = r.branches[Number(wbIndex)];
        if (!entry) { setWbError(`No watched branch at index ${wbIndex}`); return; }
        setWbBranchName(entry.branch);
        setWbBranch(entry.branch_fqn);
        setRepo(entry.repo);
      })
      .catch((e: unknown) => { setWbError(e instanceof Error ? e.message : String(e)); });
  }, [basePath, wbIndex]);

  const refreshPr = useCallback(async () => {
    if (wbBranchName === null) return;
    try {
      setPr(await apiFetch<WatchedBranchPr>(`${basePath}/api/v1alpha/watched_branches/${encodeURIComponent(wbBranchName)}/prs/${prNumber}`));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [basePath, wbBranchName, prNumber]);

  useEffect(() => {
    if (wbBranchName === null) return;
    setLoading(true);
    void refreshPr().finally(() => { setLoading(false); });
  }, [wbBranchName, refreshPr]);

  // A build was requested (new PR or retry) but hasn't been assigned a batch yet — poll until it is.
  useEffect(() => {
    if (!pr || pr.batch || !pr.pending_build_reason || !autoRefresh) return;
    const id = setInterval(() => { void refreshPr(); }, REFRESH_INTERVAL_MS);
    return () => { clearInterval(id); };
  }, [pr, refreshPr, autoRefresh]);

  useEffect(() => {
    if (wbBranch === null) return;
    const q = `test=1 pr=${prNumber} target_branch=${wbBranch} user:ci`;
    fetchAllBatches(batchBaseUrl, q)
      .then(setHistory)
      .catch(() => { /* non-critical */ });
  }, [batchBaseUrl, prNumber, wbBranch]);

  useEffect(() => {
    if (wbBranch === null || !pr || isActivePr(pr) || !pr.merged || !pr.merge_commit_sha) return;
    const q = `deploy=1 target_branch=${wbBranch} sha=${pr.merge_commit_sha} user:ci`;
    fetchAllBatches(batchBaseUrl, q)
      .then(setDeployHistory)
      .catch(() => { /* non-critical */ });
  }, [batchBaseUrl, pr, wbBranch]);

  if (wbError) {
    return <div className="mt-8 text-red-600">Error loading watched branch: {wbError}</div>;
  }

  if (wbBranchName === null || wbBranch === null || repo === null || loading) {
    return (
      <div className="flex items-center justify-center mt-24">
        <span className="text-5xl font-light text-sky-600">Loading&hellip;</span>
      </div>
    );
  }

  if (error || !pr) {
    return <div className="mt-8 text-red-600">Error loading PR: {error ?? 'unknown error'}</div>;
  }

  const active = isActivePr(pr);

  return (
    <div className="pb-8">
      <nav className="flex items-center gap-2 text-xl font-light text-zinc-500 flex-wrap">
        <a href={`${basePath}/`} className="hover:text-sky-600">CI</a>
        <span className="text-zinc-300">›</span>
        <span className="text-zinc-800">{wbBranch}</span>
        <span className="text-zinc-300">›</span>
        <span className="text-zinc-800">
          <a
            href={`https://github.com/${repo}/pull/${prNumber}`}
            target="_blank"
            rel="noopener noreferrer"
            className="underline hover:text-sky-600"
          >
            #{prNumber}
          </a>
          : {pr.title}
        </span>
      </nav>

      {active && pr.batch && batchStatus && !batchStatus.complete && (
        <div className="mt-2">
          <AutoRefreshBar
            autoRefresh={autoRefresh}
            onToggle={setAutoRefresh}
            countdownKey={countdownKey}
            refreshing={refreshing}
            intervalMs={REFRESH_INTERVAL_MS}
          />
        </div>
      )}

      <div className="mt-1 text-sm">
        <button
          type="button"
          onClick={() => { document.cookie = 'hail_react_ui=; max-age=0; path=/; SameSite=Lax'; location.reload(); }}
          className="text-sky-600 hover:underline cursor-pointer"
        >
          Back to classic layout
        </button>
      </div>

      <div className="mt-4 text-sm">
        Status:{' '}
        {active ? (
          <><Icon name="merge" className="text-green-600" /> open (live)</>
        ) : pr.merged ? (
          <><Icon name="merge" className="text-purple-600" /> merged</>
        ) : pr.merged === null ? (
          <><Icon name="question_mark" className="text-zinc-400" /> closed</>
        ) : (
          <><Icon name="block" className="text-red-600" /> closed without merging</>
        )}
      </div>

      {active && <MergeEligibility pr={pr} basePath={basePath} batchBaseUrl={batchBaseUrl} wbIndex={wbIndex} />}
      {active && (
        <BuildPanel
          pr={pr}
          basePath={basePath}
          batchBaseUrl={batchBaseUrl}
          wbBranchName={wbBranchName}
          prNumber={prNumber}
          batchStatus={batchStatus}
          jobs={jobs}
          jobsError={jobsError}
        />
      )}

      {!active && pr.merged && (
        <div className="mt-4">
          <h2 className="text-lg font-semibold text-zinc-700 mb-2">Deploy Batch History</h2>
          <BatchHistoryTable batches={deployHistory} batchBaseUrl={batchBaseUrl} />
        </div>
      )}

      <div className="mt-4">
        <h2 className="text-lg font-semibold text-zinc-700 mb-2">Build Batch History</h2>
        <BatchHistoryTable batches={history} batchBaseUrl={batchBaseUrl} />
      </div>
    </div>
  );
}

const container = document.getElementById('pr-details-root');
if (container) {
  const basePath = container.dataset.basePath ?? '';
  const batchBaseUrl = container.dataset.batchBaseUrl ?? '';
  const wbIndex = container.dataset.watchedBranchIndex ?? '';
  const prNumber = container.dataset.prNumber ?? '';
  createRoot(container).render(
    <PrPage
      basePath={basePath}
      batchBaseUrl={batchBaseUrl}
      wbIndex={wbIndex}
      prNumber={prNumber}
    />
  );
}
