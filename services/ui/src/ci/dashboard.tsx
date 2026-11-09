import { Fragment, useCallback, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  createCiApi,
  type CiApi,
  type CIEvent,
  type CiDeveloperStatus,
  type CiOperatorStatus,
  type DeployStatus,
  type DeveloperBranchStatus,
  type OperatorBranchStatus,
  type RateLimit,
  type WatchedBranchSummary,
} from '../shared/ciApi';
import { hasPermission } from '../shared/authUtils';
import type { Batch } from '../shared/batchApi';
import { BatchStateIcon } from '../shared/BatchStateIcon';
import { AutoRefreshBar } from '../shared/AutoRefreshBar';
import { Panel } from '../shared/Panel';
import { DataTr, EmptyCell, Td, Th } from '../shared/Table';
import { EventLog, MessageCell } from './components/EventLog';
import { currentMergeFailures } from './components/eventLogModel';
import { formatRelative, formatTimestamp } from '../shared/timeUtils';
import {
  deployDisplay,
  failedDeploysKey,
  failedJobs,
  isDeployFailed,
  isRateLimitLow,
  isTraceback,
  mergePausedLabel,
  mergeQueueStatus,
  type Tone,
} from './components/dashboardModel';

const REFRESH_INTERVAL_MS = 30_000;
const AUTO_REFRESH_KEY = 'ci.dashboard.autoRefresh';

function toneClass(tone: Tone): string {
  switch (tone) {
    case 'good': return 'text-green-700';
    case 'bad': return 'text-red-700';
    case 'warn': return 'text-amber-700';
    default: return 'text-zinc-600';
  }
}

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function Icon({ name, className }: { name: string; className: string }): JSX.Element {
  return <span aria-hidden="true" className={`material-symbols-outlined align-middle text-lg ${className}`}>{name}</span>;
}

// Links a PR number to its CI details page.
function PrLink({ pr, prBase }: { pr: number; prBase: string }): JSX.Element {
  return <a href={`${prBase}/${pr}`} className="text-sky-600 hover:underline">#{pr}</a>;
}

function Field({ label, children }: { label: string; children: React.ReactNode }): JSX.Element {
  return (
    <div className="flex gap-2 text-sm py-0.5">
      <span className="text-zinc-500 w-32 shrink-0">{label}</span>
      <span className="text-zinc-800 min-w-0 break-words">{children}</span>
    </div>
  );
}

function Tile({ title, className = '', children }: { title: string; className?: string; children: React.ReactNode }): JSX.Element {
  return (
    <section className={`rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 ${className}`}>
      <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-zinc-500">{title}</h3>
      {children}
    </section>
  );
}

function HealthStrip({ dev, op }: { dev: CiDeveloperStatus; op: CiOperatorStatus | null }): JSX.Element {
  return (
    <div
      className={`h-full rounded-lg border px-4 py-3 ${dev.ci_unhealthy ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'}`}
    >
      <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-sm">
        {dev.ci_unhealthy
          ? <span className="font-semibold text-red-700"><Icon name="error" className="text-red-600" /> CI looks unhealthy</span>
          : <span className="font-semibold text-green-700"><Icon name="check_circle" className="text-green-600" /> CI looks healthy</span>}
        <span className="text-zinc-500" title={formatTimestamp(dev.started_at_ms)}>
          CI started {formatRelative(dev.started_at_ms)}; status below resets on restart
        </span>
      </div>
      {op !== null && op.health_problems.length > 0 && (
        <ul className="mt-2 list-disc pl-6 text-sm text-red-700">
          {op.health_problems.map((p) => <li key={p}>{p}</li>)}
        </ul>
      )}
      {op === null && dev.ci_unhealthy && (
        <p className="mt-1 text-xs text-zinc-500">Details are visible to operators.</p>
      )}
    </div>
  );
}

function ManagePanel({ api, frozen, onChanged }: { api: CiApi; frozen: boolean; onChanged: () => void }): JSX.Element {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ text: string; tone: Tone } | null>(null);

  const toggleFreeze = async (): Promise<void> => {
    const verb = frozen ? 'Unfreeze' : 'Freeze';
    if (!window.confirm(`${verb} all merges and deploys?`)) return;
    setBusy(true);
    try {
      await (frozen ? api.unfreeze() : api.freeze());
      setMessage({ text: frozen ? 'Unfroze merges and deploys.' : 'Froze merges and deploys.', tone: 'good' });
      onChanged();
    } catch (e: unknown) {
      setMessage({ text: errorMessage(e), tone: 'bad' });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Panel title="Manage CI">
      <div className="flex flex-col items-start gap-3 text-sm">
        <div className="flex items-center gap-2">
          <span className="text-zinc-500">Merges &amp; deploys</span>
          {frozen
            ? (
              <span className="rounded-full bg-amber-100 px-2 py-0.5 font-semibold text-amber-800">
                <Icon name="ac_unit" className="text-amber-600" /> Frozen
              </span>
            )
            : (
              <span className="rounded-full bg-green-100 px-2 py-0.5 font-semibold text-green-800">
                <Icon name="check_circle" className="text-green-600" /> Running
              </span>
            )}
        </div>
        <button
          type="button"
          disabled={busy}
          onClick={() => { void toggleFreeze(); }}
          className="px-3 py-1.5 rounded border border-red-300 text-red-700 hover:bg-red-50 disabled:opacity-50"
        >
          {frozen ? 'Unfreeze merges & deploys' : 'Freeze merges & deploys'}
        </button>
      </div>
      {message && <p className={`mt-2 text-sm ${toneClass(message.tone)}`}>{message.text}</p>}
    </Panel>
  );
}

function DeployFailure({ info, batchBaseUrl }: { info: DeployStatus; batchBaseUrl: string }): JSX.Element | null {
  const failure = info.failure_information;
  if (isTraceback(failure)) {
    return (
      <pre className="mt-2 max-h-64 overflow-auto rounded bg-zinc-900 p-3 text-xs text-zinc-100">{failure.join('')}</pre>
    );
  }
  const jobs = failedJobs(failure);
  if (jobs.length === 0) return null;
  return (
    <ul className="mt-1 text-sm">
      {jobs.map((j) => (
        <li key={`${j.batch_id}/${j.job_id}`}>
          <Icon name="close" className="text-red-600" />{' '}
          <a href={`${batchBaseUrl}/batches/${j.batch_id}/jobs/${j.job_id}`} className="text-sky-600 hover:underline">
            {j.name ?? `job ${j.job_id}`}
          </a>{' '}
          <span className="text-zinc-500">{j.state}</span>
        </li>
      ))}
    </ul>
  );
}

function DeploySection({ branch, repo, batchBaseUrl, failure, failureError }: {
  branch: DeveloperBranchStatus;
  repo: string;
  batchBaseUrl: string;
  failure: DeployStatus | undefined;
  failureError: string | null;
}): JSX.Element {
  const deploy = deployDisplay(branch);
  return (
    <Tile title="Deploy">
      <Field label="HEAD">
        {branch.sha
          ? (
            <a
              href={`https://github.com/${repo}/commit/${branch.sha}`}
              target="_blank"
              rel="noopener noreferrer"
              className="font-mono text-sky-600 hover:underline"
            >
              {branch.sha.slice(0, 12)}
            </a>
          )
          : 'unknown'}
      </Field>
      <Field label="State">
        <span className={toneClass(deploy.tone)}>{deploy.label}</span>
        {branch.deploy_batch_id !== null && (
          <>
            {' · '}
            <a href={`${batchBaseUrl}/batches/${branch.deploy_batch_id}`} className="text-sky-600 hover:underline">
              batch {branch.deploy_batch_id}
            </a>
          </>
        )}
      </Field>
      {isDeployFailed(branch.deploy_state) && (
        failureError
          ? <p className="text-sm text-red-600">Error loading deploy failure: {failureError}</p>
          : failure
            ? <DeployFailure info={failure} batchBaseUrl={batchBaseUrl} />
            : <p className="text-sm text-zinc-500">Loading failure details&hellip;</p>
      )}
    </Tile>
  );
}

function MergeSection({ branch, branchFqn, prBase, nowMs, mergeFailedEvents }: {
  branch: DeveloperBranchStatus;
  branchFqn: string | undefined;
  prBase: string;
  nowMs: number;
  mergeFailedEvents: CIEvent[];
}): JSX.Element {
  const paused = mergePausedLabel(branch.merge_paused_reason);
  const failures = branchFqn === undefined
    ? new Map<number, CIEvent>()
    : currentMergeFailures(mergeFailedEvents, branchFqn, branch.sha, branch.merge_queue);
  return (
    <Tile title="Merge queue">
      {paused && <p className="text-sm text-amber-700 mb-1">{paused}</p>}
      {branch.pending_merge && (
        <p className="text-sm text-zinc-700 mb-1">
          Merging <PrLink pr={branch.pending_merge.pr} prBase={prBase} /> (requested{' '}
          {formatRelative(branch.pending_merge.requested_at_ms)})
        </p>
      )}
      {branch.merge_queue.length === 0
        ? <p className="text-sm text-zinc-500">No approved PRs waiting to merge.</p>
        : (
          <ol className="text-sm">
            {branch.merge_queue.map((entry) => {
              const status = mergeQueueStatus(entry, nowMs);
              const failure = failures.get(entry.pr);
              return (
                <li key={entry.pr} className="py-0.5">
                  <div className="flex gap-2">
                    <span className="w-16 shrink-0"><PrLink pr={entry.pr} prBase={prBase} /></span>
                    {entry.pr === branch.merge_candidate && (
                      <span className="shrink-0 rounded bg-sky-100 px-1.5 text-xs leading-5 text-sky-800">candidate</span>
                    )}
                    <span className={toneClass(status.tone)}>{status.text}</span>
                  </div>
                  {failure && (
                    <div className="ml-[4.5rem] text-xs text-zinc-600">
                      <span className="text-red-600" title={formatTimestamp(failure.time)}>
                        merge failed {formatRelative(Date.parse(failure.time))}:
                      </span>{' '}
                      <MessageCell message={failure.message} />
                    </div>
                  )}
                </li>
              );
            })}
          </ol>
        )}
    </Tile>
  );
}

function BuildSlotsSection({ branch, prBase }: { branch: DeveloperBranchStatus; prBase: string }): JSX.Element {
  const slots = branch.build_slots;
  return (
    <Tile title="Build slots">
      <Field label="In use">
        {slots.used} / {slots.max}
        {slots.used > slots.max && <span className="text-zinc-500"> (the merge candidate doesn&rsquo;t count against the limit)</span>}
      </Field>
      <Field label="Running">
        {slots.running.length === 0
          ? <span className="text-zinc-500">none</span>
          : slots.running.map((pr, i) => <span key={pr}>{i > 0 && ', '}<PrLink pr={pr} prBase={prBase} /></span>)}
      </Field>
      <Field label="On deck">
        {slots.on_deck === null ? <span className="text-zinc-500">none</span> : <PrLink pr={slots.on_deck} prBase={prBase} />}
      </Field>
      <Field label="Waiting">
        {slots.queued.length === 0
          ? <span className="text-zinc-500">none</span>
          : (
            <ul>
              {slots.queued.map((q) => (
                <li key={q.pr}><PrLink pr={q.pr} prBase={prBase} /> <span className="text-zinc-500">{q.reason}</span></li>
              ))}
            </ul>
          )}
      </Field>
    </Tile>
  );
}

function RateLimitValue({ rl, extra }: { rl: RateLimit | null; extra?: string }): JSX.Element {
  if (rl === null) return <span className="text-zinc-500">unknown</span>;
  return (
    <span className={isRateLimitLow(rl) ? 'text-red-700' : undefined}>
      {rl.remaining} / {rl.limit} left{extra ? ` · ${extra}` : ''} · resets {formatTimestamp(Date.parse(rl.reset_at))}
    </span>
  );
}

function OperatorBranchTiles({ op }: { op: OperatorBranchStatus }): JSX.Element {
  return (
    <>
      <Tile title="Branch updates" className="xl:col-span-2">
        <Field label="Updating">
          {op.updating && op.update_started_ms !== null ? `yes, since ${formatRelative(op.update_started_ms)}` : 'no'}
        </Field>
        <Field label="Pending changes">{op.dirty.length === 0 ? 'none' : op.dirty.join(', ')}</Field>
        <Field label="Last update error">
          {op.last_update_error === null
            ? 'none'
            : <pre className="whitespace-pre-wrap text-xs text-red-700">{op.last_update_error}</pre>}
        </Field>
        {op.deploy_exception !== null && (
          <Field label="Deploy exception"><span className="text-red-700">{op.deploy_exception}</span></Field>
        )}
      </Tile>
      <Tile title="GitHub rate limits" className="xl:col-span-2">
        <Field label="REST"><RateLimitValue rl={op.rate_limit.rest} /></Field>
        <Field label="GraphQL">
          <RateLimitValue
            rl={op.rate_limit.graphql}
            extra={op.rate_limit.graphql ? `last update cost ${op.rate_limit.graphql.last_update_cost}` : undefined}
          />
        </Field>
      </Tile>
    </>
  );
}

// The whole tile links to the branch's PR list.
function PrCountTile({ href, n }: { href: string; n: number }): JSX.Element {
  return (
    <a
      href={href}
      className="flex flex-col items-center justify-center rounded-md border border-sky-200 bg-sky-50 px-3 py-4 text-sky-700 hover:bg-sky-100"
    >
      <span className="text-4xl font-light tabular-nums">{n}</span>
      <span className="text-sm">open PR{n === 1 ? '' : 's'} &rarr;</span>
    </a>
  );
}

// Hidden for viewers who aren't CI developers, since the endpoint 404s for them.
function DevDeploysPanel({ api, batchBaseUrl, refreshKey }: {
  api: CiApi;
  batchBaseUrl: string;
  refreshKey: number;
}): JSX.Element | null {
  const [deploys, setDeploys] = useState<Batch[] | null>(null);
  useEffect(() => {
    let cancelled = false;
    api.getDevDeploys('me')
      .then((r) => { if (!cancelled) setDeploys(r); })
      .catch((e: unknown) => { console.warn('Could not load dev deploys', e); });
    return () => { cancelled = true; };
  }, [api, refreshKey]);

  if (deploys === null) return null;
  return (
    <Panel title="My dev deploys" subtitle="Your 10 most recent hailctl dev deploy batches">
      {deploys.length === 0 ? (
        <p className="text-sm text-zinc-500">No dev deploys.</p>
      ) : (
        <div className="overflow-x-auto rounded border border-zinc-200">
          <table className="w-full text-sm">
            <thead>
              <tr>
                <Th>Batch</Th>
                <Th>Branch</Th>
                <Th>State</Th>
                <Th>Created</Th>
              </tr>
            </thead>
            <tbody>
              {deploys.map((b) => (
                <DataTr key={b.id}>
                  <Td>
                    <a href={`${batchBaseUrl}/batches/${b.id}`} className="text-sky-600 hover:underline">{b.id}</a>
                  </Td>
                  <Td className="break-all">{b.attributes?.target_branch ?? <EmptyCell />}</Td>
                  <Td className="whitespace-nowrap"><BatchStateIcon state={b.state} /> {b.state}</Td>
                  <Td className="whitespace-nowrap">{formatTimestamp(b.time_created)}</Td>
                </DataTr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

function BranchPanel({ index, branch, summary, op, basePath, batchBaseUrl, failure, failureError, nowMs, mergeFailedEvents }: {
  index: number;
  branch: DeveloperBranchStatus;
  summary: WatchedBranchSummary | undefined;
  op: OperatorBranchStatus | undefined;
  basePath: string;
  batchBaseUrl: string;
  failure: DeployStatus | undefined;
  failureError: string | null;
  nowMs: number;
  mergeFailedEvents: CIEvent[];
}): JSX.Element {
  const prBase = `${basePath}/watched_branches/${index}/pr`;
  const lastUpdate = branch.last_update;
  const subtitle = lastUpdate
    ? `Last updated ${formatRelative(lastUpdate.finished_ms)} (took ${lastUpdate.duration_s.toFixed(1)}s)`
    : 'Not updated since CI started';
  return (
    <Panel title={summary?.branch_fqn ?? branch.branch} subtitle={subtitle}>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        <PrCountTile href={`${basePath}/watched_branches/${index}/prs`} n={branch.n_prs} />
        <DeploySection
          branch={branch}
          repo={summary?.repo ?? ''}
          batchBaseUrl={batchBaseUrl}
          failure={failure}
          failureError={failureError}
        />
        <MergeSection
          branch={branch}
          branchFqn={summary?.branch_fqn}
          prBase={prBase}
          nowMs={nowMs}
          mergeFailedEvents={mergeFailedEvents}
        />
        <BuildSlotsSection branch={branch} prBase={prBase} />
        {op && <OperatorBranchTiles op={op} />}
      </div>
    </Panel>
  );
}

function OperatorPanel({ op }: { op: CiOperatorStatus }): JSX.Element {
  const webhooks = Object.entries(op.github.last_webhook_ms).sort(([a], [b]) => a.localeCompare(b));
  const loops = Object.entries(op.loops).sort(([a], [b]) => a.localeCompare(b));
  const apiError = op.github.last_api_error;
  return (
    <Panel title="CI internals">
      {/* One row when full width; stacked when it sits beside dev deploys on wide screens */}
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3 xl:grid-cols-1">
        <Tile title="Last webhook, by event">
          {webhooks.length === 0
            ? <p className="text-sm text-zinc-500">None since CI started.</p>
            : webhooks.map(([event, ms]) => (
              <Field key={event} label={event}><span title={formatTimestamp(ms)}>{formatRelative(ms)}</span></Field>
            ))}
        </Tile>
        <Tile title="Last GitHub API error">
          {apiError === null
            ? <p className="text-sm text-zinc-500">None since CI started.</p>
            : (
              <>
                <Field label="When"><span title={formatTimestamp(apiError.at_ms)}>{formatRelative(apiError.at_ms)}</span></Field>
                <Field label="Branch">{apiError.branch}</Field>
                <Field label="Operation">{apiError.operation}</Field>
                <Field label="HTTP status">{apiError.status ?? 'none'}</Field>
                <Field label="Message"><span className="text-red-700">{apiError.message}</span></Field>
              </>
            )}
        </Tile>
        <Tile title="Background loops, last success">
          {loops.length === 0
            ? <p className="text-sm text-zinc-500">No background loops in this namespace.</p>
            : (
              // Grid rather than Field: loop names are long identifiers that overflow Field's fixed label width
              <div className="grid grid-cols-[auto_1fr] gap-x-4 text-sm">
                {loops.map(([loop, { last_success_ms }]) => (
                  <Fragment key={loop}>
                    <span className="text-zinc-500 py-0.5">{loop}</span>
                    <span className="text-zinc-800 py-0.5">
                      {last_success_ms === null
                        ? <span className="text-zinc-500">not yet</span>
                        : <span title={formatTimestamp(last_success_ms)}>{formatRelative(last_success_ms)}</span>}
                    </span>
                  </Fragment>
                ))}
              </div>
            )}
        </Tile>
      </div>
    </Panel>
  );
}

const DAY_MS = 24 * 60 * 60 * 1000;

function FilterChip({ label, active, onToggle }: { label: string; active: boolean; onToggle: () => void }): JSX.Element {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onToggle}
      className={`text-xs px-2 py-0.5 rounded-full border transition-colors ${active ? 'bg-sky-500 border-sky-500 text-white' : 'border-zinc-300 text-zinc-500 hover:border-sky-400 hover:text-sky-600 bg-white'}`}
    >
      {label}
    </button>
  );
}

// One feed for every event; the filters stack. PR events are off by default since each PR page has its own
// feed, but PR-scoped errors still show.
function EventsPanel({ basePath, batchBaseUrl, refreshKey, prHref }: {
  basePath: string;
  batchBaseUrl: string;
  refreshKey: number;
  prHref: (e: CIEvent) => string | null;
}): JSX.Element {
  const [errorsOnly, setErrorsOnly] = useState(false);
  const [lastDay, setLastDay] = useState(true);
  const [prEvents, setPrEvents] = useState(false);
  return (
    <Panel
      title="Events"
      viewSelector={
        <div className="flex flex-wrap gap-1.5">
          <FilterChip label="Errors" active={errorsOnly} onToggle={() => { setErrorsOnly((v) => !v); }} />
          <FilterChip label="Last 24h" active={lastDay} onToggle={() => { setLastDay((v) => !v); }} />
          <FilterChip label="PR events" active={prEvents} onToggle={() => { setPrEvents((v) => !v); }} />
        </div>
      }
    >
      <EventLog
        basePath={basePath}
        batchBaseUrl={batchBaseUrl}
        errorsOnly={errorsOnly}
        sinceWindowMs={lastDay ? DAY_MS : undefined}
        includePrEvents={prEvents}
        hideBranch
        refreshKey={refreshKey}
        prHref={prHref}
      />
    </Panel>
  );
}

function readAutoRefresh(): boolean {
  try {
    return localStorage.getItem(AUTO_REFRESH_KEY) !== 'false';
  } catch {
    return true;
  }
}

function DashboardPage({ basePath, batchBaseUrl }: { basePath: string; batchBaseUrl: string }): JSX.Element {
  const [api] = useState(() => createCiApi(basePath));
  const [isOperator] = useState(() => hasPermission('read_deployed_system_state'));
  const [canManage] = useState(() => hasPermission('manage_ci'));

  const [summaries, setSummaries] = useState<WatchedBranchSummary[]>([]);
  const [dev, setDev] = useState<CiDeveloperStatus | null>(null);
  const [op, setOp] = useState<CiOperatorStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deployStatus, setDeployStatus] = useState<DeployStatus[]>([]);
  const [deployError, setDeployError] = useState<string | null>(null);

  const [refreshing, setRefreshing] = useState(false);
  const [countdownKey, setCountdownKey] = useState(0);
  const [autoRefresh, setAutoRefreshState] = useState<boolean>(readAutoRefresh);
  const setAutoRefresh = useCallback((v: boolean) => {
    setAutoRefreshState(v);
    try {
      localStorage.setItem(AUTO_REFRESH_KEY, String(v));
    } catch { /* ignore */ }
  }, []);

  // Branch names and repos for display and links; these don't change while CI runs.
  useEffect(() => {
    let cancelled = false;
    api.getWatchedBranches()
      .then((r) => { if (!cancelled) setSummaries(r.branches); })
      .catch(() => { /* branch names fall back to the status payload's */ });
    return () => { cancelled = true; };
  }, [api]);

  const refresh = useCallback(async (isRefresh: boolean) => {
    if (isRefresh) setRefreshing(true);
    try {
      const [devStatus, opStatus] = await Promise.all([
        api.getDeveloperStatus(),
        isOperator ? api.getOperatorStatus() : Promise.resolve(null),
      ]);
      setDev(devStatus);
      setOp(opStatus);
      setError(null);
    } catch (e: unknown) {
      setError(errorMessage(e));
    } finally {
      setCountdownKey((k) => k + 1);
      if (isRefresh) setRefreshing(false);
    }
  }, [api, isOperator]);

  useEffect(() => { void refresh(false); }, [refresh]);

  useEffect(() => {
    if (!autoRefresh) return;
    const id = setInterval(() => { void refresh(true); }, REFRESH_INTERVAL_MS);
    return () => { clearInterval(id); };
  }, [autoRefresh, refresh]);

  // Recent merge failures, for the reasons shown next to merge-queue entries. Refetched with each
  // status refresh; the reasons are a nicety, so errors are ignored.
  const [mergeFailedEvents, setMergeFailedEvents] = useState<CIEvent[]>([]);
  useEffect(() => {
    let cancelled = false;
    api.getEvents({ event: 'merge_failed', limit: 100 })
      .then((r) => { if (!cancelled) setMergeFailedEvents(r.rows); })
      .catch(() => { /* leave the previous reasons */ });
    return () => { cancelled = true; };
  }, [api, countdownKey]);

  const prHref = useCallback((e: CIEvent) => {
    const summary = summaries.find((s) => s.branch_fqn === e.target_branch);
    return summary === undefined ? null : `${basePath}/watched_branches/${summary.index}/pr/${e.pr_number}`;
  }, [basePath, summaries]);

  const failedKey = dev ? failedDeploysKey(dev.branches) : '';
  useEffect(() => {
    if (failedKey === '') {
      setDeployStatus([]);
      return;
    }
    let cancelled = false;
    api.getDeployStatus()
      .then((r) => {
        if (!cancelled) {
          setDeployStatus(r);
          setDeployError(null);
        }
      })
      .catch((e: unknown) => { if (!cancelled) setDeployError(errorMessage(e)); });
    return () => { cancelled = true; };
  }, [api, failedKey]);

  let body: JSX.Element;
  if (dev === null) {
    body = error
      ? <p className="mt-8 text-red-600">Error loading CI status: {error}</p>
      : (
        <div className="flex items-center justify-center mt-24">
          <span className="text-5xl font-light text-sky-600">Loading&hellip;</span>
        </div>
      );
  } else {
    const nowMs = Date.now();
    body = (
      <>
        {error && <p className="mt-2 text-sm text-red-600">Refresh failed, showing older data: {error}</p>}
        <div className="mt-4 flex flex-col gap-4">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className={canManage ? 'md:col-span-2' : 'md:col-span-3'}>
              <HealthStrip dev={dev} op={op} />
            </div>
            {canManage && <ManagePanel api={api} frozen={dev.frozen} onChanged={() => { void refresh(true); }} />}
          </div>
          {dev.branches.map((branch, i) => {
            const summary = summaries.find((s) => s.branch === branch.branch);
            return (
              <BranchPanel
                key={branch.branch}
                // Status lists branches in watched-branch order, so position is right if summaries didn't load
                index={summary?.index ?? i}
                branch={branch}
                summary={summary}
                op={op?.branches.find((b) => b.branch === branch.branch)}
                basePath={basePath}
                batchBaseUrl={batchBaseUrl}
                failure={deployStatus.at(i)}
                failureError={deployError}
                nowMs={nowMs}
                mergeFailedEvents={mergeFailedEvents}
              />
            );
          })}
          <EventsPanel
            basePath={basePath}
            batchBaseUrl={batchBaseUrl}
            refreshKey={countdownKey}
            prHref={prHref}
          />
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <DevDeploysPanel api={api} batchBaseUrl={batchBaseUrl} refreshKey={countdownKey} />
            {op && <OperatorPanel op={op} />}
          </div>
        </div>
      </>
    );
  }

  return (
    <div className="pb-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-light text-zinc-800">CI</h1>
        </div>
        <div className="w-48">
          <AutoRefreshBar
            autoRefresh={autoRefresh}
            onToggle={setAutoRefresh}
            countdownKey={countdownKey}
            refreshing={refreshing}
            intervalMs={REFRESH_INTERVAL_MS}
          />
        </div>
      </div>
      {body}
    </div>
  );
}

const container = document.getElementById('ci-dashboard-root');
if (container) {
  const basePath = container.dataset.basePath ?? '';
  const batchBaseUrl = container.dataset.batchBaseUrl ?? '';
  createRoot(container).render(<DashboardPage basePath={basePath} batchBaseUrl={batchBaseUrl} />);
}
