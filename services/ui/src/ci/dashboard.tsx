import { useCallback, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  createCiApi,
  type CiApi,
  type CiDeveloperStatus,
  type CiOperatorStatus,
  type DeployStatus,
  type DeveloperBranchStatus,
  type OperatorBranchStatus,
  type RateLimit,
  type WatchedBranchSummary,
} from '../shared/ciApi';
import { hasPermission } from '../shared/authUtils';
import { AutoRefreshBar } from '../shared/AutoRefreshBar';
import { Panel } from '../shared/Panel';
import { disableReactUi } from '../shared/reactUiCookie';
import { formatRelative, formatTime } from '../shared/timeUtils';
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
      <span className="text-zinc-500 w-40 shrink-0">{label}</span>
      <span className="text-zinc-800 min-w-0 break-words">{children}</span>
    </div>
  );
}

function SubHeading({ children }: { children: React.ReactNode }): JSX.Element {
  return <h3 className="text-sm font-semibold text-zinc-700 mt-4 mb-1">{children}</h3>;
}

function HealthStrip({ dev, op }: { dev: CiDeveloperStatus; op: CiOperatorStatus | null }): JSX.Element {
  return (
    <div
      className={`mt-4 rounded-lg border px-4 py-3 ${dev.ci_unhealthy ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'}`}
    >
      <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-sm">
        {dev.ci_unhealthy
          ? <span className="font-semibold text-red-700"><Icon name="error" className="text-red-600" /> CI looks unhealthy</span>
          : <span className="font-semibold text-green-700"><Icon name="check_circle" className="text-green-600" /> CI looks healthy</span>}
        {dev.frozen && (
          <span className="font-semibold text-amber-700"><Icon name="ac_unit" className="text-amber-600" /> Merges and deploys frozen</span>
        )}
        <span className="text-zinc-500" title={formatTime(dev.started_at_ms)}>
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
  const [sha, setSha] = useState('');

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

  const authorize = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault();
    const trimmed = sha.trim();
    if (!trimmed) return;
    setBusy(true);
    try {
      await api.authorizeSha(trimmed);
      setMessage({ text: `SHA ${trimmed} authorized.`, tone: 'good' });
      setSha('');
    } catch (err: unknown) {
      setMessage({ text: errorMessage(err), tone: 'bad' });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Panel title="Manage CI">
      <div className="flex flex-wrap items-center gap-6 text-sm">
        <button
          type="button"
          disabled={busy}
          onClick={() => { void toggleFreeze(); }}
          className="px-3 py-1.5 rounded border border-red-300 text-red-700 hover:bg-red-50 disabled:opacity-50"
        >
          {frozen ? 'Unfreeze merges & deploys' : 'Freeze merges & deploys'}
        </button>
        <form onSubmit={(e) => { void authorize(e); }} className="flex items-center gap-2">
          <label htmlFor="authorize-sha" className="text-zinc-600">Authorize SHA</label>
          <input
            id="authorize-sha"
            type="text"
            value={sha}
            onChange={(e) => { setSha(e.target.value); }}
            className="w-80 px-2 py-1 font-mono text-xs border border-zinc-300 rounded focus:outline-none focus:ring-2 focus:ring-sky-500"
          />
          <button
            type="submit"
            disabled={busy || sha.trim() === ''}
            className="px-3 py-1.5 rounded border border-zinc-300 hover:bg-zinc-50 disabled:opacity-50"
          >
            Authorize
          </button>
        </form>
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
    <div>
      <SubHeading>Deploy</SubHeading>
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
    </div>
  );
}

function MergeSection({ branch, prBase, nowMs }: { branch: DeveloperBranchStatus; prBase: string; nowMs: number }): JSX.Element {
  const paused = mergePausedLabel(branch.merge_paused_reason);
  return (
    <div>
      <SubHeading>Merge queue</SubHeading>
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
              return (
                <li key={entry.pr} className="flex gap-2 py-0.5">
                  <span className="w-16 shrink-0"><PrLink pr={entry.pr} prBase={prBase} /></span>
                  {entry.pr === branch.merge_candidate && (
                    <span className="shrink-0 rounded bg-sky-100 px-1.5 text-xs leading-5 text-sky-800">candidate</span>
                  )}
                  <span className={toneClass(status.tone)}>{status.text}</span>
                </li>
              );
            })}
          </ol>
        )}
    </div>
  );
}

function BuildSlotsSection({ branch, prBase }: { branch: DeveloperBranchStatus; prBase: string }): JSX.Element {
  const slots = branch.build_slots;
  return (
    <div>
      <SubHeading>Build slots</SubHeading>
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
      <Field label="Waiting for a slot">
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
    </div>
  );
}

function RateLimitValue({ rl, extra }: { rl: RateLimit | null; extra?: string }): JSX.Element {
  if (rl === null) return <span className="text-zinc-500">unknown</span>;
  return (
    <span className={isRateLimitLow(rl) ? 'text-red-700' : undefined}>
      {rl.remaining} / {rl.limit} left{extra ? ` · ${extra}` : ''} · resets {formatTime(Date.parse(rl.reset_at))}
    </span>
  );
}

function OperatorBranchSection({ op }: { op: OperatorBranchStatus }): JSX.Element {
  return (
    <div className="col-span-full border-t border-zinc-100 pt-2">
      <SubHeading>Operator</SubHeading>
      <Field label="Updating">
        {op.updating && op.update_started_ms !== null ? `yes, since ${formatRelative(op.update_started_ms)}` : 'no'}
      </Field>
      <Field label="Pending changes">{op.dirty.length === 0 ? 'none' : op.dirty.join(', ')}</Field>
      <Field label="Last update error">
        {op.last_update_error === null
          ? 'none'
          : <pre className="whitespace-pre-wrap text-xs text-red-700">{op.last_update_error}</pre>}
      </Field>
      <Field label="REST rate limit"><RateLimitValue rl={op.rate_limit.rest} /></Field>
      <Field label="GraphQL rate limit">
        <RateLimitValue
          rl={op.rate_limit.graphql}
          extra={op.rate_limit.graphql ? `last update cost ${op.rate_limit.graphql.last_update_cost}` : undefined}
        />
      </Field>
      {op.deploy_exception !== null && (
        <Field label="Deploy exception"><span className="text-red-700">{op.deploy_exception}</span></Field>
      )}
      <Field label="pip version">{op.pip_version ?? 'unknown'}</Field>
    </div>
  );
}

function BranchPanel({ index, branch, summary, op, basePath, batchBaseUrl, failure, failureError, nowMs }: {
  index: number;
  branch: DeveloperBranchStatus;
  summary: WatchedBranchSummary | undefined;
  op: OperatorBranchStatus | undefined;
  basePath: string;
  batchBaseUrl: string;
  failure: DeployStatus | undefined;
  failureError: string | null;
  nowMs: number;
}): JSX.Element {
  const prBase = `${basePath}/watched_branches/${index}/pr`;
  const lastUpdate = branch.last_update;
  const subtitle = lastUpdate
    ? `Last updated ${formatRelative(lastUpdate.finished_ms)} (took ${lastUpdate.duration_s.toFixed(1)}s)`
    : 'Not updated since CI started';
  return (
    <Panel
      title={summary?.branch_fqn ?? branch.branch}
      subtitle={subtitle}
      viewSelector={
        <a href={`${basePath}/watched_branches/${index}/prs`} className="text-sm text-sky-600 hover:underline">
          {branch.n_prs} PR{branch.n_prs === 1 ? '' : 's'} &rarr;
        </a>
      }
    >
      <div className="grid grid-cols-1 gap-x-8 lg:grid-cols-3">
        <DeploySection
          branch={branch}
          repo={summary?.repo ?? ''}
          batchBaseUrl={batchBaseUrl}
          failure={failure}
          failureError={failureError}
        />
        <MergeSection branch={branch} prBase={prBase} nowMs={nowMs} />
        <BuildSlotsSection branch={branch} prBase={prBase} />
        {op && <OperatorBranchSection op={op} />}
      </div>
    </Panel>
  );
}

function OperatorPanel({ op }: { op: CiOperatorStatus }): JSX.Element {
  const webhooks = Object.entries(op.github.last_webhook_ms).sort(([a], [b]) => a.localeCompare(b));
  const loops = Object.entries(op.loops).sort(([a], [b]) => a.localeCompare(b));
  const apiError = op.github.last_api_error;
  return (
    <Panel title="CI internals" subtitle="Operators only">
      <div className="grid grid-cols-1 gap-x-8 lg:grid-cols-3">
        <div>
          <SubHeading>Last webhook, by event</SubHeading>
          {webhooks.length === 0
            ? <p className="text-sm text-zinc-500">None since CI started.</p>
            : webhooks.map(([event, ms]) => (
              <Field key={event} label={event}><span title={formatTime(ms)}>{formatRelative(ms)}</span></Field>
            ))}
        </div>
        <div>
          <SubHeading>Last GitHub API error</SubHeading>
          {apiError === null
            ? <p className="text-sm text-zinc-500">None since CI started.</p>
            : (
              <>
                <Field label="When"><span title={formatTime(apiError.at_ms)}>{formatRelative(apiError.at_ms)}</span></Field>
                <Field label="Branch">{apiError.branch}</Field>
                <Field label="Operation">{apiError.operation}</Field>
                <Field label="HTTP status">{apiError.status ?? 'none'}</Field>
                <Field label="Message"><span className="text-red-700">{apiError.message}</span></Field>
              </>
            )}
        </div>
        <div>
          <SubHeading>Background loops, last success</SubHeading>
          {loops.length === 0
            ? <p className="text-sm text-zinc-500">No background loops in this namespace.</p>
            : loops.map(([loop, { last_success_ms }]) => (
              <Field key={loop} label={loop}>
                {last_success_ms === null
                  ? <span className="text-zinc-500">not yet</span>
                  : <span title={formatTime(last_success_ms)}>{formatRelative(last_success_ms)}</span>}
              </Field>
            ))}
        </div>
      </div>
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
        <HealthStrip dev={dev} op={op} />
        <div className="mt-4 flex flex-col gap-4">
          {canManage && <ManagePanel api={api} frozen={dev.frozen} onChanged={() => { void refresh(true); }} />}
          {dev.branches.map((branch, i) => (
            <BranchPanel
              key={branch.branch}
              index={i}
              branch={branch}
              summary={summaries.at(i)}
              op={op?.branches.find((b) => b.branch === branch.branch)}
              basePath={basePath}
              batchBaseUrl={batchBaseUrl}
              failure={deployStatus.at(i)}
              failureError={deployError}
              nowMs={nowMs}
            />
          ))}
          {op && <OperatorPanel op={op} />}
        </div>
      </>
    );
  }

  return (
    <div className="pb-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-light text-zinc-800">CI</h1>
          <button type="button" onClick={disableReactUi} className="text-sm text-sky-600 hover:underline cursor-pointer">
            Back to classic layout
          </button>
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
