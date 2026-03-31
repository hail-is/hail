import { useState, useEffect, useCallback, ReactNode } from 'react';
import { GanttChart, GanttRow } from '../../shared/GanttChart';
import { JobSpecPanel } from './JobSpecPanel';
import { AttemptPanel } from './AttemptPanel';
import { CodeBlock } from './CodeBlock';
import { RelativeTime } from './RelativeTime';

type ContainerTiming = {
  pulling?: { duration?: number | null } | null;
  running?: { duration?: number | null } | null;
  uploading_resource_usage?: { duration?: number | null } | null;
};

type ContainerStatus = {
  name: string;
  state: string;
  short_error?: string | null;
  timing: ContainerTiming;
};

type JobStatus = {
  container_statuses?: {
    input?: ContainerStatus | null;
    main?: ContainerStatus | null;
    output?: ContainerStatus | null;
  };
  error?: string | null;
};

type JobSpec = {
  process?: { type: 'docker' | 'jvm'; image?: string; command?: string[] };
  user_code?: string;
  resources?: Record<string, unknown>;
  env?: { name: string; value: string }[];
  input_files?: [string, string][];
  output_files?: [string, string][];
  always_run?: boolean;
  n_max_attempts?: number;
  network?: string;
  regions?: string[];
};

type Job = {
  id: number;
  batch_id: number;
  state: string;
  exit_code?: number | null;
  duration?: string;
  cost?: string;
  cost_breakdown?: { resource: string; cost: string }[] | null;
  user?: string;
  billing_project?: string;
  always_run?: boolean;
  attributes?: Record<string, string>;
  spec?: JobSpec | null;
  spec_defaulted_fields?: string[];
  status?: JobStatus | null;
};

type Attempt = {
  attempt_id: string;
  instance_name?: string;
  start_time?: string;
  start_time_ms?: number;
  end_time?: string;
  end_time_ms?: number;
  duration?: string;
  duration_ms?: number;
  reason?: string;
};

type TopTab = 'job_spec' | 'raw_status' | string; // string for attempt_id

const PRISM_COLORS = [
  '#5778a4', '#e49444', '#d1615d', '#85b6b2', '#6a9f58',
  '#e7ca60', '#a87c9f', '#f1a2a9', '#967662', '#b8b0ac',
];

const STEP_LABELS = ['setting up network', 'pulling', 'running', 'uploading_log', 'uploading_resource_usage'];

function buildGanttRows(job: Job, attempts: Attempt[]): GanttRow[] {
  const rows: GanttRow[] = [];

  // Prior attempts (grey)
  const priorAttempts = attempts.slice(0, -1);
  for (const a of priorAttempts) {
    if (a.start_time_ms != null && a.end_time_ms != null) {
      rows.push({
        label: `Attempt ${a.attempt_id}`,
        start: new Date(a.start_time_ms),
        end: new Date(a.end_time_ms),
        category: 'prior attempt',
        tooltip: `Attempt ${a.attempt_id}: ${a.reason ?? 'ended'}`,
      });
    }
  }

  // Latest attempt sub-steps
  const latest = attempts[attempts.length - 1];
  if (latest?.start_time_ms != null) {
    const statuses = job.status?.container_statuses ?? {};
    const allSteps = ['input', 'main', 'output'] as const;
    let colorIdx = 0;
    for (const container of allSteps) {
      const cs = statuses[container];
      if (!cs) continue;
      const timing = cs.timing;
      // Reconstruct sub-step start times sequentially from attempt start
      let cursor = latest.start_time_ms;
      const subSteps: [string, number][] = [
        ['pulling', timing.pulling?.duration ?? 0],
        ['running', timing.running?.duration ?? 0],
        ['uploading_resource_usage', timing.uploading_resource_usage?.duration ?? 0],
      ];
      for (const [stepName, durationMs] of subSteps) {
        const dur = durationMs ?? 0;
        if (dur > 0) {
          rows.push({
            label: `${container}/${stepName}`,
            start: new Date(cursor),
            end: new Date(cursor + dur),
            category: `${container}/${stepName}`,
            tooltip: `${container}/${stepName}: ${(dur / 1000).toFixed(1)}s`,
          });
          colorIdx++;
        }
        cursor += dur;
      }
    }
  }

  return rows;
}

function buildColorMap(rows: GanttRow[]): Record<string, string> {
  const categories = [...new Set(rows.map((r) => r.category))];
  const map: Record<string, string> = { 'prior attempt': '#9ca3af' };
  let idx = 0;
  for (const cat of categories) {
    if (cat !== 'prior attempt') {
      map[cat] = PRISM_COLORS[idx % PRISM_COLORS.length];
      idx++;
    }
  }
  return map;
}

function stateColor(state: string): string {
  switch (state) {
    case 'Success': return 'text-green-600';
    case 'Running': case 'Creating': return 'text-sky-600';
    case 'Failed': case 'Error': return 'text-red-600';
    case 'Cancelled': return 'text-zinc-400';
    default: return 'text-zinc-600';
  }
}

const RUNNING_STATES = new Set(['Running', 'Creating']);

function CollapsibleItem({ title, summary, children }: {
  title: string;
  summary?: ReactNode;
  children: ReactNode;
}): JSX.Element {
  const [open, setOpen] = useState(false);
  return (
    <li>
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex justify-between items-center px-4 py-3 text-sm text-left hover:bg-slate-100"
      >
        <span className="font-medium">{title}</span>
        <div className="flex items-center gap-2 text-zinc-400 text-xs">
          {summary != null && <span>{summary}</span>}
          <span>{open ? '▴' : '▾'}</span>
        </div>
      </button>
      {open && <div className="px-4 pb-3">{children}</div>}
    </li>
  );
}

type Props = {
  basePath: string;
  batchId: string;
  jobId: string;
  disableReactUrl: string;
};

export function JobPage({ basePath, batchId, jobId, disableReactUrl }: Props): JSX.Element {
  const [job, setJob] = useState<Job | null>(null);
  const [attempts, setAttempts] = useState<Attempt[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // URL-synced tab state
  const getInitialTab = (): TopTab => {
    const params = new URLSearchParams(window.location.search);
    const tab = params.get('tab');
    if (tab === 'raw_status' || tab === 'job_spec') return tab;
    const attemptId = params.get('attempt_id');
    if (tab === 'attempt' && attemptId) return attemptId;
    return 'job_spec';
  };

  const [topTab, setTopTabState] = useState<TopTab>(getInitialTab);

  const [specSubTab, setSpecSubTab] = useState<'input' | 'main' | 'output'>(() => {
    const params = new URLSearchParams(window.location.search);
    const sub = params.get('subtab');
    return sub === 'input' || sub === 'output' ? sub : 'main';
  });

  const [attemptSubTabs, setAttemptSubTabs] = useState<Record<string, 'details' | 'charts' | 'input' | 'main' | 'output' | 'raw'>>({});

  const setTopTab = useCallback((tab: TopTab) => {
    setTopTabState(tab);
    const params = new URLSearchParams(window.location.search);
    if (tab === 'job_spec' || tab === 'raw_status') {
      params.set('tab', tab);
      params.delete('attempt_id');
    } else {
      params.set('tab', 'attempt');
      params.set('attempt_id', tab);
    }
    window.history.replaceState(null, '', `?${params.toString()}`);
  }, []);

  const updateSpecSubTab = useCallback((sub: 'input' | 'main' | 'output') => {
    setSpecSubTab(sub);
    const params = new URLSearchParams(window.location.search);
    params.set('subtab', sub);
    window.history.replaceState(null, '', `?${params.toString()}`);
  }, []);

  const updateAttemptSubTab = useCallback((attemptId: string, sub: 'details' | 'charts' | 'input' | 'main' | 'output' | 'raw') => {
    setAttemptSubTabs((prev) => ({ ...prev, [attemptId]: sub }));
    const params = new URLSearchParams(window.location.search);
    params.set('subtab', sub);
    window.history.replaceState(null, '', `?${params.toString()}`);
  }, []);

  const fetchData = useCallback(async () => {
    const apiBase = `${basePath}/api/v1alpha/batches/${batchId}/jobs/${jobId}`;
    try {
      const [jobResp, attemptsResp] = await Promise.all([
        fetch(apiBase, { credentials: 'same-origin' }),
        fetch(`${apiBase}/attempts`, { credentials: 'same-origin' }),
      ]);
      if (!jobResp.ok) throw new Error(`Job fetch: HTTP ${jobResp.status}`);
      const [jobData, attemptsData] = await Promise.all([
        jobResp.json() as Promise<Job>,
        attemptsResp.ok ? (attemptsResp.json() as Promise<Attempt[]>) : Promise.resolve(null),
      ]);
      setJob(jobData);
      setAttempts(attemptsData);
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [basePath, batchId, jobId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Auto-refresh for running jobs
  useEffect(() => {
    if (!job || !RUNNING_STATES.has(job.state)) return;
    const id = setInterval(() => fetchData(), 10_000);
    return () => clearInterval(id);
  }, [job, fetchData]);

  if (loading) {
    return (
      <div className="flex items-center justify-center mt-24">
        <span className="text-5xl font-light text-sky-600">Loading…</span>
      </div>
    );
  }

  if (error || !job) {
    return (
      <div className="mt-8 text-red-600">
        Error loading job: {error ?? 'unknown error'}
      </div>
    );
  }

  const latestAttempt = attempts && attempts.length > 0 ? attempts[attempts.length - 1] : null;
  const ganttRows = attempts ? buildGanttRows(job, attempts) : [];
  const colorMap = buildColorMap(ganttRows);
  // Determine active attempt for tab
  const activeAttempt = attempts?.find((a) => a.attempt_id === topTab) ?? latestAttempt;

  return (
    <div className="pb-8">
      {/* Back link */}
      <a href={`${basePath}/batches/${batchId}`} className="pt-8 flex items-center space-x-1 hover:text-sky-600">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
          <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18" />
        </svg>
        <span className="text-xl text-center font-light">Batch {batchId}</span>
      </a>
      <div className="mt-1 text-sm">
        <a href={disableReactUrl} className="text-sky-600 hover:underline">Back to classic layout</a>
      </div>

      {/* Top section: metadata + Gantt */}
      <div className="flex flex-wrap justify-between items-start pt-6 gap-4">
        <div className="w-full lg:basis-1/4 drop-shadow-sm shrink-0">
          <ul className="border border-collapse divide-y bg-slate-50 rounded">
            <li className="p-4">
              <div className="flex w-full justify-between items-center">
                <div className="text-xl font-light">Batch {batchId} Job {jobId}</div>
                <span className={`font-medium ${stateColor(job.state)}`}>{job.state}</span>
              </div>
              {job.attributes?.name && (
                <div className="text-lg font-light py-1 overflow-auto">{job.attributes.name}</div>
              )}
              {job.user && (
                <div className="font-light text-zinc-500 text-sm">Submitted by {job.user}</div>
              )}
              {job.billing_project && (
                <div className="font-light text-zinc-500 text-sm">Billed to {job.billing_project}</div>
              )}
              {job.always_run && (
                <div className="text-sm font-semibold mt-1">Always Run</div>
              )}
              {latestAttempt?.start_time_ms != null && (
                <div className="text-sm text-zinc-400 mt-1">
                  Started <RelativeTime ms={latestAttempt.start_time_ms} />
                </div>
              )}
            </li>
            <CollapsibleItem title="Environment Variables">
              <table className="text-xs w-full">
                <tbody className="divide-y">
                  {(job.spec?.env ?? []).map(({ name, value }) => (
                    <tr key={name}>
                      <td className="py-1 pr-2">{name}</td>
                      <td className="py-1 text-right break-all">{value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CollapsibleItem>
            {job.attributes && Object.keys(job.attributes).length > 0 && (
              <CollapsibleItem title="Attributes">
                <table className="text-xs w-full">
                  <tbody className="divide-y">
                    {Object.entries(job.attributes).map(([k, v]) => (
                      <tr key={k}>
                        <td className="py-1 pr-2 text-zinc-500">{k}</td>
                        <td className="py-1 text-right break-all">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CollapsibleItem>
            )}
            <CollapsibleItem title="Resources">
              <table className="text-xs w-full">
                <tbody className="divide-y">
                  {Object.entries(job.spec?.resources ?? {}).map(([k, v]) => (
                    <tr key={k}>
                      <td className="py-1 pr-2 text-zinc-500">{k}</td>
                      <td className="py-1 text-right">{String(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CollapsibleItem>
            {job.cost && (
              <CollapsibleItem title="Cost" summary={job.cost}>
                {job.cost_breakdown && (
                  <table className="text-xs w-full">
                    <tbody className="divide-y">
                      {job.cost_breakdown.map(({ resource, cost }) => (
                        <tr key={resource}>
                          <td className="py-1 pr-2 text-zinc-500">{resource}</td>
                          <td className="py-1 text-right">{cost}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </CollapsibleItem>
            )}
          </ul>
        </div>

        {ganttRows.length > 0 && (
          <div className="w-full lg:flex-1 bg-slate-100 border rounded overflow-hidden p-2">
            <GanttChart rows={ganttRows} colorMap={colorMap} width={700} />
          </div>
        )}
      </div>

      {/* Main tab section */}
      <div className="mt-6">
        <div className="flex border-b text-lg overflow-auto bg-white">
          <button
            onClick={() => setTopTab('job_spec')}
            className={`px-4 pt-4 pb-2 hover:opacity-100 border-b-2 ${
              topTab === 'job_spec' ? 'border-black' : 'border-transparent opacity-50'
            }`}
          >
            Job Spec
          </button>
          <button
            onClick={() => setTopTab('raw_status')}
            className={`px-4 pt-4 pb-2 hover:opacity-100 border-b-2 ${
              topTab === 'raw_status' ? 'border-black' : 'border-transparent opacity-50'
            }`}
          >
            Raw Status
          </button>
          {(attempts ?? []).map((attempt) => {
            const isLatest = attempt === latestAttempt;
            const isActive = topTab === attempt.attempt_id;
            return (
              <button
                key={attempt.attempt_id}
                onClick={() => setTopTab(attempt.attempt_id)}
                className={`px-4 pt-4 pb-2 hover:opacity-100 border-b-2 flex items-center gap-1 ${
                  isActive ? 'border-black' : 'border-transparent opacity-50'
                }`}
              >
                {isLatest ? (
                  <span className={stateColor(job.state)}>●</span>
                ) : (
                  <span className="text-red-400">✕</span>
                )}
                <span>Attempt {attempt.attempt_id}</span>
              </button>
            );
          })}
        </div>

        <div className="pt-4">
          {topTab === 'job_spec' && (
            <JobSpecPanel
              spec={job.spec ?? null}
              defaultedFields={new Set(job.spec_defaulted_fields ?? [])}
              activeSubTab={specSubTab}
              setActiveSubTab={updateSpecSubTab}
            />
          )}

          {topTab === 'raw_status' && (
            <CodeBlock code={JSON.stringify(job.status ?? job, null, 2)} />
          )}

          {activeAttempt && topTab !== 'job_spec' && topTab !== 'raw_status' && (
            <AttemptPanel
              attempt={activeAttempt}
              batchId={batchId}
              jobId={jobId}
              basePath={basePath}
              isLatest={activeAttempt === latestAttempt}
              activeSubTab={attemptSubTabs[activeAttempt.attempt_id] ?? 'details'}
              setActiveSubTab={(sub) => updateAttemptSubTab(activeAttempt.attempt_id, sub)}
            />
          )}
        </div>
      </div>
    </div>
  );
}
