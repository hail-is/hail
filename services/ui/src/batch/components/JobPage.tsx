import { useState, useEffect, useCallback, useRef, ReactNode } from 'react';
import { GanttChart, GanttRow } from '../../shared/GanttChart';
import { JobSpecPanel } from './JobSpecPanel';
import { AttemptPanel } from './AttemptPanel';
import { CodeBlock } from './CodeBlock';
import { RelativeTime } from './RelativeTime';

type TimingEntry = {
  start_time?: number | null;
  finish_time?: number | null;
  duration?: number | null;
};

type ContainerTiming = {
  [key: string]: TimingEntry | null | undefined;
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

type TopTab = 'job_spec' | 'raw_status' | 'current_attempt' | string; // string for attempt_id

// px.colors.qualitative.Prism from Plotly — same palette used in the classic job page
const PLOTLY_PRISM = [
  '#5F4690', '#1D6996', '#38A6A5', '#0F8554', '#73AF48',
  '#EDAD08', '#E17C05', '#CC503E', '#94346E', '#6F4070', '#994E95', '#666666',
];

// Fixed color assignments matching the classic Plotly chart (task_names order)
const STEP_COLOR_MAP: Record<string, string> = {
  'pulling':                   PLOTLY_PRISM[0],
  'setting up overlay':        PLOTLY_PRISM[1],
  'setting up network':        PLOTLY_PRISM[2],
  'running':                   PLOTLY_PRISM[3],
  'uploading_log':             PLOTLY_PRISM[4],
  'uploading_resource_usage':  PLOTLY_PRISM[5],
  'prior attempt':             '#9ca3af',
  'scheduling':                '#e2e8f0',
};

type GanttData = {
  rows: GanttRow[];
  ruleXs: { x: Date; label: string }[];
};

function buildGanttRows(job: Job, attempts: Attempt[], showPriorAttempts: boolean): GanttData {
  const rows: GanttRow[] = [];
  const ruleXs: { x: Date; label: string }[] = [];
  const nowMs = Date.now();

  // Prior attempts: one coarse bar per attempt row (all except the last)
  if (showPriorAttempts) {
    for (const a of attempts.slice(0, -1)) {
      const startMs = a.start_time_ms;
      if (startMs == null) continue;
      // A prior attempt with a reason is complete — don't stretch its bar to nowMs
      const isComplete = a.reason != null || a.end_time_ms != null;
      const endMs = a.end_time_ms ?? (isComplete ? startMs : nowMs);
      const durationMs = endMs - startMs;
      rows.push({
        label: `attempt ${a.attempt_id.slice(0, 8)}`,
        start: new Date(startMs),
        end: new Date(endMs),
        category: 'prior attempt',
        tooltip: [
          `Row: attempt ${a.attempt_id.slice(0, 8)}`,
          `Task: prior attempt`,
          `Start: ${new Date(startMs).toLocaleString()}`,
          `Finish: ${new Date(endMs).toLocaleString()}`,
          `Duration: ${(durationMs / 1000).toFixed(1)}s`,
          a.reason ? `Detail: ${a.reason}` : '',
        ].filter(Boolean).join('\n'),
      });
    }
  }

  // Latest attempt: all containers combined onto a single row
  const latest = attempts[attempts.length - 1];
  if (latest?.start_time_ms != null) {
    const latestLabel = `attempt ${latest.attempt_id.slice(0, 8)}`;
    const isLatestComplete = latest.reason != null || latest.end_time_ms != null;
    const latestFallbackEndMs = isLatestComplete ? (latest.end_time_ms ?? latest.start_time_ms) : nowMs;
    const statuses = job.status?.container_statuses ?? {};

    // Find the earliest task start across all containers to bound the scheduling block
    let firstTaskStartMs: number | null = null;
    for (const container of ['input', 'main', 'output'] as const) {
      const cs = statuses[container];
      if (!cs) continue;
      for (const [, timingData] of Object.entries(cs.timing)) {
        if (!timingData || timingData.start_time == null) continue;
        if (firstTaskStartMs === null || timingData.start_time < firstTaskStartMs) {
          firstTaskStartMs = timingData.start_time;
        }
      }
    }

    // Scheduling placeholder: time from attempt start to first task start
    const schedEndMs = firstTaskStartMs ?? latestFallbackEndMs;
    if (schedEndMs > latest.start_time_ms) {
      const durationMs = schedEndMs - latest.start_time_ms;
      rows.push({
        label: latestLabel,
        start: new Date(latest.start_time_ms),
        end: new Date(schedEndMs),
        category: 'scheduling',
        tooltip: [
          `Row: ${latestLabel}`,
          `Task: scheduling`,
          `Start: ${new Date(latest.start_time_ms).toLocaleString()}`,
          `Finish: ${new Date(schedEndMs).toLocaleString()}`,
          `Duration: ${(durationMs / 1000).toFixed(1)}s`,
        ].join('\n'),
      });
      ruleXs.push({ x: new Date(latest.start_time_ms), label: 'scheduling' });
    }

    // Task bars — all containers merged onto the same row
    for (const container of ['input', 'main', 'output'] as const) {
      const cs = statuses[container];
      if (!cs) continue;
      let containerStartMs: number | null = null;
      for (const [stepName, timingData] of Object.entries(cs.timing)) {
        if (!timingData || timingData.start_time == null) continue;
        const startMs = timingData.start_time;
        const endMs = timingData.finish_time ?? latestFallbackEndMs;
        const durationMs = timingData.duration ?? (endMs - startMs);
        rows.push({
          label: latestLabel,
          start: new Date(startMs),
          end: new Date(endMs),
          category: stepName,
          tooltip: [
            `Row: ${latestLabel}`,
            `Task: ${stepName} (${container})`,
            `Start: ${new Date(startMs).toLocaleString()}`,
            `Finish: ${new Date(endMs).toLocaleString()}`,
            `Duration: ${(durationMs / 1000).toFixed(1)}s`,
          ].join('\n'),
        });
        if (containerStartMs === null || startMs < containerStartMs) {
          containerStartMs = startMs;
        }
      }
      // One rule per container division, at its earliest task start
      if (containerStartMs !== null) {
        ruleXs.push({ x: new Date(containerStartMs), label: container });
      }
    }
  }

  return { rows, ruleXs };
}

function buildColorMap(rows: GanttRow[]): Record<string, string> {
  const map: Record<string, string> = {};
  for (const cat of new Set(rows.map((r) => r.category))) {
    map[cat] = STEP_COLOR_MAP[cat] ?? '#9ca3af';
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

function stateIcon(state: string): string {
  switch (state) {
    case 'Success': return 'check';
    case 'Failed': case 'Error': return 'close';
    case 'Cancelled': return 'close';
    default: return 'schedule';
  }
}

function StateIcon({ state }: { state: string }): JSX.Element {
  if (state === 'Running') {
    return (
      <svg className="animate-spin h-4 w-4 text-sky-600 flex-shrink-0" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
      </svg>
    );
  }
  const icon = stateIcon(state);
  const color = icon === 'schedule' ? 'text-zinc-400' : stateColor(state);
  return <span className={`material-symbols-outlined text-base leading-none ${color}`}>{icon}</span>;
}

const TERMINAL_STATES = new Set(['Success', 'Failed', 'Error', 'Cancelled']);

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
  const [showPriorAttempts, setShowPriorAttempts] = useState(true);
  const [refreshTick, setRefreshTick] = useState(0);
  const isInitialLoad = useRef(true);


  // URL-synced tab state
  const getInitialTab = (): TopTab => {
    const params = new URLSearchParams(window.location.search);
    const tab = params.get('tab');
    if (tab === 'raw_status' || tab === 'job_spec') return tab;
    const attemptId = params.get('attempt_id');
    if (tab === 'attempt' && attemptId) return attemptId;
    // Default: show the latest attempt once loaded. Use a sentinel since the
    // attempt ID is not yet known at initialisation time.
    return 'current_attempt';
  };

  const [topTab, setTopTabState] = useState<TopTab>(getInitialTab);

  const [specSubTab, setSpecSubTab] = useState<'input' | 'main' | 'output'>(() => {
    const params = new URLSearchParams(window.location.search);
    const sub = params.get('subtab');
    return sub === 'input' || sub === 'output' ? sub : 'main';
  });

  const [attemptSubTabs, setAttemptSubTabs] = useState<Record<string, 'details' | 'charts' | 'input' | 'main' | 'output' | 'raw'>>(() => {
    const params = new URLSearchParams(window.location.search);
    const attemptId = params.get('attempt_id');
    const sub = params.get('subtab');
    const validSub = sub === 'details' || sub === 'charts' || sub === 'input' || sub === 'main' || sub === 'output' || sub === 'raw' ? sub : null;
    if (attemptId && validSub) return { [attemptId]: validSub };
    return {};
  });

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
      if (!isInitialLoad.current) {
        setRefreshTick((t) => t + 1);
      }
      isInitialLoad.current = false;
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [basePath, batchId, jobId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Auto-refresh for non-terminal jobs
  useEffect(() => {
    if (!job || TERMINAL_STATES.has(job.state)) return;
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
  const hasPriorAttempts = attempts != null && attempts.length > 1;
  const { rows: ganttRows, ruleXs: ganttRuleXs } = attempts
    ? buildGanttRows(job, attempts, showPriorAttempts)
    : { rows: [], ruleXs: [] };
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
          <div className="w-full lg:flex-1 bg-slate-100 border rounded overflow-hidden max-h-[32rem] overflow-y-auto">
            {hasPriorAttempts && (
              <div className="flex items-center gap-2.5 px-3 pt-2 pb-1">
                <button
                  type="button"
                  role="switch"
                  aria-checked={showPriorAttempts}
                  onClick={() => setShowPriorAttempts((v) => !v)}
                  className={`relative h-5 w-9 flex-shrink-0 rounded-full p-0 transition-colors duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 ${showPriorAttempts ? 'bg-sky-600' : 'bg-slate-300'}`}
                >
                  <span
                    className="absolute h-4 w-4 rounded-full bg-white shadow-sm"
                    style={{ top: '2px', left: showPriorAttempts ? '18px' : '2px', transition: 'left 200ms ease-in-out' }}
                  />
                </button>
                <button
                  type="button"
                  className="text-sm text-slate-600 bg-transparent border-none p-0 cursor-pointer select-none focus:outline-none focus-visible:underline"
                  onClick={() => setShowPriorAttempts((v) => !v)}
                >
                  Show prior attempts
                </button>
              </div>
            )}
            <GanttChart rows={ganttRows} colorMap={colorMap} ruleXs={ganttRuleXs} extendToNow={!TERMINAL_STATES.has(job.state)} />
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
          {[...(attempts ?? [])].reverse().map((attempt) => {
            const isLatest = attempt === latestAttempt;
            const isActive = topTab === attempt.attempt_id || (topTab === 'current_attempt' && isLatest);
            return (
              <button
                key={attempt.attempt_id}
                onClick={() => setTopTab(attempt.attempt_id)}
                className={`px-4 pt-4 pb-2 hover:opacity-100 border-b-2 flex items-center gap-1 ${
                  isActive ? 'border-black' : 'border-transparent opacity-50'
                }`}
              >
                {isLatest ? (
                  <StateIcon state={job.state} />
                ) : (
                  <span className="material-symbols-outlined text-base leading-none text-red-400">close</span>
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
              hasInput={(job.spec?.input_files ?? []).length > 0}
              hasOutput={(job.spec?.output_files ?? []).length > 0}
              activeSubTab={attemptSubTabs[activeAttempt.attempt_id] ?? 'main'}
              setActiveSubTab={(sub) => updateAttemptSubTab(activeAttempt.attempt_id, sub)}
              refreshTick={refreshTick}
            />
          )}
        </div>
      </div>
    </div>
  );
}
