import { useState, useEffect, useCallback, useRef } from 'react';
import { Job, Attempt, TERMINAL_STATES } from './jobModels';
import { JobStatusPanel } from './JobStatusPanel';
import { JobTimelineGantt } from './JobTimelineGantt';
import { StateIcon } from './StateIcon';
import { JobSpecPanel } from './JobSpecPanel';
import { AttemptPanel } from './AttemptPanel';
import { CodeBlock } from './CodeBlock';

type TopTab = 'job_spec' | 'raw_status' | 'current_attempt' | string; // string for attempt_id

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

  // Once the initial load completes, if there are no attempts (null response or
  // empty array) and we're still on the sentinel, default to the Job Spec tab.
  useEffect(() => {
    if (!loading && topTab === 'current_attempt' && !(attempts && attempts.length > 0)) {
      setTopTab('job_spec');
    }
  }, [loading, attempts, topTab, setTopTab]);

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
  const isTerminal = TERMINAL_STATES.has(job.state);
  const activeAttempt = attempts?.find((a) => a.attempt_id === topTab) ?? latestAttempt;

  return (
    <div className="pb-8">
      {/* Breadcrumb */}
      <nav className="flex items-center gap-2 text-xl font-light text-zinc-500 flex-wrap">
        <a href={`${basePath}/batches`} className="hover:text-sky-600">Batches</a>
        <span className="text-zinc-300">›</span>
        <a href={`${basePath}/batches/${batchId}`} className="hover:text-sky-600">Batch {batchId}</a>
        <span className="text-zinc-300">›</span>
        <span className="text-zinc-800">
          Job {jobId}{job.attributes?.name ? <span className="text-zinc-400"> ({job.attributes.name})</span> : null}
        </span>
      </nav>
      <div className="mt-1 text-sm">
        <a href={disableReactUrl} className="text-sky-600 hover:underline">Back to classic layout</a>
      </div>

      {/* Top section: metadata + Gantt */}
      <div className="flex flex-wrap justify-between items-start pt-6 gap-4">
        <JobStatusPanel batchId={batchId} jobId={jobId} job={job} latestAttempt={latestAttempt} />
        {attempts && <JobTimelineGantt job={job} attempts={attempts} isTerminal={isTerminal} />}
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
                  // Job is terminal: its state is the definitive answer.
                  // Job is still running: if the attempt already has a reason it
                  // failed/was preempted even though the job hasn't settled yet.
                  isTerminal || !attempt.reason
                    ? <StateIcon state={job.state} />
                    : <span className="material-symbols-outlined text-base leading-none text-red-400">close</span>
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
              attributes={job.attributes}
              instColl={job.inst_coll}
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
              resources={job.spec?.resources}
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
