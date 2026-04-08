import { useState, useEffect, useRef } from 'react';
import { LogViewer } from './LogViewer';
import { CodeBlock } from './CodeBlock';
import { ResourceCharts, ResourceUsageData } from './ResourceCharts';

type Attempt = {
  attempt_id: string;
  instance_name?: string;
  start_time_ms?: number;
  end_time_ms?: number;
  duration?: string;
  reason?: string;
};

type SubTab = 'details' | 'charts' | 'input' | 'main' | 'output' | 'raw';

const ALL_SUB_TABS: { id: SubTab; label: string; requires?: 'input' | 'output' }[] = [
  { id: 'details', label: 'Details' },
  { id: 'charts', label: 'Charts' },
  { id: 'input', label: 'Input Log', requires: 'input' },
  { id: 'main', label: 'Main Log' },
  { id: 'output', label: 'Output Log', requires: 'output' },
  { id: 'raw', label: 'Raw Status' },
];

type LogMap = Record<string, string | null>;

type AttemptData = {
  logs: LogMap;
  resourceUsage: ResourceUsageData | null;
  rawStatus: string | null;
  loading: boolean;
  error: string | null;
};

type Props = {
  attempt: Attempt;
  batchId: string;
  jobId: string;
  basePath: string;
  isLatest: boolean;
  hasInput: boolean;
  hasOutput: boolean;
  resources?: Record<string, unknown>;
  activeSubTab: SubTab;
  setActiveSubTab: (t: SubTab) => void;
  refreshTick: number;
};

async function fetchText(url: string): Promise<string> {
  const resp = await fetch(url, { credentials: 'same-origin' });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.text();
}

async function fetchJson<T>(url: string): Promise<T> {
  const resp = await fetch(url, { credentials: 'same-origin' });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json() as Promise<T>;
}

export function AttemptPanel({
  attempt,
  batchId,
  jobId,
  basePath,
  isLatest,
  hasInput,
  hasOutput,
  resources,
  activeSubTab,
  setActiveSubTab,
  refreshTick,
}: Props): JSX.Element {
  const coresMcpu = resources?.['cores_mcpu'] as number | undefined;
  const memoryBytes = resources?.['memory_bytes'] as number | undefined;
  const storageGib = resources?.['storage_gib'] as number | undefined;
  const cores = coresMcpu != null ? coresMcpu / 1000 : undefined;
  const memoryGib = memoryBytes != null ? (memoryBytes / (1024 ** 3)).toFixed(1) : undefined;
  const cache = useRef<Record<string, AttemptData>>({});
  const committedLogsRef = useRef<LogMap>({});
  // Tracks the last attempt_id the effect ran for so we can distinguish a
  // "same attempt, new poll" refresh from "different attempt now showing".
  const prevAttemptIdRef = useRef<string | null>(null);
  const [data, setData] = useState<AttemptData>({
    logs: {},
    resourceUsage: null,
    rawStatus: null,
    loading: false,
    error: null,
  });
  // Log text that is actually rendered — updated immediately on initial load but
  // held back on auto-refresh so scroll position and text selection are preserved.
  const [committedLogs, setCommittedLogs] = useState<LogMap>({});
  const [hasPendingLogs, setHasPendingLogs] = useState(false);

  const applyLogs = (logs: LogMap) => {
    committedLogsRef.current = logs;
    setCommittedLogs(logs);
    setHasPendingLogs(false);
  };

  useEffect(() => {
    const isFirstRun = prevAttemptIdRef.current === null;
    const attemptJustChanged = !isFirstRun && prevAttemptIdRef.current !== attempt.attempt_id;
    prevAttemptIdRef.current = attempt.attempt_id;

    // A background refresh only happens when the *same* latest attempt is being
    // polled again.  First mount (e.g. arriving from the Job Spec tab) and
    // attempt switches are always treated as fresh loads so that stale logs and
    // any pending-update banner are never shown before real content arrives.
    const isRefreshFetch = !isFirstRun && !attemptJustChanged && refreshTick > 0 && isLatest;

    // Bust the cache only when auto-refreshing the latest attempt.
    if (isRefreshFetch) {
      delete cache.current[attempt.attempt_id];
    }

    const cached = cache.current[attempt.attempt_id];
    if (cached) {
      // Switching to an already-loaded attempt — commit logs immediately.
      setData(cached);
      applyLogs(cached.logs);
      return;
    }

    // Only show the loading spinner on the initial load, not on background refreshes.
    if (!isRefreshFetch) {
      setData((d) => ({ ...d, loading: true, error: null }));
      setHasPendingLogs(false);
    }

    const attemptParam = `?attempt_id=${attempt.attempt_id}`;
    const apiBase = `${basePath}/api/v1alpha/batches/${batchId}/jobs/${jobId}`;

    Promise.all([
      hasInput ? fetchText(`${apiBase}/log/input${attemptParam}`).catch(() => null) : Promise.resolve(null),
      fetchText(`${apiBase}/log/main${attemptParam}`).catch(() => null),
      hasOutput ? fetchText(`${apiBase}/log/output${attemptParam}`).catch(() => null) : Promise.resolve(null),
      fetchJson<ResourceUsageData>(`${apiBase}/resource_usage${attemptParam}`).catch(() => null),
    ]).then(([inputLog, mainLog, outputLog, resourceUsage]) => {
      const result: AttemptData = {
        logs: { input: inputLog, main: mainLog, output: outputLog },
        resourceUsage,
        rawStatus: null,
        loading: false,
        error: null,
      };
      cache.current[attempt.attempt_id] = result;

      if (isRefreshFetch) {
        // Keep the displayed log text stable — only surface a notifier if content changed.
        setData(result);
        const changed = (['input', 'main', 'output'] as const).some(
          (k) => result.logs[k] !== committedLogsRef.current[k]
        );
        if (changed) setHasPendingLogs(true);
      } else {
        setData(result);
        applyLogs(result.logs);
      }
    });
  }, [attempt.attempt_id, batchId, jobId, basePath, hasInput, hasOutput, refreshTick]);

  const apiBase = `${basePath}/api/v1alpha/batches/${batchId}/jobs/${jobId}`;
  const attemptParam = `?attempt_id=${attempt.attempt_id}`;

  return (
    <div>
      <div className="flex border-b text-base overflow-auto bg-white mb-4">
        {ALL_SUB_TABS.filter(({ requires }) =>
          requires === 'input' ? hasInput : requires === 'output' ? hasOutput : true
        ).map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setActiveSubTab(id)}
            className={`px-4 pt-3 pb-2 hover:opacity-100 border-b-2 ${
              activeSubTab === id
                ? 'border-black font-medium'
                : 'border-transparent opacity-50'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {data.loading && (
        <div className="py-8 text-center text-zinc-400 text-sm">Loading…</div>
      )}

      {!data.loading && activeSubTab === 'details' && (
        <table className="text-sm border-collapse w-full max-w-lg">
          <tbody>
            <tr className="border-t">
              <td className="py-2 pr-4 text-zinc-500">Attempt ID</td>
              <td className="py-2 font-mono">{attempt.attempt_id}</td>
            </tr>
            {attempt.instance_name && (
              <tr className="border-t">
                <td className="py-2 pr-4 text-zinc-500">Worker</td>
                <td className="py-2 font-mono">{attempt.instance_name}</td>
              </tr>
            )}
            {attempt.start_time_ms != null && (
              <tr className="border-t">
                <td className="py-2 pr-4 text-zinc-500">Started</td>
                <td className="py-2">{new Date(attempt.start_time_ms).toLocaleString()}</td>
              </tr>
            )}
            {attempt.end_time_ms != null && (
              <tr className="border-t">
                <td className="py-2 pr-4 text-zinc-500">Ended</td>
                <td className="py-2">{new Date(attempt.end_time_ms).toLocaleString()}</td>
              </tr>
            )}
            {attempt.duration && (
              <tr className="border-t">
                <td className="py-2 pr-4 text-zinc-500">Duration</td>
                <td className="py-2">{attempt.duration}</td>
              </tr>
            )}
            {attempt.reason && (
              <tr className="border-t">
                <td className="py-2 pr-4 text-zinc-500">Reason</td>
                <td className="py-2">{attempt.reason}</td>
              </tr>
            )}
            {cores != null && (
              <tr className="border-t">
                <td className="py-2 pr-4 text-zinc-500">Cores allocated</td>
                <td className="py-2">{cores}</td>
              </tr>
            )}
            {memoryGib != null && (
              <tr className="border-t">
                <td className="py-2 pr-4 text-zinc-500">Memory allocated</td>
                <td className="py-2">{memoryGib} GiB</td>
              </tr>
            )}
            {storageGib != null && (
              <tr className="border-t">
                <td className="py-2 pr-4 text-zinc-500">Storage allocated</td>
                <td className="py-2">{storageGib} GiB</td>
              </tr>
            )}
          </tbody>
        </table>
      )}

      {!data.loading && activeSubTab === 'charts' && (
        <div>
          {data.resourceUsage ? (
            <ResourceCharts data={data.resourceUsage} />
          ) : (
            <div className="text-zinc-400 text-sm py-4">No resource usage data.</div>
          )}
        </div>
      )}

      {!data.loading &&
        (['input', 'main', 'output'] as const).map((step) =>
          activeSubTab === step ? (
            <div key={step}>
              {committedLogs[step] != null ? (
                <LogViewer
                  text={committedLogs[step]!}
                  downloadUrl={`${apiBase}/log/${step}${attemptParam}`}
                  downloadName={`batch-${batchId}-${jobId}-${step}.log`}
                  hasPendingUpdate={hasPendingLogs}
                  onLoadUpdate={() => applyLogs(data.logs)}
                />
              ) : (
                <div className="text-zinc-400 text-sm py-4">No {step} log available.</div>
              )}
            </div>
          ) : null
        )}

      {!data.loading && activeSubTab === 'raw' && (
        <div>
          {data.rawStatus != null ? (
            <CodeBlock code={data.rawStatus} />
          ) : (
            <div className="text-zinc-400 text-sm py-4">No raw status available.</div>
          )}
        </div>
      )}
    </div>
  );
}
