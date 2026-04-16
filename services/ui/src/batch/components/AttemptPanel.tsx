import { useEffect } from 'react';
import { LogViewer } from './LogViewer';
import { ResourceCharts } from './ResourceCharts';
import { AttemptCache } from '../hooks/useJobDetails';
import { Attempt } from './jobModels';

type SubTab = 'details' | 'charts' | 'input' | 'main' | 'output';

const ALL_SUB_TABS: { id: SubTab; label: string; requires?: 'input' | 'output' }[] = [
  { id: 'details', label: 'Details' },
  { id: 'charts', label: 'Charts' },
  { id: 'input', label: 'Input Log', requires: 'input' },
  { id: 'main', label: 'Main Log' },
  { id: 'output', label: 'Output Log', requires: 'output' },
];

interface Props {
  attempt: Attempt;
  batchId: string;
  jobId: string;
  basePath: string;
  hasInput: boolean;
  hasOutput: boolean;
  resources?: Record<string, unknown>;
  activeSubTab: SubTab;
  setActiveSubTab: (_tab: SubTab) => void;
  attemptData: AttemptCache;
  onEnsureLoaded: () => void;
  onCommitLogs: () => void;
}

export function AttemptPanel({
  attempt,
  batchId,
  jobId,
  basePath,
  hasInput,
  hasOutput,
  resources,
  activeSubTab,
  setActiveSubTab,
  attemptData,
  onEnsureLoaded,
  onCommitLogs,
}: Props): JSX.Element {
  const coresMcpu = resources?.cores_mcpu as number | undefined;
  const memoryBytes = resources?.memory_bytes as number | undefined;
  const storageGib = resources?.storage_gib as number | undefined;
  const cores = coresMcpu != null ? coresMcpu / 1000 : undefined;
  const memoryGib = memoryBytes != null ? (memoryBytes / (1024 ** 3)).toFixed(1) : undefined;

  useEffect(() => {
    onEnsureLoaded();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attempt.attempt_id]);

  return (
    <div>
      <div className="flex border-b text-base overflow-auto bg-white mb-4">
        {ALL_SUB_TABS.filter(({ requires }) =>
          requires === 'input' ? hasInput : requires === 'output' ? hasOutput : true
        ).map(({ id, label }) => (
          <button
            key={id}
            onClick={() => { setActiveSubTab(id); }}
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

      {attemptData.loading && (
        <div className="py-8 text-center text-zinc-400 text-sm">Loading…</div>
      )}

      {!attemptData.loading && activeSubTab === 'details' && (
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

      {!attemptData.loading && activeSubTab === 'charts' && (
        <div>
          {attemptData.resourceUsage ? (
            <ResourceCharts data={attemptData.resourceUsage} />
          ) : (
            <div className="text-zinc-400 text-sm py-4">No resource usage data.</div>
          )}
        </div>
      )}

      {!attemptData.loading &&
        (['input', 'main', 'output'] as const).map((step) => {
          if (activeSubTab !== step) return null;
          const logText = attemptData.committedLogs.get(step);
          return (
            <div key={step}>
              {logText != null ? (
                <LogViewer
                  text={logText}
                  downloadUrl={`${basePath}/api/v1alpha/batches/${batchId}/jobs/${jobId}/log/${step}?attempt_id=${attempt.attempt_id}`}
                  downloadName={`batch-${batchId}-${jobId}-${step}.log`}
                  hasPendingUpdate={attemptData.hasPendingLogs}
                  onLoadUpdate={() => { onCommitLogs(); }}
                  isRefreshing={attemptData.isRefreshing}
                />
              ) : (
                <div className="text-zinc-400 text-sm py-4">No {step} log available.</div>
              )}
            </div>
          );
        })}
    </div>
  );
}
