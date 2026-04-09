import { Job, Attempt } from './jobModels';
import { stateColor } from './StateIcon';
import { CollapsibleItem } from './CollapsibleItem';
import { RelativeTime } from './RelativeTime';

const MIN_VISIBLE_COST = 0.01;

function CostDisplay({ cost }: { cost: number }): JSX.Element {
  if (cost > 0 && cost < MIN_VISIBLE_COST) {
    return (
      <span title={`$${cost}`} className="cursor-help">
        {`< $${MIN_VISIBLE_COST.toFixed(2)}`}
      </span>
    );
  }
  return <>${cost.toFixed(2)}</>;
}

type Props = {
  batchId: string;
  jobId: string;
  job: Job;
  latestAttempt: Attempt | null;
  autoRefresh: boolean;
  isTerminal: boolean;
  onAutoRefreshToggle: (checked: boolean) => void;
  countdownKey: number;
  refreshIntervalMs: number;
  jobRefreshing: boolean;
};

export function JobStatusPanel({ batchId, jobId, job, latestAttempt, autoRefresh, isTerminal, onAutoRefreshToggle, countdownKey, refreshIntervalMs, jobRefreshing }: Props): JSX.Element {
  return (
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
          {!isTerminal && (
            <div className="mt-2">
              <label className="flex items-center gap-1.5 text-zinc-500 text-sm cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(e) => onAutoRefreshToggle(e.target.checked)}
                  className="cursor-pointer"
                />
                Auto-refresh
                {autoRefresh && jobRefreshing && (
                  <span className="material-symbols-outlined text-sm animate-spin text-sky-400" style={{ animationDuration: '1s' }}>
                    progress_activity
                  </span>
                )}
              </label>
              <div className="mt-1.5 h-0.5 bg-zinc-200 rounded-full overflow-hidden">
                {autoRefresh && !jobRefreshing && (
                  <div
                    key={countdownKey}
                    className="h-full bg-sky-400 origin-left"
                    style={{ animation: `countdown-grow ${refreshIntervalMs}ms linear forwards` }}
                  />
                )}
                {autoRefresh && jobRefreshing && (
                  <div className="h-full bg-sky-300 w-full animate-pulse" />
                )}
              </div>
            </div>
          )}
          {job.user && (
            <div className="font-light text-zinc-500 text-sm mt-2">Submitted by {job.user}</div>
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
        {job.cost != null && (
          <CollapsibleItem title="Cost" summary={<CostDisplay cost={job.cost} />}>
            {job.cost_breakdown && (
              <table className="text-xs w-full">
                <tbody className="divide-y">
                  {job.cost_breakdown.map(({ resource, cost }) => (
                    <tr key={resource}>
                      <td className="py-1 pr-2 text-zinc-500">{resource}</td>
                      <td className="py-1 text-right">
                        <CostDisplay cost={cost} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CollapsibleItem>
        )}
      </ul>
    </div>
  );
}
