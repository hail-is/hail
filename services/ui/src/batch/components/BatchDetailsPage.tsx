import { useState } from 'react';
import type { Batch, BatchJob } from '../../shared/batchApi';
import { formatDurationMs } from './batchModels';
import { StateIcon, BatchStateIcon } from './StateIcon';
import { CollapsibleItem } from './CollapsibleItem';
import { CostDisplay } from './CostDisplay';
import { AutoRefreshBar } from '../../shared/AutoRefreshBar';
import { SpinnerIcon } from '../../shared/SpinnerIcon';
import { formatIsoTime } from '../../shared/timeUtils';
import { QueryBuilder } from '../../shared/QueryBuilder';
import { jobQueryFields } from '../../shared/queryFields';
import { useBatchDetails } from '../hooks/useBatchDetails';

interface Props {
  basePath: string;
  batchId: string;
}

function BillingProjectBadge({ billingProject, accruedCost, limit }: {
  billingProject: string;
  accruedCost: number;
  limit?: number | null;
}): JSX.Element {
  const fraction = limit != null && limit > 0 ? accruedCost / limit : limit === 0 ? 1 : null;
  const level = fraction == null ? 'no_limit' : fraction >= 1 ? 'error' : fraction >= 0.8 ? 'warning' : 'ok';
  return (
    <span className="relative group cursor-default inline-flex items-center gap-0.5">
      <span className="material-symbols-outlined text-base text-zinc-400">info</span>
      <span className="underline decoration-dotted decoration-zinc-400">{billingProject}</span>
      <span className="absolute left-0 bottom-full mb-1 hidden group-hover:block bg-zinc-800 text-white text-xs rounded px-2 py-1 whitespace-nowrap z-10 pointer-events-none">
        {level === 'error' ? 'Billing project over limit: ' : level === 'warning' ? 'Billing project nearing limit: ' : 'Billing project spend: '}
        <CostDisplay cost={accruedCost} />{limit != null && <> / <CostDisplay cost={limit} /></>}
      </span>
      {level === 'error' && <span className="material-symbols-outlined text-base text-red-600">error</span>}
      {level === 'warning' && <span className="material-symbols-outlined text-base text-yellow-500">warning</span>}
    </span>
  );
}

function StatusFilterLinks({ onFilter, onExclude }: { onFilter: () => void; onExclude: () => void }): JSX.Element {
  return (
    <>
      <button type="button" onClick={onFilter} className="hover:text-sky-600" title="Filter to these jobs">
        <span className="material-symbols-outlined text-sm">filter_alt</span>
      </button>
      <button type="button" onClick={onExclude} className="hover:text-sky-600" title="Filter to everything else">
        <span className="material-symbols-outlined text-sm">filter_alt_off</span>
      </button>
    </>
  );
}

function StatusCountRow({ label, count, onFilter, onExclude }: {
  label: string;
  count: number;
  onFilter: () => void;
  onExclude: () => void;
}): JSX.Element {
  return (
    <tr>
      <td className="py-1 pr-2 text-zinc-500">{label}</td>
      <td className="py-1 text-right">
        <div className="inline-flex items-center gap-1">
          {count}
          {count > 0 && <StatusFilterLinks onFilter={onFilter} onExclude={onExclude} />}
        </div>
      </td>
    </tr>
  );
}

function JobStatusTable({ batch, onSearch }: { batch: Batch; onSearch: (q: string) => void }): JSX.Element {
  const incomplete = batch.n_jobs - batch.n_completed;
  return (
    <table className="text-xs w-full">
      <tbody className="divide-y">
        <StatusCountRow
          label="Incomplete (Blocked, Queued or Running)"
          count={incomplete}
          onFilter={() => { onSearch('state != success\nstate != bad\nstate != cancelled'); }}
          onExclude={() => { onSearch('state != pending\nstate != ready\nstate != creating\nstate != running'); }}
        />
        <StatusCountRow
          label="Succeeded"
          count={batch.n_succeeded}
          onFilter={() => { onSearch('state = success'); }}
          onExclude={() => { onSearch('state != success'); }}
        />
        <StatusCountRow
          label="Failure or error"
          count={batch.n_failed}
          onFilter={() => { onSearch('state = bad'); }}
          onExclude={() => { onSearch('state != bad'); }}
        />
        <StatusCountRow
          label="Cancelled"
          count={batch.n_cancelled}
          onFilter={() => { onSearch('state = cancelled'); }}
          onExclude={() => { onSearch('state != cancelled'); }}
        />
      </tbody>
    </table>
  );
}

function JobRow({ basePath, job }: { basePath: string; job: BatchJob }): JSX.Element {
  return (
    <tr className="border border-collapse hover:bg-slate-100">
      <td className="font-light pl-4 w-20">
        <a
          href={`${basePath}/batches/${job.batch_id}/jobs/${job.job_id}`}
          className="hover:text-sky-600 hover:underline underline-offset-2"
        >
          {job.job_id}
        </a>
      </td>
      <td className="py-1 block overflow-x-auto">
        <div className="flex flex-col space-x-0 md:space-x-2 md:flex-row md:flex-wrap items-start md:items-center">
          <a
            href={`${basePath}/batches/${job.batch_id}/jobs/${job.job_id}`}
            className={`hover:text-sky-600 hover:underline underline-offset-2 ${job.name ? '' : 'text-zinc-400 italic'}`}
          >
            {job.name ?? 'no name'}
          </a>
          <span className="flex items-center">
            <StateIcon state={job.state} />
          </span>
        </div>
      </td>
      <td className="hidden lg:table-cell font-light">
        {job.exit_code ?? (job.state === 'Running' ? <SpinnerIcon className="text-zinc-400" /> : <span className="text-zinc-400">--</span>)}
      </td>
      <td className="hidden lg:table-cell font-light">
        {job.duration != null
          ? formatDurationMs(job.duration)
          : job.state === 'Running' ? <SpinnerIcon className="text-zinc-400" /> : <span className="text-zinc-400">--</span>}
      </td>
      <td className="hidden md:table-cell font-light"><CostDisplay cost={job.cost} /></td>
    </tr>
  );
}

export function BatchDetailsPage({ basePath, batchId }: Props): JSX.Element {
  const {
    batch,
    jobs,
    billingProjectInfo,
    error,
    loading,
    autoRefresh,
    setAutoRefresh,
    countdownKey,
    refreshIntervalMs,
    batchRefreshing,
    jobsLoading,
    q,
    hasPreviousPage,
    lastJobId,
    setSearch,
    goToNextPage,
    goToPreviousPage,
    cancelBatch,
    deleteBatch,
  } = useBatchDetails(basePath, batchId);

  const [actionError, setActionError] = useState<string | null>(null);
  const [queryDirty, setQueryDirty] = useState(false);

  if (loading) {
    return (
      <div className="flex items-center justify-center mt-24">
        <span className="text-5xl font-light text-sky-600">Loading…</span>
      </div>
    );
  }

  if (!batch) {
    return (
      <div className="mt-8 text-red-600">
        Error loading batch: {error ?? 'unknown error'}
      </div>
    );
  }

  const handleCancel = () => {
    setActionError(null);
    cancelBatch().catch((e: unknown) => { setActionError(String(e)); });
  };

  const handleDelete = () => {
    setActionError(null);
    deleteBatch()
      .then(() => { window.location.href = `${basePath}/batches`; })
      .catch((e: unknown) => { setActionError(String(e)); });
  };

  return (
    <div className="pb-8">
      {/* Breadcrumb */}
      <nav className="flex items-center gap-2 text-xl font-light text-zinc-500 flex-wrap">
        <a href={`${basePath}/batches`} className="hover:text-sky-600">Batches</a>
        <span className="text-zinc-300">›</span>
        <span className="text-zinc-800">
          Batch {batchId}{batch.attributes?.name ? <span className="text-zinc-400"> ({batch.attributes.name})</span> : null}
        </span>
      </nav>
      <div className="mt-1 text-sm">
        <button
          type="button"
          onClick={() => { document.cookie = 'hail_react_ui=; max-age=0; path=/; SameSite=Lax'; location.reload(); }}
          className="text-sky-600 hover:underline cursor-pointer"
        >
          Back to classic layout
        </button>
      </div>

      <div className="flex flex-wrap justify-around pt-6 gap-y-4">
        <div className="drop-shadow-sm w-full md:basis-2/3 lg:basis-1/3">
          <ul className="border border-collapse divide-y bg-slate-50 rounded">
            <li className="p-4">
              <div className="flex w-full justify-between items-center">
                <div className="text-xl font-light">Batch {batch.id}</div>
                <span className="flex items-center">
                  <BatchStateIcon state={batch.state} nJobs={batch.n_jobs} nCompleted={batch.n_completed} />
                </span>
              </div>
              {!batch.complete && (
                <div className="mt-2">
                  <AutoRefreshBar
                    autoRefresh={autoRefresh}
                    onToggle={setAutoRefresh}
                    countdownKey={countdownKey}
                    refreshing={batchRefreshing}
                    intervalMs={refreshIntervalMs}
                  />
                </div>
              )}
              <div className="flex justify-between items-center mt-2">
                <div>
                  <div className="font-light text-zinc-500">Submitted by {batch.user}</div>
                  <div className="font-light text-zinc-500 flex items-center gap-1">
                    Billed to{' '}
                    {billingProjectInfo ? (
                      <BillingProjectBadge
                        billingProject={batch.billing_project}
                        accruedCost={billingProjectInfo.accrued_cost}
                        limit={billingProjectInfo.limit}
                      />
                    ) : (
                      batch.billing_project
                    )}
                  </div>
                </div>
                {!batch.complete && batch.state !== 'cancelled' && (
                  <button
                    type="button"
                    onClick={handleCancel}
                    className="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700"
                  >
                    Cancel
                  </button>
                )}
                {batch.complete && (
                  <button
                    type="button"
                    onClick={handleDelete}
                    className="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700"
                  >
                    Delete
                  </button>
                )}
              </div>
              {actionError && <div className="mt-2 text-sm text-red-600">{actionError}</div>}
            </li>

            <CollapsibleItem title="Jobs" summary={batch.n_jobs} startOpen>
              <JobStatusTable batch={batch} onSearch={setSearch} />
            </CollapsibleItem>

            {batch.attributes && Object.keys(batch.attributes).length > 0 && (
              <CollapsibleItem title="Attributes">
                <table className="text-xs w-full">
                  <tbody className="divide-y">
                    {Object.entries(batch.attributes).map(([k, v]) => (
                      <tr key={k}>
                        <td className="py-1 pr-2 text-zinc-500">{k}</td>
                        <td className="py-1 text-right">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CollapsibleItem>
            )}

            <CollapsibleItem title="Duration" summary={batch.duration ?? ''}>
              <table className="text-xs w-full">
                <tbody className="divide-y">
                  <tr><td className="py-1 pr-2 text-zinc-500">Created</td><td className="py-1 text-right">{formatIsoTime(batch.time_created)}</td></tr>
                  <tr><td className="py-1 pr-2 text-zinc-500">Completed</td><td className="py-1 text-right">{formatIsoTime(batch.time_completed)}</td></tr>
                </tbody>
              </table>
            </CollapsibleItem>

            <CollapsibleItem title="Cost" summary={<CostDisplay cost={batch.cost} />}>
              {batch.cost_breakdown && (
                <table className="text-xs w-full">
                  <tbody className="divide-y">
                    {[...batch.cost_breakdown].sort((a, b) => a.resource.localeCompare(b.resource)).map(({ resource, cost }) => (
                      <tr key={resource}>
                        <td className="py-1 pr-2 text-zinc-500">{resource}</td>
                        <td className="py-1 text-right"><CostDisplay cost={cost} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </CollapsibleItem>
          </ul>
        </div>

        <div className="flex flex-col w-full lg:basis-3/5">
          <QueryBuilder fields={jobQueryFields} q={q} onSearch={setSearch} onDirtyChange={setQueryDirty} />

          {error && jobs == null ? (
            <div className="mt-4 text-red-600">Error loading jobs: {error}</div>
          ) : (
            <div className={`relative flex flex-col mt-4 transition-opacity ${jobsLoading || queryDirty ? 'opacity-50' : ''}`}>
              {jobsLoading && (
                <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
                  <SpinnerIcon className="h-8 w-8 text-sky-600" />
                </div>
              )}
              <table className="table-auto w-full" id="batch">
                <thead>
                  <tr>
                    <th className="h-12 bg-slate-200 font-light text-md text-left px-4 rounded-tl">ID</th>
                    <th className="h-12 bg-slate-200 font-light text-md text-left rounded-tr md:rounded-tr-none">Name</th>
                    <th className="h-12 bg-slate-200 font-light text-md text-left hidden lg:table-cell">Exit Code</th>
                    <th className="h-12 bg-slate-200 font-light text-md text-left hidden lg:table-cell">Duration</th>
                    <th className="h-12 bg-slate-200 font-light text-md text-left hidden md:table-cell rounded-tr">Cost</th>
                  </tr>
                </thead>
                <tbody className="border border-collapse border-slate-50">
                  {(jobs ?? []).map((job) => <JobRow key={job.job_id} basePath={basePath} job={job} />)}
                </tbody>
              </table>
            </div>
          )}

          <div className="pt-2 flex w-full justify-end gap-2">
            {hasPreviousPage && (
              <button
                type="button"
                onClick={goToPreviousPage}
                disabled={jobsLoading}
                className="px-3 py-1 bg-slate-200 rounded text-sm hover:bg-slate-300 disabled:opacity-50"
              >
                Previous page
              </button>
            )}
            {lastJobId != null && (
              <button
                type="button"
                onClick={goToNextPage}
                disabled={jobsLoading}
                className="px-3 py-1 bg-slate-200 rounded text-sm hover:bg-slate-300 disabled:opacity-50"
              >
                Next page
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
