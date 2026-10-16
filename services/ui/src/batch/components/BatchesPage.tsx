import { useState, useEffect } from 'react';
import type { Batch } from '../../shared/batchApi';
import { costByCategory, formatCpuHours, formatDurationMs, hasFieldTerm, jobCountsSummary, withFieldTerm } from './batchModels';
import { BatchStateIcon } from './StateIcon';
import { CostDisplay } from './CostDisplay';
import { RelativeTime } from './RelativeTime';
import { AutoRefreshBar } from '../../shared/AutoRefreshBar';
import { SegmentedBar } from '../../shared/SegmentedBar';
import { SpinnerIcon } from '../../shared/SpinnerIcon';
import { QueryBuilder } from '../../shared/QueryBuilder';
import { batchQueryFields } from '../../shared/queryFields';
import { useBatchList } from '../hooks/useBatchList';
import { disableReactUi } from '../../shared/reactUiCookie';

interface Props {
  basePath: string;
  username: string;
}

function useNow(intervalMs: number): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => { setNow(Date.now()); }, intervalMs);
    return () => { clearInterval(id); };
  }, [intervalMs]);
  return now;
}

function parseIso(iso: string | null | undefined): number | null {
  if (iso == null) return null;
  const ms = Date.parse(iso);
  return Number.isNaN(ms) ? null : ms;
}

function stateLabel(batch: Batch): { label: string; color: string } {
  const finished = batch.n_jobs === batch.n_completed;
  switch (batch.state) {
    case 'open': return { label: 'Open', color: 'text-zinc-500' };
    case 'success': return { label: 'Success', color: 'text-green-700' };
    case 'failure': return { label: batch.complete ? 'Failed' : 'Failing', color: 'text-red-700' };
    case 'cancelled': return { label: finished ? 'Cancelled' : 'Cancelling', color: 'text-yellow-700' };
    default: return { label: 'Running', color: 'text-sky-700' };
  }
}

function BatchStatus({ batch }: { batch: Batch }): JSX.Element {
  const { label, color } = stateLabel(batch);
  return (
    <span className={`inline-flex items-center gap-1 text-sm ${color}`}>
      <span className="inline-flex w-4 justify-center">
        <BatchStateIcon state={batch.state} nJobs={batch.n_jobs} nCompleted={batch.n_completed} />
      </span>
      {label}
    </span>
  );
}

function JobCounts({ batch }: { batch: Batch }): JSX.Element {
  return (
    <div className="pr-4" title={jobCountsSummary(batch)}>
      <SegmentedBar
        className="h-1.5 w-full"
        total={batch.n_jobs}
        segments={[
          [batch.n_succeeded, 'bg-green-500', 'succeeded'],
          [batch.n_failed, 'bg-red-500', 'failed'],
          [batch.n_cancelled, 'bg-yellow-400', 'cancelled'],
        ]}
      />
      <div className="mt-0.5 flex justify-between text-xs text-zinc-500 tabular-nums">
        <span>{batch.n_completed.toLocaleString()} / {batch.n_jobs.toLocaleString()}</span>
        {batch.n_failed > 0 && <span className="text-red-600">{batch.n_failed.toLocaleString()} failed</span>}
      </div>
    </div>
  );
}

// A cell value that narrows the query to itself when clicked (or plain text, once it already has).
function TermFilterButton({ field, value, q, onSearch }: {
  field: string;
  value: string;
  q: string;
  onSearch: (_q: string) => void;
}): JSX.Element {
  if (hasFieldTerm(q, field, value)) {
    return <span className="font-light text-zinc-800 truncate block" title={value}>{value}</span>;
  }
  return (
    <button
      type="button"
      onClick={() => { onSearch(withFieldTerm(q, field, value)); }}
      title={`Filter to ${field} = ${value}`}
      className="group inline-flex max-w-full items-center gap-0.5 font-light text-zinc-500 hover:text-sky-600"
    >
      <span className="truncate">{value}</span>
      <span aria-hidden="true" className="material-symbols-outlined text-sm invisible group-hover:visible">filter_alt</span>
    </button>
  );
}

function QuickFilters({ q, username, onSearch }: {
  q: string;
  username: string;
  onSearch: (_q: string) => void;
}): JSX.Element {
  // Running and Failed both set the `state` term, so they're mutually exclusive; either combines
  // with Mine.
  const chips = [
    ...(username ? [{ label: 'Mine', field: 'user', value: username }] : []),
    { label: 'Running', field: 'state', value: 'running' },
    { label: 'Failed', field: 'state', value: 'failure' },
  ];
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="text-sm text-zinc-500 mr-1">Quick filters:</span>
      {chips.map(({ label, field, value }) => {
        const active = hasFieldTerm(q, field, value);
        return (
          <button
            type="button"
            key={label}
            aria-pressed={active}
            onClick={() => { onSearch(withFieldTerm(q, field, active ? null : value)); }}
            className={`text-xs px-2.5 py-0.5 rounded-full border transition-colors ${active ? 'bg-sky-500 border-sky-500 text-white' : 'border-zinc-300 text-zinc-500 hover:border-sky-400 hover:text-sky-600 bg-white'}`}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}

// Cancel/delete are irreversible, and a dense list makes misclicks easy, so each takes a second
// confirming click.
function RowActions({ batch, onCancel, onDelete }: {
  batch: Batch;
  onCancel: () => Promise<void>;
  onDelete: () => Promise<void>;
}): JSX.Element | null {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);

  const action = batch.complete
    ? { label: 'Delete', icon: 'delete', run: onDelete }
    : batch.state !== 'cancelled'
      ? { label: 'Cancel', icon: 'cancel', run: onCancel }
      : null;
  if (action == null) return null;

  if (busy) return <SpinnerIcon className="text-zinc-400" />;

  if (!confirming) {
    return (
      <button
        type="button"
        onClick={() => { setConfirming(true); }}
        title={`${action.label} batch ${batch.id}`}
        aria-label={`${action.label} batch ${batch.id}`}
        className="text-zinc-400 hover:text-red-600"
      >
        <span aria-hidden="true" className="material-symbols-outlined text-lg leading-none">{action.icon}</span>
      </button>
    );
  }

  return (
    <span className="inline-flex items-center gap-1">
      <button
        type="button"
        onClick={() => {
          setBusy(true);
          action.run().finally(() => { setBusy(false); setConfirming(false); });
        }}
        className="px-1.5 py-0.5 bg-red-600 text-white rounded text-xs hover:bg-red-700"
      >
        {action.label}?
      </button>
      <button
        type="button"
        onClick={() => { setConfirming(false); }}
        title="Never mind"
        aria-label="Never mind"
        className="text-zinc-400 hover:text-zinc-700"
      >
        <span aria-hidden="true" className="material-symbols-outlined text-sm leading-none">close</span>
      </button>
    </span>
  );
}

function CostCell({ batch }: { batch: Batch }): JSX.Element {
  const categories = batch.cost_breakdown ? costByCategory(batch.cost_breakdown).filter((c) => c.cost > 0) : [];
  return (
    <span className="relative group inline-block tabular-nums">
      <span className={categories.length > 0 ? 'underline decoration-dotted decoration-zinc-300 cursor-help' : ''}>
        <CostDisplay cost={batch.cost} />
      </span>
      {categories.length > 0 && (
        <span className="absolute right-0 top-full mt-1 hidden group-hover:block bg-zinc-800 text-white text-xs rounded px-2 py-1 z-10 pointer-events-none whitespace-nowrap text-left">
          <table>
            <tbody>
              {categories.map(({ resource, cost }) => (
                <tr key={resource}>
                  <td className="pr-3">{resource}</td>
                  <td className="text-right"><CostDisplay cost={cost} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </span>
      )}
    </span>
  );
}

function DurationCell({ batch, now }: { batch: Batch; now: number }): JSX.Element {
  if (batch.duration_ms != null) return <>{formatDurationMs(batch.duration_ms)}</>;
  const created = parseIso(batch.time_created);
  if (!batch.complete && created != null) {
    return <span className="text-sky-700" title="Elapsed so far">{formatDurationMs(now - created)}</span>;
  }
  return <span className="text-zinc-400">--</span>;
}

function TimeCell({ iso, fallback }: { iso: string | null | undefined; fallback: string }): JSX.Element {
  const ms = parseIso(iso);
  return ms != null ? <RelativeTime ms={ms} /> : <span className="text-zinc-400">{fallback}</span>;
}

function BatchRow({ basePath, batch, q, now, onSearch, onCancel, onDelete }: {
  basePath: string;
  batch: Batch;
  q: string;
  now: number;
  onSearch: (_q: string) => void;
  onCancel: () => Promise<void>;
  onDelete: () => Promise<void>;
}): JSX.Element {
  const href = `${basePath}/batches/${batch.id}`;
  const name = batch.attributes?.name;
  return (
    <tr className="border border-collapse hover:bg-slate-100">
      <td className="pl-3 py-2"><BatchStatus batch={batch} /></td>
      <td className="font-light">
        <a href={href} className="hover:text-sky-600 hover:underline underline-offset-2">{batch.id}</a>
      </td>
      <td className="pr-4">
        <a
          href={href}
          title={name}
          className={`block truncate hover:text-sky-600 hover:underline underline-offset-2 ${name ? '' : 'text-zinc-400 italic'}`}
        >
          {name ?? 'no name'}
        </a>
      </td>
      <td className="hidden lg:table-cell pr-4">
        <TermFilterButton field="user" value={batch.user} q={q} onSearch={onSearch} />
      </td>
      <td className="hidden lg:table-cell pr-4">
        <TermFilterButton field="billing_project" value={batch.billing_project} q={q} onSearch={onSearch} />
      </td>
      <td className="hidden lg:table-cell"><JobCounts batch={batch} /></td>
      <td className="hidden lg:table-cell font-light"><TimeCell iso={batch.time_created} fallback="" /></td>
      <td className="hidden 2xl:table-cell font-light">
        <TimeCell iso={batch.time_completed} fallback="--" />
      </td>
      <td className="hidden xl:table-cell font-light tabular-nums"><DurationCell batch={batch} now={now} /></td>
      <td className="hidden xl:table-cell font-light text-right pr-4 tabular-nums">{formatCpuHours(batch.msec_mcpu)}</td>
      <td className="hidden md:table-cell font-light text-right pr-4"><CostCell batch={batch} /></td>
      <td className="hidden md:table-cell text-center"><RowActions batch={batch} onCancel={onCancel} onDelete={onDelete} /></td>
    </tr>
  );
}

const PAGE_BUTTON = 'px-3 py-1 bg-slate-200 rounded text-sm hover:bg-slate-300 disabled:opacity-50';

export function BatchesPage({ basePath, username }: Props): JSX.Element {
  const {
    batches,
    error,
    loading,
    batchesLoading,
    refreshing,
    autoRefresh,
    setAutoRefresh,
    refreshActive,
    countdownKey,
    refreshIntervalMs,
    q,
    hasPreviousPage,
    isFirstPage,
    hasNextPage,
    setSearch,
    goToNextPage,
    goToPreviousPage,
    goToFirstPage,
    cancelBatch,
    deleteBatch,
  } = useBatchList(basePath, username);

  const [queryDirty, setQueryDirty] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const runAction = (run: () => Promise<void>) => async () => {
    setActionError(null);
    try {
      await run();
    } catch (e) {
      setActionError(String(e));
    }
  };
  const now = useNow(refreshIntervalMs);

  if (loading) {
    return (
      <div className="flex items-center justify-center mt-24">
        <span className="text-5xl font-light text-sky-600">Loading…</span>
      </div>
    );
  }

  const pageCost = (batches ?? []).reduce((sum, b) => sum + b.cost, 0);

  return (
    <div className="pb-8">
      {/* Breadcrumb */}
      <nav className="flex items-center gap-2 text-xl font-light text-zinc-800 flex-wrap">
        <span>Batches</span>
      </nav>
      <div className="mt-1 text-sm">
        <button
          type="button"
          onClick={disableReactUi}
          className="text-sky-600 hover:underline cursor-pointer"
        >
          Back to classic layout
        </button>
      </div>

      <div className="flex flex-col w-full mt-4 space-y-4">
        <QueryBuilder
          fields={batchQueryFields}
          q={q}
          onSearch={setSearch}
          onDirtyChange={setQueryDirty}
          emptyLabel="Showing all batches in your billing projects"
        />

        <div className="flex flex-wrap items-end justify-between gap-4">
          <QuickFilters q={q} username={username} onSearch={setSearch} />
          {refreshActive && (
            <div className="w-56">
              <AutoRefreshBar
                autoRefresh={autoRefresh}
                onToggle={setAutoRefresh}
                countdownKey={countdownKey}
                refreshing={refreshing}
                intervalMs={refreshIntervalMs}
              />
            </div>
          )}
        </div>

        {error && <div className="text-red-600">Error loading batches: {error}</div>}
        {actionError && <div className="text-red-600">{actionError}</div>}

        {batches != null && (
          <div className={`relative transition-opacity ${batchesLoading || queryDirty ? 'opacity-50' : ''}`}>
            {batchesLoading && (
              <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
                <SpinnerIcon className="h-8 w-8 text-sky-600" />
              </div>
            )}
            <table className="table-fixed w-full" id="batches">
              <thead>
                <tr>
                  <th className="h-12 bg-slate-200 font-light text-md text-left pl-3 rounded-tl w-28">Status</th>
                  <th className="h-12 bg-slate-200 font-light text-md text-left w-16">ID</th>
                  <th className="h-12 bg-slate-200 font-light text-md text-left rounded-tr md:rounded-tr-none">Batch Name</th>
                  <th className="h-12 bg-slate-200 font-light text-md text-left hidden lg:table-cell w-24">User</th>
                  <th className="h-12 bg-slate-200 font-light text-md text-left hidden lg:table-cell w-32">Billing Project</th>
                  <th className="h-12 bg-slate-200 font-light text-md text-left hidden lg:table-cell w-32">Jobs</th>
                  <th className="h-12 bg-slate-200 font-light text-md text-left hidden lg:table-cell w-24">Created</th>
                  <th className="h-12 bg-slate-200 font-light text-md text-left hidden 2xl:table-cell w-24">Completed</th>
                  <th className="h-12 bg-slate-200 font-light text-md text-left hidden xl:table-cell w-20">Duration</th>
                  <th className="h-12 bg-slate-200 font-light text-md text-right pr-4 hidden xl:table-cell w-24" title="Total CPU-hours reserved by this batch's jobs">CPU hrs</th>
                  <th className="h-12 bg-slate-200 font-light text-md text-right pr-4 hidden md:table-cell w-24">Cost</th>
                  <th className="h-12 bg-slate-200 font-light text-md hidden md:table-cell rounded-tr w-24"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody className="border border-collapse border-slate-50">
                {batches.map((batch) => (
                  <BatchRow
                    key={batch.id}
                    basePath={basePath}
                    batch={batch}
                    q={q}
                    now={now}
                    onSearch={setSearch}
                    onCancel={runAction(() => cancelBatch(batch.id))}
                    onDelete={runAction(() => deleteBatch(batch.id))}
                  />
                ))}
                {batches.length === 0 && (
                  <tr>
                    <td colSpan={12} className="py-8 text-center text-zinc-400 italic">No batches match this query</td>
                  </tr>
                )}
              </tbody>
              {batches.length > 0 && (
                <tfoot className="hidden md:table-footer-group">
                  <tr>
                    <td colSpan={12} className="pt-2 pr-4 text-right text-sm text-zinc-500 tabular-nums">
                      Total for this page: <CostDisplay cost={pageCost} />
                    </td>
                  </tr>
                </tfoot>
              )}
            </table>
          </div>
        )}

        <div className="flex w-full justify-end gap-2">
          {!isFirstPage && !hasPreviousPage && (
            <button type="button" onClick={goToFirstPage} disabled={batchesLoading} className={PAGE_BUTTON}>
              First page
            </button>
          )}
          {hasPreviousPage && (
            <button type="button" onClick={goToPreviousPage} disabled={batchesLoading} className={PAGE_BUTTON}>
              Previous page
            </button>
          )}
          {hasNextPage && (
            <button type="button" onClick={goToNextPage} disabled={batchesLoading} className={PAGE_BUTTON}>
              Next page
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
