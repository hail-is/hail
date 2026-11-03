import { useEffect, useMemo, useState } from 'react';
import { createCiApi } from '../../shared/ciApi';
import type { CIEvent } from '../../shared/ciApi';
import { formatTimestamp } from '../../shared/timeUtils';
import { collapseRepeats, eventDetails, isFailureEvent } from './eventLogModel';

const PAGE_SIZE = 50;

interface EventLogProps {
  basePath: string;
  batchBaseUrl: string;
  // Filters passed straight to /api/v1alpha/events. A column the filter pins is hidden.
  targetBranch?: string;
  pr?: number;
  event?: string;
  // Changing this refetches the first page, so the host page decides when to poll.
  refreshKey?: number;
  prHref?: (pr: number) => string;
}

function MessageCell({ message }: { message: string | null }): JSX.Element | null {
  const [expanded, setExpanded] = useState(false);
  if (!message) return null;
  const firstLine = message.split('\n', 1)[0];
  if (firstLine === message) return <span className="break-words">{message}</span>;
  return expanded ? (
    <div>
      <button type="button" onClick={() => { setExpanded(false); }} className="text-sky-600 hover:underline">
        hide
      </button>
      <pre className="mt-1 max-h-96 overflow-auto whitespace-pre-wrap text-xs bg-zinc-50 p-2 rounded">{message}</pre>
    </div>
  ) : (
    <span className="break-words">
      {firstLine}{' '}
      <button type="button" onClick={() => { setExpanded(true); }} className="text-sky-600 hover:underline">
        more
      </button>
    </span>
  );
}

export function EventLog({
  basePath, batchBaseUrl, targetBranch, pr, event, refreshKey = 0, prHref,
}: EventLogProps): JSX.Element {
  const api = useMemo(() => createCiApi(basePath), [basePath]);
  const [rows, setRows] = useState<CIEvent[] | null>(null);
  const [cursor, setCursor] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api.getEvents({ targetBranch, pr, event, limit: PAGE_SIZE })
      .then((r) => {
        if (cancelled) return;
        setRows(r.rows);
        setCursor(r.cursor);
        setError(null);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => { cancelled = true; };
  }, [api, targetBranch, pr, event, refreshKey]);

  const loadMore = async () => {
    if (cursor === null) return;
    setLoadingMore(true);
    try {
      const r = await api.getEvents({ targetBranch, pr, event, cursor, limit: PAGE_SIZE });
      setRows((prev) => [...(prev ?? []), ...r.rows]);
      setCursor(r.cursor);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingMore(false);
    }
  };

  const groups = useMemo(() => collapseRepeats(rows ?? []), [rows]);

  if (error) return <p className="text-sm text-red-600">Error loading events: {error}</p>;
  if (rows === null) return <p className="text-sm text-zinc-500">Loading events…</p>;
  if (rows.length === 0) return <p className="text-sm text-zinc-500">No events.</p>;

  const showBranch = targetBranch === undefined;
  const showPr = pr === undefined;

  return (
    <div>
      <table className="w-full text-sm border border-zinc-200 rounded overflow-hidden">
        <thead>
          <tr className="bg-zinc-100 text-left text-xs uppercase text-zinc-500">
            <th className="px-3 py-0.5">time</th>
            <th className="px-3 py-0.5">event</th>
            {showBranch && <th className="px-3 py-0.5">branch</th>}
            {showPr && <th className="px-3 py-0.5">pr</th>}
            <th className="px-3 py-0.5">user</th>
            <th className="px-3 py-0.5">batch</th>
            <th className="px-3 py-0.5">details</th>
            <th className="px-3 py-0.5">message</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-100 align-top">
          {groups.map(({ latest: e, earliest, count }) => (
            <tr key={e.id}>
              <td className="px-3 py-0.5 whitespace-nowrap">
                {formatTimestamp(e.time)}
                {count > 1 && (
                  <div className="text-xs text-zinc-500">×{count}, first {formatTimestamp(earliest.time)}</div>
                )}
              </td>
              <td className={`px-3 py-0.5 whitespace-nowrap font-mono text-xs ${isFailureEvent(e.event) ? 'text-red-600' : 'text-zinc-700'}`}>
                {e.event}
              </td>
              {showBranch && <td className="px-3 py-0.5 whitespace-nowrap">{e.target_branch ?? ''}</td>}
              {showPr && (
                <td className="px-3 py-0.5">
                  {e.pr_number !== null && (prHref ? (
                    <a href={prHref(e.pr_number)} className="text-sky-600 hover:underline">#{e.pr_number}</a>
                  ) : `#${e.pr_number}`)}
                </td>
              )}
              <td className="px-3 py-0.5">{e.username ?? <span className="text-zinc-400">ci</span>}</td>
              <td className="px-3 py-0.5">
                {e.batch_id !== null && (
                  <a href={`${batchBaseUrl}/batches/${e.batch_id}`} className="text-sky-600 hover:underline">{e.batch_id}</a>
                )}
              </td>
              <td className="px-3 py-0.5 whitespace-nowrap font-mono text-xs text-zinc-600">
                {eventDetails(e).map((d) => <div key={d}>{d}</div>)}
              </td>
              <td className="px-3 py-0.5"><MessageCell message={e.message} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      {cursor !== null && (
        <button
          type="button"
          onClick={() => { void loadMore(); }}
          disabled={loadingMore}
          className="mt-2 text-sm text-sky-600 hover:underline disabled:text-zinc-400"
        >
          {loadingMore ? 'Loading…' : 'Load more'}
        </button>
      )}
    </div>
  );
}
