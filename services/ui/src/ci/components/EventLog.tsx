import { useEffect, useMemo, useState } from 'react';
import { createCiApi } from '../../shared/ciApi';
import type { CIEvent } from '../../shared/ciApi';
import { DataTr, EmptyCell, Td, Th } from '../../shared/Table';
import { formatTimestamp } from '../../shared/timeUtils';
import { collapseRepeats, eventDetails } from './eventLogModel';

const DEFAULT_PAGE_SIZE = 50;

interface EventLogProps {
  basePath: string;
  batchBaseUrl: string;
  // Filters passed straight to /api/v1alpha/events. A column the filter pins is hidden.
  targetBranch?: string;
  pr?: number;
  event?: string;
  errorsOnly?: boolean;
  includePrEvents?: boolean;
  // Only events from this long before each fetch.
  sinceWindowMs?: number;
  hideBranch?: boolean;
  // Changing this refetches the first page, so the host page decides when to poll.
  refreshKey?: number;
  pageSize?: number;
  // Link for an event's PR number; null leaves it as plain text.
  prHref?: (e: CIEvent) => string | null;
}

export function MessageCell({ message }: { message: string | null }): JSX.Element {
  const [expanded, setExpanded] = useState(false);
  if (!message) return <EmptyCell />;
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

function PrCell({ event, prHref }: { event: CIEvent; prHref?: (e: CIEvent) => string | null }): JSX.Element {
  const href = prHref?.(event) ?? null;
  const label = `#${event.pr_number}`;
  return href === null ? <>{label}</> : <a href={href} className="text-sky-600 hover:underline">{label}</a>;
}

export function EventLog({
  basePath, batchBaseUrl, targetBranch, pr, event, errorsOnly, includePrEvents, sinceWindowMs, hideBranch = false,
  refreshKey = 0, pageSize = DEFAULT_PAGE_SIZE, prHref,
}: EventLogProps): JSX.Element {
  const api = useMemo(() => createCiApi(basePath), [basePath]);
  const [rows, setRows] = useState<CIEvent[] | null>(null);
  const [cursor, setCursor] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  // Fixed at the first page's fetch so "Load more" pages use the same window.
  const [sinceMs, setSinceMs] = useState<number | undefined>(undefined);

  useEffect(() => {
    let cancelled = false;
    const since = sinceWindowMs === undefined ? undefined : Date.now() - sinceWindowMs;
    api.getEvents({ targetBranch, pr, event, errorsOnly, includePrEvents, sinceMs: since, limit: pageSize })
      .then((r) => {
        if (cancelled) return;
        setSinceMs(since);
        setRows(r.rows);
        setCursor(r.cursor);
        setError(null);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => { cancelled = true; };
  }, [api, targetBranch, pr, event, errorsOnly, includePrEvents, sinceWindowMs, refreshKey, pageSize]);

  const loadMore = async () => {
    if (cursor === null) return;
    setLoadingMore(true);
    try {
      const r = await api.getEvents({ targetBranch, pr, event, errorsOnly, includePrEvents, sinceMs, cursor, limit: pageSize });
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

  const showBranch = targetBranch === undefined && !hideBranch;
  const showPr = pr === undefined;

  return (
    <div>
      {/* Scrolls inside a fixed height so a long history doesn't push the rest of the page down */}
      <div className="max-h-[32rem] overflow-auto rounded border border-zinc-200">
        <table className="w-full text-sm">
          <thead className="sticky top-0 z-10">
            <tr>
              <Th>Time</Th>
              <Th>Event</Th>
              {showBranch && <Th>Branch</Th>}
              {showPr && <Th>PR</Th>}
              <Th>User</Th>
              <Th>Batch</Th>
              <Th>Details</Th>
              <Th>Message</Th>
            </tr>
          </thead>
          <tbody className="align-top">
            {groups.map(({ latest: e, earliest, count }) => {
              const details = eventDetails(e);
              return (
                <DataTr key={e.id}>
                  <Td className="whitespace-nowrap">
                    {formatTimestamp(e.time)}
                    {count > 1 && (
                      <div className="text-xs text-zinc-500">×{count}, first {formatTimestamp(earliest.time)}</div>
                    )}
                  </Td>
                  <Td className={`whitespace-nowrap font-mono text-xs ${e.is_error ? 'text-red-600' : 'text-zinc-700'}`}>
                    {e.event}
                  </Td>
                  {showBranch && <Td className="whitespace-nowrap">{e.target_branch ?? <EmptyCell />}</Td>}
                  {showPr && (
                    <Td>
                      {e.pr_number === null ? <EmptyCell /> : <PrCell event={e} prHref={prHref} />}
                    </Td>
                  )}
                  <Td>{e.username ?? <span className="text-zinc-400">ci</span>}</Td>
                  <Td>
                    {e.batch_id === null
                      ? <EmptyCell />
                      : <a href={`${batchBaseUrl}/batches/${e.batch_id}`} className="text-sky-600 hover:underline">{e.batch_id}</a>}
                  </Td>
                  <Td className="whitespace-nowrap font-mono text-xs text-zinc-600">
                    {details.length === 0 ? <EmptyCell /> : details.map((d) => <div key={d}>{d}</div>)}
                  </Td>
                  <Td><MessageCell message={e.message} /></Td>
                </DataTr>
              );
            })}
          </tbody>
        </table>
      </div>
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
