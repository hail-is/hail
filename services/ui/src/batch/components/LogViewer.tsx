import { useState, useMemo, useEffect, useRef } from 'react';

type Props = {
  text: string;
  downloadUrl: string;
  downloadName: string;
  hasPendingUpdate?: boolean;
  onLoadUpdate?: () => void;
  isRefreshing?: boolean;
};

export function LogViewer({ text, downloadUrl, downloadName, hasPendingUpdate, onLoadUpdate, isRefreshing }: Props): JSX.Element {
  const [query, setQuery] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [text]);

  const lines = useMemo(() => text.split('\n'), [text]);

  const filteredLines = useMemo(
    () => (query.trim() ? lines.filter((l) => l.includes(query)) : lines),
    [lines, query]
  );

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 flex-wrap">
        <input
          type="text"
          placeholder="Filter lines…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="border rounded px-2 py-1 text-sm flex-1 max-w-sm"
        />
        <a
          href={downloadUrl}
          download={downloadName}
          className="text-sm text-sky-600 hover:underline flex items-center gap-1"
        >
          Download
        </a>
        {hasPendingUpdate && onLoadUpdate ? (
          <span className="text-sm text-sky-700 bg-sky-50 border border-sky-200 rounded px-2 py-1">
            New log data available -{' '}
            <button onClick={onLoadUpdate} className="font-medium underline hover:no-underline">
              show
            </button>
          </span>
        ) : isRefreshing ? (
          <span className="text-sm text-zinc-400 bg-zinc-50 border border-zinc-200 rounded px-2 py-1 flex items-center gap-1.5">
            <span className="material-symbols-outlined text-sm animate-spin" style={{ animationDuration: '1s' }}>progress_activity</span>
            Checking for updates…
          </span>
        ) : null}
      </div>
      <div ref={scrollRef} className="bg-slate-50 border rounded overflow-auto" style={{ maxHeight: '32rem' }}>
        <pre className="text-xs p-2 whitespace-pre-wrap break-all">
          {filteredLines.join('\n')}
        </pre>
      </div>
      {query && (
        <div className="text-xs text-zinc-400">
          {filteredLines.length} / {lines.length} lines
        </div>
      )}
    </div>
  );
}
