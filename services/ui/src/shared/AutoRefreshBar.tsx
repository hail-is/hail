import { useState, useEffect } from 'react';

function CountdownFill({ intervalMs }: { intervalMs: number }): JSX.Element {
  const [go, setGo] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setGo(true));
    return () => cancelAnimationFrame(id);
  }, []);
  return (
    <div
      className="h-full bg-sky-400"
      style={{
        width: go ? '100%' : '0%',
        transition: go ? `width ${intervalMs}ms linear` : 'none',
      }}
    />
  );
}

interface Props {
  autoRefresh: boolean;
  onToggle: (_v: boolean) => void;
  countdownKey: number;
  refreshing: boolean;
  intervalMs: number;
}

export function AutoRefreshBar({ autoRefresh, onToggle, countdownKey, refreshing, intervalMs }: Props): JSX.Element {
  return (
    <div>
      <label className="flex items-center gap-1.5 text-zinc-500 text-sm cursor-pointer select-none">
        <input
          type="checkbox"
          checked={autoRefresh}
          onChange={(e) => { onToggle(e.target.checked); }}
          className="cursor-pointer"
        />
        Auto-refresh
        {autoRefresh && refreshing && (
          <span className="material-symbols-outlined text-sm animate-spin text-sky-400" style={{ animationDuration: '1s' }}>
            progress_activity
          </span>
        )}
      </label>
      <div className="mt-1.5 h-0.5 w-64 bg-zinc-300 rounded-full overflow-hidden">
        {autoRefresh && !refreshing && <CountdownFill key={countdownKey} intervalMs={intervalMs} />}
        {autoRefresh && refreshing && <div className="h-full bg-sky-300 w-full animate-pulse" />}
      </div>
    </div>
  );
}
