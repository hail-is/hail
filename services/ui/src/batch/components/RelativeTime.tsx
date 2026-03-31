import { useState, useEffect } from 'react';

function formatRelative(ms: number): string {
  const diff = Date.now() - ms;
  const abs = Math.abs(diff);
  if (abs < 60_000) return `${Math.round(abs / 1000)}s ago`;
  if (abs < 3_600_000) return `${Math.round(abs / 60_000)}m ago`;
  if (abs < 86_400_000) return `${Math.round(abs / 3_600_000)}h ago`;
  return `${Math.round(abs / 86_400_000)}d ago`;
}

function formatAbsolute(ms: number): string {
  return new Date(ms).toLocaleString();
}

type Props = { ms: number };

export function RelativeTime({ ms }: Props): JSX.Element {
  const [, setTick] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 30_000);
    return () => clearInterval(id);
  }, []);

  return (
    <span title={formatAbsolute(ms)} className="cursor-help">
      {formatRelative(ms)}
    </span>
  );
}
