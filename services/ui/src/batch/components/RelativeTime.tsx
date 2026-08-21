import { useState, useEffect } from 'react';
import { formatRelative } from '../../shared/timeUtils';

function formatAbsolute(ms: number): string {
  return new Date(ms).toLocaleString();
}

type Props = { ms: number };

export function RelativeTime({ ms }: Props): JSX.Element {
  const [, setTick] = useState(0);

  useEffect(() => {
    const id = setInterval(() => { setTick((t) => t + 1); }, 30_000);
    return () => { clearInterval(id); };
  }, []);

  return (
    <span title={formatAbsolute(ms)} className="cursor-help">
      {formatRelative(ms)}
    </span>
  );
}
