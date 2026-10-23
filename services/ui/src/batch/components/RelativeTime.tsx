import { useState, useEffect } from 'react';
import { formatRelative, formatTime } from '../../shared/timeUtils';
import { HoverTip } from './HoverTip';

type Props = { ms: number };

export function RelativeTime({ ms }: Props): JSX.Element {
  const [, setTick] = useState(0);

  useEffect(() => {
    const id = setInterval(() => { setTick((t) => t + 1); }, 30_000);
    return () => { clearInterval(id); };
  }, []);

  return (
    <HoverTip tip={formatTime(ms)}>
      <span className="cursor-help">{formatRelative(ms)}</span>
    </HoverTip>
  );
}
