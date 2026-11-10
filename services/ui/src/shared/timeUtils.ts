import prettyMilliseconds from 'pretty-ms';

// Time since a timestamp in its largest unit, e.g. "5s ago", "3m ago", "4d ago".
export function formatRelative(ms: number | null): string {
  if (ms == null) return '';
  return `${prettyMilliseconds(Math.abs(Date.now() - ms), { compact: true })} ago`;
}

// Compact duration, e.g. "45s", "2m 5s", "1h 30m".
export function formatDuration(ms: number | null | undefined): string {
  if (ms == null || ms === 0) return '0';
  return prettyMilliseconds(ms, { unitCount: 2, secondsDecimalDigits: 0 });
}

// Full local date and time with timezone. Accepts epoch ms or an ISO 8601 string (as the batch API sends).
export function formatTimestamp(t: number | string | null | undefined): string {
  if (t == null) return '';
  return new Date(t).toLocaleString(undefined, {
    year: 'numeric', month: 'numeric', day: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    timeZoneName: 'short',
  });
}

// Axis tick label: local HH:MM:SS, with the date on a second line above the first tick.
export function formatClockTick(d: Date | number, i: number): string {
  const date = new Date(d);
  const time = date.toLocaleTimeString(undefined, {
    hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23',
  });
  if (i !== 0) return time;
  // Short month name keeps the date unambiguous across locales
  return `${date.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })}\n${time}`;
}
