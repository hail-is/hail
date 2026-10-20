export function formatRelative(ms: number): string {
  const abs = Math.abs(Date.now() - ms);
  if (abs < 60_000) return `${Math.round(abs / 1000)}s ago`;
  if (abs < 3_600_000) return `${Math.round(abs / 60_000)}m ago`;
  if (abs < 86_400_000) return `${Math.round(abs / 3_600_000)}h ago`;
  return `${Math.round(abs / 86_400_000)}d ago`;
}

export function formatDuration(ms: number | null): string {
  if (ms == null) return '';
  const diffSec = Math.floor((Date.now() - ms) / 1000);
  if (diffSec < 60) return `${diffSec}s`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m`;
  const diffHr = Math.floor(diffMin / 60);
  return `${diffHr}h ${diffMin % 60}m`;
}

export function formatTime(ms: number | null): string {
  if (ms == null) return '';
  return new Date(ms).toLocaleString(undefined, {
    year: 'numeric', month: 'numeric', day: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    timeZoneName: 'short',
  });
}

// The batch API sends timestamps as ISO 8601 UTC strings (e.g. "2026-09-18T12:34:56.789Z").
export function formatIsoTime(iso: string | null | undefined): string {
  if (iso == null) return '';
  return formatTime(Date.parse(iso));
}

// Local wall-clock time as HH:MM:SS.
export function formatClockTime(d: Date | number): string {
  const date = new Date(d);
  const hh = String(date.getHours()).padStart(2, '0');
  const mm = String(date.getMinutes()).padStart(2, '0');
  const ss = String(date.getSeconds()).padStart(2, '0');
  return `${hh}:${mm}:${ss}`;
}

// Axis tick label: HH:MM:SS, with the date on a second line above the first tick.
export function formatClockTick(d: Date | number, i: number): string {
  const time = formatClockTime(d);
  if (i === 0) {
    // Short month name keeps the date unambiguous across locales
    const date = new Date(d).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });
    return `${date}\n${time}`;
  }
  return time;
}

// Compact elapsed-time label for chart axes, e.g. "45s", "2m 5s", "1h 30m".
export function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds * 10) / 10}s`;
  const totalSec = Math.round(seconds);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  if (h > 0) return m > 0 ? `${h}h ${m}m` : `${h}h`;
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}
