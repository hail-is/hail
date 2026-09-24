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
