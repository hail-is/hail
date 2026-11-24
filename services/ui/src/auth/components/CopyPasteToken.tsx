import { useEffect, useState } from 'react';
import { CopyableValue } from './CopyableValue';

const COPY_PASTE_TOKEN_TTL_SECONDS = 300;

function csrfToken(): string {
  return document.head.querySelector('meta[name="csrf"]')?.getAttribute('value') ?? '';
}

function formatCountdown(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

export function CopyPasteToken({ authBaseUrl }: { authBaseUrl: string }): JSX.Element {
  const [token, setToken] = useState<string | null>(null);
  const [expiresAt, setExpiresAt] = useState<number | null>(null);
  const [remainingSeconds, setRemainingSeconds] = useState(0);
  const [fetchingToken, setFetchingToken] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    if (expiresAt === null) return;
    const tick = () => { setRemainingSeconds(Math.max(0, Math.round((expiresAt - Date.now()) / 1000))); };
    tick();
    const interval = setInterval(tick, 1000);
    return () => { clearInterval(interval); };
  }, [expiresAt]);

  async function fetchToken() {
    setFetchingToken(true);
    setFetchError(null);
    try {
      const response = await fetch(`${authBaseUrl}/api/v1alpha/copy-paste-token`, {
        method: 'POST',
        headers: { 'X-CSRF-Token': csrfToken() },
      });
      if (!response.ok) {
        setToken(null);
        setExpiresAt(null);
        setFetchError(`HTTP ${response.status}`);
        return;
      }
      const newToken = await response.text();
      setToken(newToken);
      setExpiresAt(Date.now() + COPY_PASTE_TOKEN_TTL_SECONDS * 1000);
    } catch {
      setToken(null);
      setExpiresAt(null);
      setFetchError('network error');
    } finally {
      setFetchingToken(false);
    }
  }

  const tokenExpired = token !== null && expiresAt !== null && remainingSeconds <= 0;

  if (fetchingToken) {
    return <span className="text-sm text-zinc-500">Requesting…</span>;
  }

  if (token && !tokenExpired) {
    return (
      <span className="inline-flex items-center gap-2">
        <CopyableValue value={token} />
        <span className="text-xs text-zinc-400">(expires in {formatCountdown(remainingSeconds)})</span>
        <button type="button" onClick={() => { void fetchToken(); }} className="text-sm text-sky-600 hover:underline">
          Request another
        </button>
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-2">
      {tokenExpired && <span className="text-sm text-zinc-500">Expired ·</span>}
      {fetchError !== null && <span className="text-sm text-red-600">Request failed ({fetchError}) ·</span>}
      <button type="button" onClick={() => { void fetchToken(); }} className="text-sm text-sky-600 hover:underline">
        {fetchError !== null ? 'Retry' : 'Request'}
      </button>
    </span>
  );
}
