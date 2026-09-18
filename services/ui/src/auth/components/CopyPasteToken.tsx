import { useEffect, useState } from 'react';
import { CodeBlock } from '../../batch/components/CodeBlock';

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

  useEffect(() => {
    if (expiresAt === null) return;
    const tick = () => { setRemainingSeconds(Math.max(0, Math.round((expiresAt - Date.now()) / 1000))); };
    tick();
    const interval = setInterval(tick, 1000);
    return () => { clearInterval(interval); };
  }, [expiresAt]);

  async function fetchToken() {
    setFetchingToken(true);
    try {
      const response = await fetch(`${authBaseUrl}/api/v1alpha/copy-paste-token`, {
        method: 'POST',
        headers: { 'X-CSRF-Token': csrfToken() },
      });
      const newToken = await response.text();
      setToken(newToken);
      setExpiresAt(Date.now() + COPY_PASTE_TOKEN_TTL_SECONDS * 1000);
    } finally {
      setFetchingToken(false);
    }
  }

  const tokenExpired = token !== null && expiresAt !== null && remainingSeconds <= 0;

  if (token && !tokenExpired) {
    return (
      <div className="space-y-3">
        <p className="text-sm text-zinc-600">
          This copy-paste token is good for one use, and expires in{' '}
          <span className="font-mono">{formatCountdown(remainingSeconds)}</span>.
        </p>
        <CodeBlock code={token} />
        <p className="text-sm text-zinc-600">
          If you need to authenticate a Jupyter Notebook session (for example, a Terra Jupyter Notebook),
          copy and paste this into your notebook:
        </p>
        <CodeBlock code={`from hailtop.auth import copy_paste_login; copy_paste_login('${token}')`} />
        <p className="text-sm text-zinc-600">
          If you need to authenticate from a terminal, copy and paste this into your terminal:
        </p>
        <CodeBlock code={`hailctl auth copy-paste-login "${token}"`} />
        <button
          onClick={() => { void fetchToken(); }}
          disabled={fetchingToken}
          className="text-sm text-sky-600 hover:underline disabled:text-zinc-400"
        >
          Create another token
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {tokenExpired && <p className="text-sm text-zinc-600">That token has expired.</p>}
      <button
        onClick={() => { void fetchToken(); }}
        disabled={fetchingToken}
        className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:bg-zinc-400"
      >
        {fetchingToken ? 'Requesting…' : 'Get a copy-paste login token'}
      </button>
    </div>
  );
}
