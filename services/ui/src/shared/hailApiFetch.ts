// Fetches JSON from a Hail service API endpoint, attaching the CSRF token (read from the
// <meta name="csrf"> tag every rendered page carries) for any request that isn't GET/HEAD —
// the server only enforces it for mutating methods, but it's harmless to have this check the
// method rather than force every caller to know that.
export async function hailApiFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const method = (init?.method ?? 'GET').toUpperCase();
  const headers = new Headers(init?.headers);
  if (method !== 'GET' && method !== 'HEAD') {
    const token = document.head.querySelector('meta[name="csrf"]')?.getAttribute('value');
    if (token) headers.set('X-CSRF-Token', token);
  }
  // 'include' (not 'same-origin'): dashboards call sibling hail services (e.g. monitoring calling
  // batch) which are cross-origin but same-site, so the shared session cookie (domain=hail.is) is
  // only attached if we explicitly ask for it.
  const resp = await fetch(url, { credentials: 'include', ...init, headers });
  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(`${method} ${url} failed (HTTP ${resp.status})${text ? `: ${text}` : ''}`);
  }
  return await resp.json() as T;
}
