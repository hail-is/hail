// Sends a request to a Hail service API endpoint, attaching the CSRF token (read from the
// <meta name="csrf"> tag every rendered page carries) for any request that isn't GET/HEAD —
// the server only enforces it for mutating methods, but it's harmless to have this check the
// method rather than force every caller to know that. Throws on a non-OK response.
async function hailFetch(url: string, init?: RequestInit): Promise<Response> {
  const method = (init?.method ?? 'GET').toUpperCase();
  const headers = new Headers(init?.headers);
  if (method !== 'GET' && method !== 'HEAD') {
    const token = document.head.querySelector('meta[name="csrf"]')?.getAttribute('value');
    if (token) headers.set('X-CSRF-Token', token);
  }
  const resp = await fetch(url, { credentials: 'same-origin', ...init, headers });
  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(`${method} ${url} failed (HTTP ${resp.status})${text ? `: ${text}` : ''}`);
  }
  return resp;
}

// Fetches and parses a JSON response body of shape T. Use this when the caller needs the
// response; if the endpoint may reply with an empty body, use hailApiFetchVoid instead — this
// throws a clear error rather than a cryptic "Unexpected end of JSON input" if the body is empty.
export async function hailApiFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await hailFetch(url, init);
  const text = await resp.text();
  if (text.length === 0) {
    const method = (init?.method ?? 'GET').toUpperCase();
    throw new Error(`${method} ${url}: expected a JSON response body but received an empty one`);
  }
  return JSON.parse(text) as T;
}

// Sends a request whose response body (if any) the caller doesn't need. Never attempts to parse
// the body as JSON, so it's safe against endpoints that reply with an empty body.
export async function hailApiFetchVoid(url: string, init?: RequestInit): Promise<void> {
  const resp = await hailFetch(url, init);
  await resp.text().catch(() => undefined);
}
