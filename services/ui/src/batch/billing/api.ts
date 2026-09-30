export interface BillingProject {
  billing_project: string;
  status: string;
  // Usernames of the billing project's members.
  users: string[];
  // Members plus the managers of the billing project's quote, with their roles
  // ("<billing_project>:member", "<quote>:owner", "<quote>:manager").
  user_roles: { user: string; roles: string[] }[];
  limit: number | null;
  quote_id: number;
  quote_name: string;
  can_view_quote: boolean;
  remaining: number | null;
  accrued_cost: number;
  description: string | null;
  billing_role: import('./permissions').BillingRole | null;
}

export interface QuoteManager {
  user: string;
  role: string;
}

export interface Quote {
  id: number;
  name: string;
  state: string;
  quote_number: string | null;
  cost_object: string;
  authorized_amount: number | null;
  pi_name: string | null;
  pm_designee: string | null;
  description: string | null;
  time_created: string;
  managers: QuoteManager[];
  billing_projects: BillingProject[];
  billing_role: import('./permissions').BillingRole | null;
}

export interface BillingEvent {
  id: number;
  timestamp: number;
  actor: string;
  action: string;
  target_user: string | null;
  target_project?: string | null;
  detail: string | null;
  comment: string | null;
}

// The built-in quote that, alone, may have an unlimited authorized amount and unlimited billing projects.
export const INTERNAL_QUOTE_NAME = 'INTERNAL';

async function responseError(resp: Response): Promise<Error> {
  const text = (await resp.text().catch(() => '')).trim();
  // aiohttp error bodies look like "400: <reason>"; the reason is the useful part to show.
  const reason = text.replace(/^\d{3}: /, '');
  return new Error(reason || `HTTP ${resp.status}`);
}

export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export async function fetchJson<T>(url: string): Promise<T> {
  const resp = await fetch(url, { credentials: 'same-origin' });
  if (!resp.ok) {
    throw await responseError(resp);
  }
  return resp.json() as Promise<T>;
}

function getCsrfToken(): string {
  return document.querySelector<HTMLMetaElement>('meta[name="csrf"]')?.getAttribute('value') ?? '';
}

export async function apiCall(method: string, url: string, body?: object): Promise<void> {
  const headers: Record<string, string> = { 'X-CSRF-Token': getCsrfToken() };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const resp = await fetch(url, {
    method,
    credentials: 'same-origin',
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!resp.ok) {
    throw await responseError(resp);
  }
}
