import { useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer } from 'recharts';
import type { BillingEvent, BillingProject, Quote } from './api';
import { fetchJson, apiCall } from './api';
import { fmtCost, fmtDollars } from './fmt';

interface BudgetBarProps {
  accrued: number;
  limit: number | null;
}

export function BudgetBar({ accrued, limit }: BudgetBarProps) {
  const isOver = limit !== null && accrued >= limit;
  const isLow = !isOver && limit !== null && accrued >= limit * 0.9;

  if (limit === null) {
    return (
      <div>
        <div className="relative h-5 bg-slate-200 rounded overflow-hidden">
          <div className="absolute left-0 top-0 h-full w-1 bg-green-400" />
        </div>
        <div className="text-xs text-slate-500 mt-1">{fmtCost(accrued)}</div>
      </div>
    );
  }

  const total = Math.max(limit, accrued);
  const data = [{ accrued, remaining: Math.max(0, limit - accrued) }];

  return (
    <div>
      <ResponsiveContainer width="100%" height={32}>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
          <XAxis type="number" domain={[0, total]} hide />
          <YAxis type="category" hide />
          <Bar dataKey="accrued" stackId="a" fill={isOver ? '#ef4444' : isLow ? '#fbbf24' : '#38bdf8'} radius={[3, 0, 0, 3]} isAnimationActive={false} />
          <Bar dataKey="remaining" stackId="a" fill="#e2e8f0" radius={[0, 3, 3, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
      <div className={`text-xs mt-1 ${isLow ? 'text-amber-700 font-medium' : 'text-slate-500'}`}>
        {fmtCost(accrued)}
      </div>
    </div>
  );
}

export function QuoteBudgetBar({ spent, allocated, authorized }: {
  spent: number;
  allocated: number;
  authorized: number | null;
}) {
  if (authorized === null) {
    return (
      <div>
        <div className="relative h-5 bg-slate-200 rounded overflow-hidden">
          <div className="absolute left-0 top-0 h-full w-1 bg-green-400" />
        </div>
        <div className="text-xs text-slate-500 mt-1">{fmtDollars(allocated)} allocated · {fmtCost(spent)} spent</div>
      </div>
    );
  }

  const allocPct = Math.min((allocated / authorized) * 100, 100);
  const spentPct = Math.min((spent / authorized) * 100, 100);

  return (
    <div>
      <div className="relative w-full h-5 bg-slate-200 rounded overflow-hidden">
        <div className="absolute left-0 top-0 h-full bg-blue-400 rounded" style={{ width: `${allocPct}%` }} />
        <div className="absolute left-0 top-0 h-full bg-teal-400 rounded" style={{ width: `${spentPct}%` }} />
      </div>
      <div className="text-xs text-slate-500 mt-1 flex gap-3">
        <span><span className="inline-block w-2 h-2 rounded-sm bg-teal-400 mr-1" />spent {fmtCost(spent)}</span>
        <span><span className="inline-block w-2 h-2 rounded-sm bg-blue-400 mr-1" />allocated {fmtDollars(allocated)}</span>
        <span><span className="inline-block w-2 h-2 rounded-sm bg-slate-200 mr-1" />unallocated {fmtDollars(authorized - allocated)}</span>
      </div>
    </div>
  );
}

export function QuoteCompactBudgetBar({ spent, allocated, authorized }: {
  spent: number;
  allocated: number;
  authorized: number | null;
}) {
  if (authorized === null) {
    return (
      <div className="relative min-w-[120px] h-3 bg-slate-200 rounded overflow-hidden">
        <div className="absolute left-0 top-0 h-full w-1 bg-green-400" />
      </div>
    );
  }

  const allocPct = Math.min((allocated / authorized) * 100, 100);
  const spentPct = Math.min((spent / authorized) * 100, 100);

  return (
    <div className="relative min-w-[120px] h-3 bg-slate-200 rounded overflow-hidden">
      <div className="absolute left-0 top-0 h-full bg-blue-400 rounded" style={{ width: `${allocPct}%` }} />
      <div className="absolute left-0 top-0 h-full bg-teal-400 rounded" style={{ width: `${spentPct}%` }} />
    </div>
  );
}

export function CompactBudgetBar({ accrued, limit }: BudgetBarProps) {
  if (limit === null) {
    return (
      <div className="relative min-w-[120px] h-3 bg-slate-200 rounded overflow-hidden">
        <div className="absolute left-0 top-0 h-full w-1 bg-green-400" />
      </div>
    );
  }

  const isOver = accrued >= limit;
  const isLow = !isOver && accrued >= limit * 0.9;
  const pct = Math.min((accrued / limit) * 100, 100);

  return (
    <div className="relative min-w-[120px] h-3 bg-slate-200 rounded overflow-hidden">
      <div
        className={`h-full rounded ${isOver ? 'bg-red-500' : isLow ? 'bg-amber-400' : 'bg-sky-400'}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function isLowBudget(bp: BillingProject): boolean {
  return bp.limit !== null && bp.accrued_cost >= bp.limit * 0.9;
}

type BPSortKey = 'name' | 'quote' | 'spent' | 'limit' | 'usage';
type SortDir = 'asc' | 'desc';

const SORT_DEFAULT_DIR: Record<BPSortKey, SortDir> = {
  name: 'asc',
  quote: 'asc',
  spent: 'desc',
  limit: 'desc',
  usage: 'desc',
};

function sortBps(bps: BillingProject[], key: BPSortKey, dir: SortDir): BillingProject[] {
  return [...bps].sort((a, b) => {
    let cmp = 0;
    switch (key) {
      case 'name':
        cmp = a.billing_project.localeCompare(b.billing_project);
        break;
      case 'quote':
        cmp = a.quote_name.localeCompare(b.quote_name);
        break;
      case 'spent':
        cmp = a.accrued_cost - b.accrued_cost;
        break;
      case 'limit':
        // treat null (unlimited) as Infinity — largest value, so first when descending
        if (a.limit === null && b.limit === null) cmp = 0;
        else if (a.limit === null) cmp = 1;
        else if (b.limit === null) cmp = -1;
        else cmp = a.limit - b.limit;
        break;
      case 'usage': {
        const pA = a.limit === null ? 0 : a.accrued_cost / a.limit;
        const pB = b.limit === null ? 0 : b.accrued_cost / b.limit;
        cmp = pA - pB;
        if (cmp === 0) {
          if (a.limit === null && b.limit === null) cmp = 0;
          else if (a.limit === null) cmp = 1;
          else if (b.limit === null) cmp = -1;
          else cmp = a.limit - b.limit;
        }
        break;
      }
    }
    return dir === 'asc' ? cmp : -cmp;
  });
}

function SortTh({ label, sortKey, current, dir, onSort }: {
  label: string;
  sortKey: BPSortKey;
  current: BPSortKey;
  dir: SortDir;
  onSort: (k: BPSortKey) => void;
}) {
  const active = sortKey === current;
  return (
    <th
      className="text-left p-3 font-medium cursor-pointer select-none hover:bg-slate-100"
      onClick={() => onSort(sortKey)}
    >
      <div className="flex items-center gap-1">
        {label}
        <span className="material-symbols-outlined text-sm text-slate-400">
          {active ? (dir === 'asc' ? 'arrow_upward' : 'arrow_downward') : 'unfold_more'}
        </span>
      </div>
    </th>
  );
}

export function BillingProjectsTable({ label, bps, basePath, emptyMessage, defaultOpen = true, showQuote = false }: {
  label: string;
  bps: BillingProject[];
  basePath: string;
  emptyMessage: string;
  defaultOpen?: boolean;
  showQuote?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const [sortKey, setSortKey] = useState<BPSortKey>('name');
  const [sortDir, setSortDir] = useState<SortDir>('asc');

  const handleSort = (key: BPSortKey) => {
    if (key === sortKey) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    else { setSortKey(key); setSortDir(SORT_DEFAULT_DIR[key]); }
  };

  const sorted = sortBps(bps, sortKey, sortDir);

  return (
    <section className="border rounded mb-6">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 bg-slate-100 px-4 py-2 rounded-t text-left hover:bg-slate-200"
      >
        <span className="material-symbols-outlined text-slate-500 text-base">
          {open ? 'expand_more' : 'chevron_right'}
        </span>
        <span className="font-medium text-sm uppercase tracking-wide text-slate-600">{label} ({bps.length})</span>
      </button>
      {open && (bps.length > 0 ? (
        <div>
          <table className="w-full text-sm">
            <thead className="bg-slate-50 sticky top-0 z-10">
              <tr>
                <SortTh label="Name"  sortKey="name"  current={sortKey} dir={sortDir} onSort={handleSort} />
                {showQuote && <SortTh label="Quote" sortKey="quote" current={sortKey} dir={sortDir} onSort={handleSort} />}
                <SortTh label="Spent" sortKey="spent" current={sortKey} dir={sortDir} onSort={handleSort} />
                <SortTh label="Limit" sortKey="limit" current={sortKey} dir={sortDir} onSort={handleSort} />
                <SortTh label="Usage" sortKey="usage" current={sortKey} dir={sortDir} onSort={handleSort} />
              </tr>
            </thead>
            <tbody>
              {sorted.map((bp) => {
                const low = isLowBudget(bp);
                return (
                  <tr key={bp.billing_project} className={`border-t hover:bg-slate-50 ${low ? 'bg-amber-50' : ''}`}>
                    <td className="p-3">
                      <a href={`${basePath}/billing_projects/${bp.billing_project}`} className="text-blue-600 hover:underline">
                        {bp.billing_project}
                      </a>
                    </td>
                    {showQuote && (
                      <td className="p-3 text-slate-700">
                        {bp.can_view_quote ? (
                          <a href={`${basePath}/billing/quotes/${bp.quote_name}`} className="text-blue-600 hover:underline">
                            {bp.quote_name}
                          </a>
                        ) : (
                          bp.quote_name
                        )}
                      </td>
                    )}
                    <td className="p-3 text-slate-700">{fmtDollars(bp.accrued_cost)}</td>
                    <td className="p-3 text-slate-700">{fmtDollars(bp.limit)}</td>
                    <td className="p-3">
                      <CompactBudgetBar accrued={bp.accrued_cost} limit={bp.limit} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="p-4 text-sm text-slate-500">{emptyMessage}</p>
      ))}
    </section>
  );
}

type QuoteSortKey = 'name' | 'cost_object' | 'pi_name' | 'pm_designee' | 'spent' | 'allocated' | 'limit' | 'usage';

const QUOTE_SORT_DEFAULT_DIR: Record<QuoteSortKey, SortDir> = {
  name: 'asc',
  cost_object: 'asc',
  pi_name: 'asc',
  pm_designee: 'asc',
  spent: 'desc',
  allocated: 'desc',
  limit: 'desc',
  usage: 'desc',
};

function totalSpent(q: Quote): number {
  return (q.billing_projects ?? []).reduce((s, bp) => s + bp.accrued_cost, 0);
}

function totalAllocated(q: Quote): number {
  return (q.billing_projects ?? []).reduce((s, bp) => s + (bp.limit ?? 0), 0);
}

function sortQuotes(quotes: Quote[], key: QuoteSortKey, dir: SortDir): Quote[] {
  return [...quotes].sort((a, b) => {
    let cmp = 0;
    switch (key) {
      case 'name':        cmp = a.name.localeCompare(b.name); break;
      case 'cost_object': cmp = a.cost_object.localeCompare(b.cost_object); break;
      case 'pi_name':     cmp = (a.pi_name ?? '').localeCompare(b.pi_name ?? ''); break;
      case 'pm_designee': cmp = (a.pm_designee ?? '').localeCompare(b.pm_designee ?? ''); break;
      case 'spent':       cmp = totalSpent(a) - totalSpent(b); break;
      case 'allocated':   cmp = totalAllocated(a) - totalAllocated(b); break;
      case 'limit':
        if (a.authorized_amount === null && b.authorized_amount === null) cmp = 0;
        else if (a.authorized_amount === null) cmp = 1;
        else if (b.authorized_amount === null) cmp = -1;
        else cmp = a.authorized_amount - b.authorized_amount;
        break;
      case 'usage': {
        const pA = a.authorized_amount === null ? 0 : totalAllocated(a) / a.authorized_amount;
        const pB = b.authorized_amount === null ? 0 : totalAllocated(b) / b.authorized_amount;
        cmp = pA - pB;
        break;
      }
    }
    return dir === 'asc' ? cmp : -cmp;
  });
}

function QuoteSortTh({ label, sortKey, current, dir, onSort }: {
  label: string; sortKey: QuoteSortKey; current: QuoteSortKey; dir: SortDir; onSort: (k: QuoteSortKey) => void;
}) {
  const active = sortKey === current;
  return (
    <th className="text-left p-3 font-medium cursor-pointer select-none hover:bg-slate-100" onClick={() => onSort(sortKey)}>
      <div className="flex items-center gap-1">
        {label}
        <span className="material-symbols-outlined text-sm text-slate-400">
          {active ? (dir === 'asc' ? 'arrow_upward' : 'arrow_downward') : 'unfold_more'}
        </span>
      </div>
    </th>
  );
}

export function QuotesTable({ label, quotes, basePath, emptyMessage, defaultOpen = true }: {
  label: string;
  quotes: Quote[];
  basePath: string;
  emptyMessage: string;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const [sortKey, setSortKey] = useState<QuoteSortKey>('name');
  const [sortDir, setSortDir] = useState<SortDir>('asc');

  const handleSort = (key: QuoteSortKey) => {
    if (key === sortKey) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    else { setSortKey(key); setSortDir(QUOTE_SORT_DEFAULT_DIR[key]); }
  };

  const sorted = sortQuotes(quotes, sortKey, sortDir);

  return (
    <section className="border rounded mb-6">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 bg-slate-100 px-4 py-2 rounded-t text-left hover:bg-slate-200"
      >
        <span className="material-symbols-outlined text-slate-500 text-base">
          {open ? 'expand_more' : 'chevron_right'}
        </span>
        <span className="font-medium text-sm uppercase tracking-wide text-slate-600">{label} ({quotes.length})</span>
      </button>
      {open && (quotes.length > 0 ? (
        <div>
          <table className="w-full text-sm">
            <thead className="bg-slate-50 sticky top-0 z-10">
              <tr>
                <QuoteSortTh label="Name"        sortKey="name"        current={sortKey} dir={sortDir} onSort={handleSort} />
                <QuoteSortTh label="Cost Object" sortKey="cost_object" current={sortKey} dir={sortDir} onSort={handleSort} />
                <QuoteSortTh label="PI Name"     sortKey="pi_name"     current={sortKey} dir={sortDir} onSort={handleSort} />
                <QuoteSortTh label="PM Designee" sortKey="pm_designee" current={sortKey} dir={sortDir} onSort={handleSort} />
                <QuoteSortTh label="Authorized"  sortKey="limit"       current={sortKey} dir={sortDir} onSort={handleSort} />
                <QuoteSortTh label="Allocated"   sortKey="allocated"   current={sortKey} dir={sortDir} onSort={handleSort} />
                <QuoteSortTh label="Spent"       sortKey="spent"       current={sortKey} dir={sortDir} onSort={handleSort} />
                <QuoteSortTh label="Usage"       sortKey="usage"       current={sortKey} dir={sortDir} onSort={handleSort} />
              </tr>
            </thead>
            <tbody>
              {sorted.map((q) => {
                const spent = totalSpent(q);
                const allocated = totalAllocated(q);
                return (
                  <tr key={q.id} className="border-t hover:bg-slate-50">
                    <td className="p-3">
                      <a href={`${basePath}/billing/quotes/${q.name}`} className="text-blue-600 hover:underline">
                        {q.name}
                      </a>
                    </td>
                    <td className="p-3 text-slate-700">{q.cost_object}</td>
                    <td className="p-3 text-slate-700">{q.pi_name ?? '—'}</td>
                    <td className="p-3 text-slate-700">{q.pm_designee ?? '—'}</td>
                    <td className="p-3 text-slate-700">{fmtDollars(q.authorized_amount)}</td>
                    <td className="p-3 text-slate-700">{fmtDollars(allocated)}</td>
                    <td className="p-3 text-slate-700">{fmtDollars(spent)}</td>
                    <td className="p-3">
                      <QuoteCompactBudgetBar spent={spent} allocated={allocated} authorized={q.authorized_amount} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="p-4 text-sm text-slate-500">{emptyMessage}</p>
      ))}
    </section>
  );
}

export function SectionHeader({ label }: { label: string }) {
  return (
    <div className="bg-slate-100 px-4 py-2 font-medium text-sm uppercase tracking-wide text-slate-600 rounded-t">
      {label}
    </div>
  );
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-2 rounded text-sm mb-4">
      {message}
    </div>
  );
}

export function BackLink({ href, label }: { href: string; label: string }) {
  return (
    <a href={href} className="text-slate-500 hover:text-slate-700 flex items-center gap-1 text-sm">
      <span className="material-symbols-outlined text-base">arrow_back</span>
      {label}
    </a>
  );
}

interface EditableRowProps {
  label: string;
  value: string;
  displayValue?: string;
  canEdit: boolean;
  inputType?: 'text' | 'number';
  prefix?: string;
  placeholder?: string;
  onSave: (val: string) => Promise<void>;
}

export function EditableRow({ label, value, displayValue, canEdit, inputType = 'text', prefix, placeholder, onSave }: EditableRowProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      await onSave(draft);
      setEditing(false);
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <tr className="border-b border-slate-100">
      <td className="py-2 pl-4 pr-8 text-slate-500 w-40 align-middle">{label}</td>
      <td className="py-2 align-middle">
        {editing ? (
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1">
              {prefix && <span className="text-sm text-slate-500">{prefix}</span>}
              <input
                type={inputType}
                min={inputType === 'number' ? '0' : undefined}
                step={inputType === 'number' ? '0.01' : undefined}
                className="border rounded px-2 py-1 w-40 text-sm"
                value={draft}
                placeholder={placeholder}
                onChange={(e) => setDraft(e.target.value)}
                autoFocus
              />
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => void handleSave()}
                disabled={saving}
                className="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700 disabled:opacity-50"
              >
                Save
              </button>
              <button
                onClick={() => { setEditing(false); setDraft(value); setError(null); }}
                className="text-slate-500 hover:text-slate-700 text-sm"
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <span>{displayValue ?? (value || '—')}</span>
        )}
        {error && <div className="text-red-600 text-xs mt-1">{error}</div>}
      </td>
      <td className="py-2 pr-4 text-right align-middle w-10">
        {canEdit && !editing && (
          <button
            onClick={() => { setDraft(value); setEditing(true); setError(null); }}
            className="hover:bg-slate-200 rounded p-0.5 text-slate-400 hover:text-slate-600"
          >
            <span className="material-symbols-outlined text-base">edit</span>
          </button>
        )}
      </td>
    </tr>
  );
}

export interface ConfirmModalProps {
  title: string;
  message?: string;
  confirmLabel?: string;
  danger?: boolean;
  onConfirm: (comment: string) => Promise<void>;
  onClose: () => void;
}

export function ConfirmModal({ title, message, confirmLabel = 'Confirm', danger = false, onConfirm, onClose }: ConfirmModalProps) {
  const [comment, setComment] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleConfirm = async () => {
    setSubmitting(true);
    setError(null);
    try {
      await onConfirm(comment);
    } catch (e) {
      setError(String(e));
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-30 flex items-center justify-center z-50">
      <div className="bg-white rounded shadow-lg w-full max-w-md p-6">
        <h2 className="text-xl font-light mb-2">{title}</h2>
        {message && <p className="text-sm text-slate-600 mb-4">{message}</p>}
        <div className="mb-4">
          <label className="block text-sm text-slate-600 mb-1">Comment <span className="text-slate-400">(optional)</span></label>
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            rows={3}
            className="border rounded px-2 py-1 w-full text-sm resize-none"
            placeholder="Reason for this action…"
            autoFocus
          />
        </div>
        {error && <div className="text-red-600 text-xs mb-3">{error}</div>}
        <div className="flex justify-end gap-2">
          <button
            onClick={onClose}
            disabled={submitting}
            className="border border-gray-300 px-3 py-1.5 rounded text-sm hover:bg-slate-50 disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={() => void handleConfirm()}
            disabled={submitting}
            className={`text-white px-4 py-1.5 rounded text-sm disabled:opacity-50 ${danger ? 'bg-red-600 hover:bg-red-700' : 'bg-blue-600 hover:bg-blue-700'}`}
          >
            {submitting ? 'Working…' : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

export function CreateBpModal({ basePath, fixedQuoteName, onClose, onCreated }: {
  basePath: string;
  fixedQuoteName?: string;
  onClose: () => void;
  onCreated: () => void;
}) {
  const [name, setName] = useState('');
  const [quoteName, setQuoteName] = useState(fixedQuoteName ?? '');
  const [quotes, setQuotes] = useState<Quote[] | null>(null);
  const [selectedQuote, setSelectedQuote] = useState<Quote | null>(null);
  const [limit, setLimit] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (fixedQuoteName) return;
    fetchJson<Quote[]>(`${basePath}/api/v1alpha/quotes`)
      .then((qs) => { setQuotes(qs); if (qs.length > 0) setQuoteName(qs[0].name); })
      .catch(() => setQuotes([]));
  }, [basePath, fixedQuoteName]);

  useEffect(() => {
    if (!quoteName) return;
    setSelectedQuote(null);
    fetchJson<Quote>(`${basePath}/api/v1alpha/quotes/${encodeURIComponent(quoteName)}`)
      .then(setSelectedQuote)
      .catch(() => {});
  }, [basePath, quoteName]);

  const allocated = selectedQuote?.billing_projects.reduce((s, bp) => s + (bp.limit ?? 0), 0) ?? 0;
  const maxAvailable = selectedQuote?.authorized_amount !== null && selectedQuote?.authorized_amount !== undefined
    ? selectedQuote.authorized_amount - allocated
    : null;
  const limitRequired = maxAvailable !== null;

  const handleSubmit = async () => {
    if (!name.trim()) { setError('Name is required.'); return; }
    if (!quoteName) { setError('Quote is required.'); return; }
    if (limitRequired && limit === '') { setError('A limit is required for this quote.'); return; }
    const limitVal = limit === '' ? null : parseFloat(limit);
    if (maxAvailable !== null && limitVal !== null && limitVal > maxAvailable) {
      setError(`Limit cannot exceed ${fmtDollars(maxAvailable)} (remaining on this quote).`);
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await apiCall('POST', `${basePath}/api/v1alpha/billing_projects/${encodeURIComponent(name)}/create`, {
        quote_name: quoteName,
        limit: limitVal,
      });
      onCreated();
      onClose();
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-30 flex items-center justify-center z-50">
      <div className="bg-white rounded shadow-lg w-full max-w-md p-6">
        <h2 className="text-xl font-light mb-4">New Billing Project</h2>
        <div className="space-y-3 text-sm">
          <div>
            <label className="block mb-1 text-slate-600">Name <span className="text-red-500">*</span></label>
            <input
              type="text" value={name} onChange={(e) => setName(e.target.value)}
              className="border rounded px-2 py-1 w-full" spellCheck={false} autoCorrect="off" autoFocus
            />
          </div>
          <div>
            <label className="block mb-1 text-slate-600">Quote <span className="text-red-500">*</span></label>
            {fixedQuoteName ? (
              <span className="text-slate-700">{fixedQuoteName}</span>
            ) : quotes === null ? (
              <span className="text-slate-400 text-xs">Loading…</span>
            ) : (
              <select
                value={quoteName}
                onChange={(e) => setQuoteName(e.target.value)}
                className="border rounded px-2 py-1 w-full"
              >
                {quotes.map((q) => <option key={q.id} value={q.name}>{q.name}</option>)}
              </select>
            )}
          </div>
          <div>
            <label className="block mb-1 text-slate-600">
              Limit{limitRequired && <span className="text-red-500"> *</span>}
            </label>
            <div className="flex items-center gap-1">
              <span className="text-slate-500">$</span>
              <input
                type="number" min="0" step="0.01" value={limit}
                onChange={(e) => setLimit(e.target.value)}
                className="border rounded px-2 py-1 w-full" placeholder="1.00"
              />
            </div>
            {selectedQuote !== null && (
              <p className="text-slate-500 text-xs mt-1">
                {maxAvailable !== null
                  ? `${fmtDollars(maxAvailable)} remains unallocated in this quote`
                  : 'leave empty for unlimited'}
              </p>
            )}
          </div>
        </div>
        {error && <div className="text-red-600 text-xs mt-2">{error}</div>}
        <div className="flex justify-end gap-2 mt-4">
          <button onClick={onClose} className="border border-gray-300 px-3 py-1.5 rounded text-sm hover:bg-slate-50">
            Cancel
          </button>
          <button
            onClick={() => void handleSubmit()} disabled={saving}
            className="bg-blue-600 text-white px-4 py-1.5 rounded text-sm hover:bg-blue-700 disabled:opacity-50"
          >
            Create
          </button>
        </div>
      </div>
    </div>
  );
}

export interface EventLogColumn {
  key: keyof BillingEvent;
  label: string;
  render?: (value: BillingEvent[keyof BillingEvent], event: BillingEvent) => ReactNode;
}

export function EventLog({ events, columns }: { events: BillingEvent[]; columns: EventLogColumn[] }) {
  if (events.length === 0) {
    return <p className="p-4 text-sm text-slate-500">No events yet.</p>;
  }
  return (
    <div className="overflow-auto max-h-96">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-slate-50">
          <tr>
            {columns.map((c) => (
              <th key={c.key} className="text-left p-2 font-medium">{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {events.map((e) => (
            <tr key={e.id} className="border-t hover:bg-slate-50">
              {columns.map((c) => {
                const val = e[c.key];
                return (
                  <td key={c.key} className="p-2 text-slate-700">
                    {c.render ? c.render(val, e) : String(val ?? '')}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
