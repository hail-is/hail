import { useState, useEffect, useCallback } from 'react';
import type { BillingProject, BillingEvent, Quote } from './api';
import { fetchJson, apiCall } from './api';
import { fmtDollars, fmtTimestamp } from './fmt';
import { can } from './permissions';
import { SectionHeader, ErrorBanner, EditableRow, EventLog, ConfirmModal, BudgetBar } from './shared';

function MoveQuoteModal({ basePath, bpName, currentQuoteName, onClose, onMoved }: {
  basePath: string;
  bpName: string;
  currentQuoteName: string;
  onClose: () => void;
  onMoved: () => void;
}) {
  const [quotes, setQuotes] = useState<Quote[] | null>(null);
  const [destQuote, setDestQuote] = useState('');
  const [comment, setComment] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchJson<Quote[]>(`${basePath}/api/v1alpha/quotes`)
      .then((qs) => {
        const others = qs.filter((q) => q.name !== currentQuoteName);
        setQuotes(others);
        if (others.length > 0) setDestQuote(others[0].name);
      })
      .catch(() => setQuotes([]));
  }, [basePath, currentQuoteName]);

  const handleSubmit = async () => {
    if (!destQuote) { setError('Select a destination quote.'); return; }
    setSaving(true);
    setError(null);
    try {
      await apiCall('POST', `${basePath}/api/v1alpha/billing_projects/${encodeURIComponent(bpName)}/change_quote`, {
        quote_name: destQuote,
        comment: comment || undefined,
      });
      onMoved();
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
        <h2 className="text-xl font-light mb-4">Move to another quote</h2>
        <div className="space-y-3 text-sm">
          <div>
            <label className="block mb-1 text-slate-600">Destination quote <span className="text-red-500">*</span></label>
            {quotes === null ? (
              <span className="text-slate-400 text-xs">Loading…</span>
            ) : quotes.length === 0 ? (
              <span className="text-slate-500 text-xs">No other quotes available.</span>
            ) : (
              <select
                value={destQuote}
                onChange={(e) => setDestQuote(e.target.value)}
                className="border rounded px-2 py-1 w-full"
              >
                {quotes.map((q) => <option key={q.id} value={q.name}>{q.name}</option>)}
              </select>
            )}
          </div>
          <div>
            <label className="block mb-1 text-slate-600">Comment <span className="text-slate-400">(optional)</span></label>
            <textarea
              value={comment} onChange={(e) => setComment(e.target.value)}
              rows={3} className="border rounded px-2 py-1 w-full text-sm resize-none"
              placeholder="Reason for moving…"
            />
          </div>
        </div>
        {error && <div className="text-red-600 text-xs mt-2">{error}</div>}
        <div className="flex justify-end gap-2 mt-4">
          <button onClick={onClose} className="border border-gray-300 px-3 py-1.5 rounded text-sm hover:bg-slate-50">
            Cancel
          </button>
          <button
            onClick={() => void handleSubmit()}
            disabled={saving || !destQuote || (quotes !== null && quotes.length === 0)}
            className="bg-blue-600 text-white px-4 py-1.5 rounded text-sm hover:bg-blue-700 disabled:opacity-50"
          >
            Move
          </button>
        </div>
      </div>
    </div>
  );
}

interface Props {
  basePath: string;
  bpName: string;
}

const BP_EVENT_COLUMNS = [
  { key: 'timestamp' as const, label: 'Time', render: (v: unknown) => <span className="whitespace-nowrap text-slate-500">{fmtTimestamp(v as number)}</span> },
  { key: 'actor' as const, label: 'Actor' },
  { key: 'action' as const, label: 'Action' },
  { key: 'target_user' as const, label: 'Target' },
  { key: 'detail' as const, label: 'Detail' },
  { key: 'comment' as const, label: 'Comment', render: (v: unknown) => <span className="text-slate-500 italic">{String(v ?? '')}</span> },
];

export function BillingProjectPage({ basePath, bpName }: Props) {
  const [bp, setBp] = useState<BillingProject | null>(null);
  const [events, setEvents] = useState<BillingEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [addUser, setAddUser] = useState('');
  const [memberError, setMemberError] = useState<string | null>(null);
  const [modal, setModal] = useState<'close' | 'reopen' | 'move' | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [bpData, evData] = await Promise.all([
        fetchJson<BillingProject>(`${basePath}/api/v1alpha/billing_projects/${encodeURIComponent(bpName)}`),
        fetchJson<BillingEvent[]>(`${basePath}/api/v1alpha/billing_projects/${encodeURIComponent(bpName)}/events`),
      ]);
      setBp(bpData);
      setEvents(evData);
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [basePath, bpName]);

  useEffect(() => { void fetchData(); }, [fetchData]);

  const patch = async (updates: object) => {
    await apiCall('PATCH', `${basePath}/api/v1alpha/billing_projects/${encodeURIComponent(bpName)}`, updates);
    await fetchData();
  };

  const handleClose = async (comment: string) => {
    await apiCall('POST', `${basePath}/api/v1alpha/billing_projects/${encodeURIComponent(bpName)}/close`, { comment: comment || undefined });
    window.location.reload();
  };

  const handleReopen = async (comment: string) => {
    await apiCall('POST', `${basePath}/api/v1alpha/billing_projects/${encodeURIComponent(bpName)}/reopen`, { comment: comment || undefined });
    window.location.reload();
  };

  const removeMember = async (user: string) => {
    setMemberError(null);
    try {
      await apiCall('POST', `${basePath}/api/v1alpha/billing_projects/${encodeURIComponent(bpName)}/users/${encodeURIComponent(user)}/remove`);
      await fetchData();
    } catch (e) {
      setMemberError(String(e));
    }
  };

  const addMember = async () => {
    const u = addUser.trim();
    if (!u) return;
    setMemberError(null);
    try {
      await apiCall('POST', `${basePath}/api/v1alpha/billing_projects/${encodeURIComponent(bpName)}/users/${encodeURIComponent(u)}/add`);
      setAddUser('');
      await fetchData();
    } catch (e) {
      setMemberError(String(e));
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center mt-24">
        <span className="text-5xl font-light text-sky-600">Loading&hellip;</span>
      </div>
    );
  }

  if (error) return <ErrorBanner message={error} />;
  if (!bp) return null;

  const billingRole = bp?.billing_role ?? null;
  const canEditLimit = can(billingRole, 'edit_bp_limit');
  const canManageMembers = can(billingRole, 'manage_bp_members');
  const canCloseReopen = can(billingRole, 'close_reopen_bp');
  const canChangeQuote = can(billingRole, 'change_bp_quote');
  const canViewQuote = can(billingRole, 'view_quote');

  return (
    <div className="max-w-4xl">
      <div className="flex items-center gap-2 mb-6 text-2xl font-light">
        <span className="text-slate-400">Billing</span>
        <span className="text-slate-300">›</span>
        {canViewQuote ? (
          <a href={`${basePath}/billing/quotes`} className="text-slate-400 hover:text-slate-600">Quotes</a>
        ) : (
          <span className="text-slate-400">Quotes</span>
        )}
        <span className="text-slate-300">›</span>
        {canViewQuote ? (
          <a href={`${basePath}/billing/quotes/${bp.quote_name}`} className="text-slate-400 hover:text-slate-600">{bp.quote_name}</a>
        ) : (
          <span className="text-slate-400">{bp.quote_name}</span>
        )}
        <span className="text-slate-300">›</span>
        <span>{bpName}</span>
        {bp.status === 'open' ? (
          <span className="text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded-full">open</span>
        ) : (
          <span className="text-xs bg-gray-200 text-gray-600 px-2 py-0.5 rounded-full italic">{bp.status}</span>
        )}
      </div>

      {/* Funding */}
      <section className="border rounded mb-6">
        <SectionHeader label="Funding" />
        <table className="w-full text-sm">
          <tbody>
            <tr className="border-b border-slate-100">
              <td className="py-2 pl-4 pr-8 text-slate-500 w-40 align-middle">Quote</td>
              <td className="py-2 align-middle" colSpan={2}>
                {canViewQuote ? (
                  <a href={`${basePath}/billing/quotes/${bp.quote_name}`} className="text-blue-600 hover:underline">
                    {bp.quote_name}
                  </a>
                ) : (
                  bp.quote_name
                )}
              </td>
            </tr>
            <EditableRow
              label="Description"
              value={bp.description ?? ''}
              canEdit={bp.status === 'open'}
              onSave={(val) => patch({ description: val || null })}
            />
            <EditableRow
              label="Limit"
              value={bp.limit !== null ? String(bp.limit) : ''}
              displayValue={fmtDollars(bp.limit)}
              canEdit={canEditLimit && bp.status === 'open'}
              inputType="number"
              prefix="$"
              placeholder="blank = unlimited"
              onSave={(val) => patch({ limit: val === '' ? null : parseFloat(val) })}
            />
            <tr className="border-b border-slate-100">
              <td className="py-2 pl-4 pr-8 text-slate-500 w-40 align-middle">Spent</td>
              <td className="py-2 pr-4 align-middle" colSpan={2}>
                <BudgetBar accrued={bp.accrued_cost} limit={bp.limit} />
              </td>
            </tr>
          </tbody>
        </table>
      </section>

      {/* Members */}
      <section className="border rounded mb-6">
        <SectionHeader label="Members" />
        <div className="p-4">
          <table className="w-full text-sm">
            <tbody>
              {(bp.users ?? []).map((entry) => {
                const isExplicitMember = entry.roles.includes(`${bpName}:member`);
                const quoteRoles = entry.roles.filter((r) => !r.endsWith(':member'));
                return (
                  <tr key={entry.user} className="hover:bg-slate-50">
                    <td className="py-1 pr-4">
                      {entry.user}
                      {quoteRoles.map((r) => {
                        const role = r.split(':')[1];
                        return (
                          <span key={r} className="ml-2 text-xs bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded-full">
                            {role === 'owner' ? 'quote owner' : 'quote manager'}
                          </span>
                        );
                      })}
                    </td>
                    <td className="py-1 text-right">
                      {canManageMembers && bp.status === 'open' && isExplicitMember && (
                        <button
                          onClick={() => void removeMember(entry.user)}
                          className="text-red-400 hover:text-red-600"
                        >
                          <span className="material-symbols-outlined text-base">close</span>
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {memberError && <div className="text-red-600 text-xs mt-1">{memberError}</div>}
          {canManageMembers && bp.status === 'open' && (
            <div className="mt-3 flex items-center gap-2">
              <input
                type="text"
                value={addUser}
                onChange={(e) => setAddUser(e.target.value)}
                placeholder="username"
                spellCheck={false}
                autoComplete="off"
                data-lpignore="true"
                data-1p-ignore
                className="border rounded px-2 py-1 text-sm w-48"
                onKeyDown={(e) => { if (e.key === 'Enter') void addMember(); }}
              />
              <button
                onClick={() => void addMember()}
                className="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700"
              >
                Add
              </button>
            </div>
          )}
        </div>
      </section>

      {/* Actions */}
      {(canCloseReopen || canChangeQuote) && (
        <section className="border rounded mb-6">
          <SectionHeader label="Actions" />
          <div className="p-4 flex flex-wrap gap-2">
            {canCloseReopen && (bp.status === 'open' ? (
              <button
                onClick={() => setModal('close')}
                className="bg-red-600 text-white px-3 py-1.5 rounded text-sm hover:bg-red-700"
              >
                Close billing project
              </button>
            ) : (
              <button
                onClick={() => setModal('reopen')}
                className="bg-blue-600 text-white px-3 py-1.5 rounded text-sm hover:bg-blue-700"
              >
                Reopen billing project
              </button>
            ))}
            {canChangeQuote && (
              <button
                onClick={() => setModal('move')}
                className="border border-slate-300 px-3 py-1.5 rounded text-sm hover:bg-slate-50"
              >
                Move to another quote
              </button>
            )}
          </div>
        </section>
      )}

      {modal === 'close' && (
        <ConfirmModal
          title={`Close "${bpName}"?`}
          message="Closing will prevent new batch submissions against this billing project."
          confirmLabel="Close billing project"
          danger
          onConfirm={handleClose}
          onClose={() => setModal(null)}
        />
      )}
      {modal === 'reopen' && (
        <ConfirmModal
          title={`Reopen "${bpName}"?`}
          confirmLabel="Reopen billing project"
          onConfirm={handleReopen}
          onClose={() => setModal(null)}
        />
      )}
      {modal === 'move' && (
        <MoveQuoteModal
          basePath={basePath}
          bpName={bpName}
          currentQuoteName={bp.quote_name}
          onClose={() => setModal(null)}
          onMoved={() => void fetchData()}
        />
      )}

      {/* Event Log */}
      <section className="border rounded mb-6">
        <SectionHeader label="Event Log" />
        <EventLog events={events} columns={BP_EVENT_COLUMNS} />
      </section>
    </div>
  );
}
