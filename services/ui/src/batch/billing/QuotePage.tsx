import { useState, useEffect, useCallback } from 'react';
import type { Quote, BillingEvent } from './api';
import { fetchJson, apiCall } from './api';
import { fmtDollars, fmtTimestamp } from './fmt';
import { can } from './permissions';
import { SectionHeader, ErrorBanner, EditableRow, EventLog, BillingProjectsTable, QuoteBudgetBar, CreateBpModal, ConfirmModal } from './shared';

interface Props {
  basePath: string;
  quoteName: string;
}

const QUOTE_EVENT_COLUMNS = [
  { key: 'timestamp' as const, label: 'Time', render: (v: unknown) => <span className="whitespace-nowrap text-slate-500">{fmtTimestamp(v as number)}</span> },
  { key: 'actor' as const, label: 'Actor' },
  { key: 'action' as const, label: 'Action' },
  { key: 'target_user' as const, label: 'Target User' },
  { key: 'target_project' as const, label: 'Target Project' },
  { key: 'detail' as const, label: 'Detail' },
  { key: 'comment' as const, label: 'Comment', render: (v: unknown) => <span className="text-slate-500 italic">{String(v ?? '')}</span> },
];



export function QuotePage({ basePath, quoteName }: Props) {
  const [quote, setQuote] = useState<Quote | null>(null);
  const [events, setEvents] = useState<BillingEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [addMgrUser, setAddMgrUser] = useState('');
  const [addMgrRole, setAddMgrRole] = useState('manager');
  const [mgrError, setMgrError] = useState<string | null>(null);
  const [showCreateBp, setShowCreateBp] = useState(false);
  const [showCloseQuote, setShowCloseQuote] = useState(false);
  const [showReopenQuote, setShowReopenQuote] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const [q, ev] = await Promise.all([
        fetchJson<Quote>(`${basePath}/api/v1alpha/quotes/${encodeURIComponent(quoteName)}`),
        fetchJson<BillingEvent[]>(`${basePath}/api/v1alpha/quotes/${encodeURIComponent(quoteName)}/events`),
      ]);
      setQuote(q);
      setEvents(ev);
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [basePath, quoteName]);

  useEffect(() => { void fetchData(); }, [fetchData]);

  const patch = async (updates: object) => {
    await apiCall('PATCH', `${basePath}/api/v1alpha/quotes/${encodeURIComponent(quoteName)}`, updates);
    await fetchData();
  };

  const removeManager = async (user: string) => {
    setMgrError(null);
    try {
      await apiCall('DELETE', `${basePath}/api/v1alpha/quotes/${encodeURIComponent(quoteName)}/managers/${encodeURIComponent(user)}`);
      await fetchData();
    } catch (e) {
      setMgrError(String(e));
    }
  };

  const addManager = async () => {
    const u = addMgrUser.trim();
    if (!u) return;
    setMgrError(null);
    try {
      await apiCall('POST', `${basePath}/api/v1alpha/quotes/${encodeURIComponent(quoteName)}/managers`, {
        user: u, role: addMgrRole,
      });
      setAddMgrUser('');
      await fetchData();
    } catch (e) {
      setMgrError(String(e));
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
  if (!quote) return null;

  const billingRole = quote.billing_role;
  const canEdit = can(billingRole, 'edit_quote');
  const canAddManagers = can(billingRole, 'add_manager');
  const canManageManagers = can(billingRole, 'manage_managers');
  const canCreateBp = can(billingRole, 'create_bp');
  const canCloseQuote = can(billingRole, 'close_quote');

  const totalDistributed = quote.billing_projects.reduce((s, bp) => s + (bp.limit ?? 0), 0);
  const totalSpent = quote.billing_projects.reduce((s, bp) => s + bp.accrued_cost, 0);

  return (
    <div className="max-w-4xl">
      <div className="flex items-center gap-2 mb-6 text-2xl font-light">
        <span className="text-slate-400">Billing</span>
        <span className="text-slate-300">›</span>
        <a href={`${basePath}/billing/quotes`} className="text-slate-400 hover:text-slate-600">Quotes</a>
        <span className="text-slate-300">›</span>
        <span>{quoteName}</span>
        {quote.state === 'open' ? (
          <span className="text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded-full">open</span>
        ) : (
          <span className="text-xs bg-gray-200 text-gray-600 px-2 py-0.5 rounded-full italic">{quote.state}</span>
        )}
      </div>

      {/* Details */}
      <section className="border rounded mb-6">
        <SectionHeader label="Details" />
        <table className="w-full text-sm">
          <tbody>
            <EditableRow
              label="Quote Number"
              value={quote.quote_number ?? ''}
              canEdit={canEdit && quote.state === 'open'}
              onSave={(val) => patch({ quote_number: val || null })}
            />
            <EditableRow
              label="Cost Object"
              value={quote.cost_object}
              canEdit={canEdit && quote.state === 'open'}
              onSave={(val) => patch({ cost_object: val })}
            />
            <EditableRow
              label="PI Name"
              value={quote.pi_name ?? ''}
              canEdit={canEdit && quote.state === 'open'}
              onSave={(val) => patch({ pi_name: val })}
            />
            <EditableRow
              label="PM Designee"
              value={quote.pm_designee ?? ''}
              canEdit={canEdit && quote.state === 'open'}
              onSave={(val) => patch({ pm_designee: val })}
            />
            <EditableRow
              label="Description"
              value={quote.description ?? ''}
              canEdit={canEdit && quote.state === 'open'}
              onSave={(val) => patch({ description: val || null })}
            />
          </tbody>
        </table>
      </section>

      {/* Funding */}
      <section className="border rounded mb-6">
        <SectionHeader label="Funding" />
        <table className="w-full text-sm">
          <tbody>
            <EditableRow
              label="Authorized Amount"
              value={quote.authorized_amount !== null ? String(quote.authorized_amount) : ''}
              displayValue={fmtDollars(quote.authorized_amount)}
              canEdit={canEdit}
              inputType="number"
              prefix="$"
              placeholder="blank = unlimited"
              onSave={(val) => patch({ authorized_amount: val === '' ? 'unlimited' : parseFloat(val) })}
            />
            <tr className="border-b border-slate-100">
              <td className="py-2 pl-4 pr-8 text-slate-500 w-40 align-middle">Usage</td>
              <td className="py-2 pr-4 align-middle" colSpan={2}>
                <QuoteBudgetBar spent={totalSpent} allocated={totalDistributed} authorized={quote.authorized_amount} />
              </td>
            </tr>
          </tbody>
        </table>
      </section>

      {/* Open Billing Projects */}
      <BillingProjectsTable
        label="Open Billing Projects"
        bps={quote.billing_projects.filter((bp) => bp.status === 'open')}
        basePath={basePath}
        emptyMessage="No open billing projects under this quote."
      />

      {/* Managers */}
      <section className="border rounded mb-6">
        <SectionHeader label="Managers" />
        <div className="p-4">
          <table className="w-full text-sm mb-3">
            <tbody>
              {quote.managers.map((m) => (
                <tr key={m.user} className="hover:bg-slate-50">
                  <td className="py-1 pr-4">{m.user}</td>
                  <td className="py-1 pr-4 text-slate-500">{m.role}</td>
                  <td className="py-1 text-right">
                    {canManageManagers && (
                      <button
                        onClick={() => void removeManager(m.user)}
                        className="text-red-400 hover:text-red-600"
                      >
                        <span className="material-symbols-outlined text-base">close</span>
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {mgrError && <div className="text-red-600 text-xs mb-2">{mgrError}</div>}
          {canAddManagers && (
            <div className="flex items-center gap-2">
              <input
                type="text" value={addMgrUser}
                onChange={(e) => setAddMgrUser(e.target.value)}
                placeholder="username" spellCheck={false}
                autoComplete="off"
                data-lpignore="true"
                data-1p-ignore
                className="border rounded px-2 py-1 text-sm w-48"
                onKeyDown={(e) => { if (e.key === 'Enter') void addManager(); }}
              />
              <select
                value={addMgrRole}
                onChange={(e) => setAddMgrRole(e.target.value)}
                className="border rounded px-2 py-1 text-sm"
              >
                <option value="manager">manager</option>
                <option value="owner">owner</option>
              </select>
              <span
                className="material-symbols-outlined text-slate-400 hover:text-slate-600 cursor-default text-base"
                title="Managers can edit quote details and manage billing projects. Owners can additionally add and remove managers."
              >
                info
              </span>
              <button
                onClick={() => void addManager()}
                className="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700"
              >
                Add
              </button>
            </div>
          )}
        </div>
      </section>

      {/* Actions */}
      {(canCreateBp || canCloseQuote) && (
        <section className="border rounded mb-6">
          <SectionHeader label="Actions" />
          <div className="p-4 flex flex-wrap gap-2">
            {canCreateBp && quote.state === 'open' && (
              <button
                onClick={() => setShowCreateBp(true)}
                className="bg-blue-600 text-white px-3 py-1.5 rounded text-sm hover:bg-blue-700"
              >
                Add Billing Project
              </button>
            )}
            {canCloseQuote && quote.state === 'open' && (
              <button
                onClick={() => setShowCloseQuote(true)}
                className="bg-red-600 text-white px-3 py-1.5 rounded text-sm hover:bg-red-700"
              >
                Close quote
              </button>
            )}
            {canCloseQuote && quote.state === 'closed' && (
              <button
                onClick={() => setShowReopenQuote(true)}
                className="bg-green-600 text-white px-3 py-1.5 rounded text-sm hover:bg-green-700"
              >
                Reopen quote
              </button>
            )}
          </div>
        </section>
      )}

      {/* Event Log */}
      <section className="border rounded mb-6">
        <SectionHeader label="Event Log" />
        <EventLog events={events} columns={QUOTE_EVENT_COLUMNS} />
      </section>

      {/* Closed Billing Projects */}
      <BillingProjectsTable
        label="Closed Billing Projects"
        bps={quote.billing_projects.filter((bp) => bp.status !== 'open')}
        basePath={basePath}
        emptyMessage="No closed billing projects under this quote."
        defaultOpen={false}
      />

      {showCreateBp && (
        <CreateBpModal
          basePath={basePath}
          fixedQuoteName={quoteName}
          onClose={() => setShowCreateBp(false)}
          onCreated={() => void fetchData()}
        />
      )}

      {showCloseQuote && (
        <ConfirmModal
          title={`Close "${quoteName}"?`}
          message="Closing will prevent new billing projects from being created under this quote."
          confirmLabel="Close quote"
          danger
          onConfirm={async (comment) => {
            await apiCall('POST', `${basePath}/api/v1alpha/quotes/${encodeURIComponent(quoteName)}/close`, { comment: comment || undefined });
            setShowCloseQuote(false);
            await fetchData();
          }}
          onClose={() => setShowCloseQuote(false)}
        />
      )}

      {showReopenQuote && (
        <ConfirmModal
          title={`Reopen "${quoteName}"?`}
          message="Reopening will allow new billing projects to be created under this quote."
          confirmLabel="Reopen quote"
          onConfirm={async (comment) => {
            await apiCall('POST', `${basePath}/api/v1alpha/quotes/${encodeURIComponent(quoteName)}/reopen`, { comment: comment || undefined });
            setShowReopenQuote(false);
            await fetchData();
          }}
          onClose={() => setShowReopenQuote(false)}
        />
      )}
    </div>
  );
}
