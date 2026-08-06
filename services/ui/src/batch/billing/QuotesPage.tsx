import { useState, useEffect, useCallback } from 'react';
import type { Quote } from './api';
import { fetchJson, apiCall } from './api';
import { ErrorBanner, QuotesTable } from './shared';

interface Props {
  basePath: string;
  canCreate: boolean;
}

function CreateQuoteModal({
  basePath,
  onClose,
  onCreated,
}: {
  basePath: string;
  onClose: () => void;
  onCreated: () => void;
}) {
  const [name, setName] = useState('');
  const [quoteNumber, setQuoteNumber] = useState('');
  const [costObject, setCostObject] = useState('');
  const [authorizedAmount, setAuthorizedAmount] = useState('');
  const [piName, setPiName] = useState('');
  const [pmDesignee, setPmDesignee] = useState('');
  const [description, setDescription] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const handleSubmit = async () => {
    if (!name.trim()) { setError('Name is required.'); return; }
    if (!costObject.trim()) { setError('Cost Object is required.'); return; }
    setSaving(true);
    setError(null);
    try {
      await apiCall('POST', `${basePath}/api/v1alpha/quotes/${encodeURIComponent(name)}`, {
        quote_number: quoteNumber || null,
        cost_object: costObject,
        authorized_amount: authorizedAmount === '' ? 'unlimited' : parseFloat(authorizedAmount),
        pi_name: piName || null,
        pm_designee: pmDesignee || null,
        description: description || null,
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
        <h2 className="text-xl font-light mb-4">New Quote</h2>
        <div className="space-y-3 text-sm">
          <div>
            <label className="block mb-1 text-slate-600">Name <span className="text-red-500">*</span></label>
            <input
              type="text" value={name} onChange={(e) => setName(e.target.value)}
              className="border rounded px-2 py-1 w-full" spellCheck={false}
            />
          </div>
          <div>
            <label className="block mb-1 text-slate-600">Quote Number</label>
            <input
              type="text" value={quoteNumber} onChange={(e) => setQuoteNumber(e.target.value)}
              className="border rounded px-2 py-1 w-full" spellCheck={false} placeholder="e.g. Q-2026-001"
            />
          </div>
          <div>
            <label className="block mb-1 text-slate-600">Cost Object <span className="text-red-500">*</span></label>
            <input
              type="text" value={costObject} onChange={(e) => setCostObject(e.target.value)}
              className="border rounded px-2 py-1 w-full" spellCheck={false}
            />
          </div>
          <div>
            <label className="block mb-1 text-slate-600">Authorized Amount (dollars, or leave blank for unlimited)</label>
            <input
              type="number" min="0" step="0.01" value={authorizedAmount}
              onChange={(e) => setAuthorizedAmount(e.target.value)}
              className="border rounded px-2 py-1 w-full" placeholder="e.g. 10000"
            />
          </div>
          <div>
            <label className="block mb-1 text-slate-600">PI Name</label>
            <input type="text" value={piName} onChange={(e) => setPiName(e.target.value)}
              className="border rounded px-2 py-1 w-full" />
          </div>
          <div>
            <label className="block mb-1 text-slate-600">PM Designee</label>
            <input type="text" value={pmDesignee} onChange={(e) => setPmDesignee(e.target.value)}
              className="border rounded px-2 py-1 w-full" />
          </div>
          <div>
            <label className="block mb-1 text-slate-600">Description</label>
            <input type="text" value={description} onChange={(e) => setDescription(e.target.value)}
              className="border rounded px-2 py-1 w-full" />
          </div>
        </div>
        {error && <div className="text-red-600 text-xs mt-2">{error}</div>}
        <div className="flex justify-end gap-2 mt-4">
          <button
            onClick={onClose}
            className="border border-gray-300 px-3 py-1.5 rounded text-sm hover:bg-slate-50"
          >
            Cancel
          </button>
          <button
            onClick={() => void handleSubmit()}
            disabled={saving}
            className="bg-blue-600 text-white px-4 py-1.5 rounded text-sm hover:bg-blue-700 disabled:opacity-50"
          >
            Create
          </button>
        </div>
      </div>
    </div>
  );
}


export function QuotesPage({ basePath, canCreate }: Props) {
  const [quotes, setQuotes] = useState<Quote[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const list = await fetchJson<Quote[]>(`${basePath}/api/v1alpha/quotes`);
      const details = await Promise.all(
        list.map((q) => fetchJson<Quote>(`${basePath}/api/v1alpha/quotes/${encodeURIComponent(q.name)}`))
      );
      setQuotes(details);
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [basePath]);

  useEffect(() => { void fetchData(); }, [fetchData]);

  const openQuotes = quotes?.filter((q) => q.state === 'open') ?? [];
  const closedQuotes = quotes?.filter((q) => q.state === 'closed') ?? [];

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2 text-2xl font-light">
          <span className="text-slate-400">Billing</span>
          <span className="text-slate-300">›</span>
          <span>Quotes</span>
        </div>
        {canCreate && (
          <button
            onClick={() => setShowCreate(true)}
            className="bg-blue-600 text-white px-4 py-2 rounded text-sm hover:bg-blue-700"
          >
            New Quote
          </button>
        )}
      </div>

      {error && <ErrorBanner message={error} />}

      {loading && (
        <div className="flex items-center justify-center mt-12">
          <span className="text-3xl font-light text-sky-600">Loading&hellip;</span>
        </div>
      )}

      {!loading && !error && (
        <>
          <QuotesTable
            label="Open Quotes"
            quotes={openQuotes}
            basePath={basePath}
            emptyMessage="You are not a manager of any open quotes."
          />
          <QuotesTable
            label="Closed Quotes"
            quotes={closedQuotes}
            basePath={basePath}
            emptyMessage="No closed quotes."
            defaultOpen={false}
          />
        </>
      )}

      {showCreate && (
        <CreateQuoteModal
          basePath={basePath}
          onClose={() => setShowCreate(false)}
          onCreated={() => void fetchData()}
        />
      )}
    </div>
  );
}

