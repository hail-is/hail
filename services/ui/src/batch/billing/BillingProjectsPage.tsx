import { useState, useEffect, useCallback } from 'react';
import type { BillingProject } from './api';
import { fetchJson } from './api';
import { ErrorBanner, BillingProjectsTable, CreateBpModal } from './shared';

interface Props {
  basePath: string;
  isGlobalBm: boolean;
  canCreateBp: boolean;
  canCreateQuotes: boolean;
}


export function BillingProjectsPage({ basePath, isGlobalBm, canCreateBp, canCreateQuotes }: Props) {
  const [bps, setBps] = useState<BillingProject[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const data = await fetchJson<BillingProject[]>(`${basePath}/api/v1alpha/billing_projects`);
      setBps(data);
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [basePath]);

  useEffect(() => { void fetchData(); }, [fetchData]);

  const openBps = bps?.filter((bp) => bp.status === 'open') ?? [];
  const closedBps = bps?.filter((bp) => bp.status === 'closed') ?? [];

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2 text-2xl font-light">
          <span className="text-slate-400">Billing</span>
          <span className="text-slate-300">›</span>
          <span>Billing Projects</span>
        </div>
        <div className="flex items-center gap-3">
          {canCreateBp && (
            <button
              onClick={() => setShowCreate(true)}
              className="bg-blue-600 text-white px-4 py-2 rounded text-sm hover:bg-blue-700"
            >
              New Billing Project
            </button>
          )}
        </div>
      </div>

      {error && <ErrorBanner message={error} />}

      {loading && (
        <div className="flex items-center justify-center mt-12">
          <span className="text-3xl font-light text-sky-600">Loading&hellip;</span>
        </div>
      )}

      {!loading && !error && (
        <>
          <BillingProjectsTable
            label="Open Billing Projects"
            bps={openBps}
            basePath={basePath}
            emptyMessage="No open billing projects found."
            showQuote
          />
          <BillingProjectsTable
            label="Closed Billing Projects"
            bps={closedBps}
            basePath={basePath}
            emptyMessage="No closed billing projects."
            defaultOpen={false}
            showQuote
          />
        </>
      )}

      {showCreate && (
        <CreateBpModal
          basePath={basePath}
          onClose={() => setShowCreate(false)}
          onCreated={() => void fetchData()}
        />
      )}
    </div>
  );
}
