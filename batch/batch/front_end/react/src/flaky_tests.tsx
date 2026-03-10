import { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';

interface FlakyTest {
  job_name: string;
  retry_count: number;
  distinct_prs: number;
  last_retried_at: string | null;
}

function Spinner() {
  return (
    <svg
      className="animate-spin h-5 w-5 text-sky-600"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      ></path>
    </svg>
  );
}

function FlakyTests({ basePath }: { basePath: string }) {
  const [tests, setTests] = useState<FlakyTest[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${basePath}/api/v1alpha/flaky_tests`)
      .then((resp) => {
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return resp.json() as Promise<FlakyTest[]>;
      })
      .then((data) => setTests(data))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [basePath]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 mt-4 text-slate-500">
        <Spinner />
        Loading&hellip;
      </div>
    );
  }

  if (error) {
    return <div className="mt-4 text-red-600">Error: {error}</div>;
  }

  if (tests.length === 0) {
    return <div className="mt-4 text-slate-500">No retried tests in the last 14 days.</div>;
  }

  return (
    <div className="mt-4 overflow-x-auto">
      <table className="min-w-full text-sm border-collapse">
        <thead>
          <tr className="border-b border-slate-200 text-left text-slate-600">
            <th className="py-2 pr-4 font-medium">#</th>
            <th className="py-2 pr-4 font-medium">Job Name</th>
            <th className="py-2 pr-4 font-medium text-right">Retries (14 d)</th>
            <th className="py-2 pr-4 font-medium text-right">Distinct PRs</th>
            <th className="py-2 font-medium">Last Retried</th>
          </tr>
        </thead>
        <tbody>
          {tests.map((t, i) => (
            <tr key={t.job_name} className="border-b border-slate-100 hover:bg-slate-50">
              <td className="py-2 pr-4 text-slate-400">{i + 1}</td>
              <td className="py-2 pr-4 font-mono">{t.job_name}</td>
              <td className="py-2 pr-4 text-right font-semibold text-sky-700">{t.retry_count}</td>
              <td className="py-2 pr-4 text-right text-slate-600">{t.distinct_prs}</td>
              <td className="py-2 text-slate-500">
                {t.last_retried_at ? new Date(t.last_retried_at).toLocaleString() : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const container = document.getElementById('flaky-tests-root');
if (container) {
  const basePath = container.dataset.basePath ?? '';
  createRoot(container).render(<FlakyTests basePath={basePath} />);
}
