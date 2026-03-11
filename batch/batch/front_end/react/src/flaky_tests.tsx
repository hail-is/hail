import { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';

interface RetriedTest {
  id: number;
  batch_id: number;
  job_id: number;
  job_name: string | null;
  state: string;
  exit_code: number | null;
  pr_number: number;
  target_branch: string;
  source_branch: string;
  source_sha: string;
  retried_by: string;
  retried_at: string;
}

interface ApiResponse {
  rows: RetriedTest[];
  cursor: number | null;
  has_more: boolean;
}

interface AggregatedTest {
  job_name: string;
  retry_count: number;
  distinct_prs: Set<number>;
  last_retried_at: string;
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

function aggregate(rows: RetriedTest[]): AggregatedTest[] {
  const byJob = new Map<string, AggregatedTest>();
  for (const row of rows) {
    const key = row.job_name ?? '(unknown)';
    const existing = byJob.get(key);
    if (existing) {
      existing.retry_count += 1;
      existing.distinct_prs.add(row.pr_number);
      if (row.retried_at > existing.last_retried_at) {
        existing.last_retried_at = row.retried_at;
      }
    } else {
      byJob.set(key, {
        job_name: key,
        retry_count: 1,
        distinct_prs: new Set([row.pr_number]),
        last_retried_at: row.retried_at,
      });
    }
  }
  return Array.from(byJob.values()).sort((a, b) => b.retry_count - a.retry_count);
}

function FlakyTests({ basePath }: { basePath: string }) {
  const [tests, setTests] = useState<AggregatedTest[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);

  useEffect(() => {
    fetch(`${basePath}/api/v1alpha/retried_tests`)
      .then((resp) => {
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return resp.json() as Promise<ApiResponse>;
      })
      .then((data) => {
        setTests(aggregate(data.rows));
        setHasMore(data.has_more);
      })
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
    <div className="mt-4">
      <div className="overflow-x-auto">
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
                <td className="py-2 pr-4 text-right text-slate-600">{t.distinct_prs.size}</td>
                <td className="py-2 text-slate-500">{new Date(t.last_retried_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {hasMore && (
        <p className="mt-3 text-sm text-amber-600">
          Results truncated — showing most recent 500 retries only. Aggregations may be incomplete.
        </p>
      )}
    </div>
  );
}

const container = document.getElementById('flaky-tests-root');
if (container) {
  const basePath = container.dataset.basePath ?? '';
  createRoot(container).render(<FlakyTests basePath={basePath} />);
}
