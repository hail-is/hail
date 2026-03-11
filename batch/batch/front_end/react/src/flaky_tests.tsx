import { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { PieChart, Pie, Cell, Tooltip, Legend, BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts';

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
  instances: RetriedTest[];
}

const COLORS = ['#0ea5e9', '#f97316', '#a855f7', '#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#ec4899', '#14b8a6', '#84cc16'];

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

function ChevronRight({ className }: { className?: string }) {
  return (
    <svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
      <path fillRule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clipRule="evenodd" />
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
      existing.instances.push(row);
    } else {
      byJob.set(key, {
        job_name: key,
        retry_count: 1,
        distinct_prs: new Set([row.pr_number]),
        last_retried_at: row.retried_at,
        instances: [row],
      });
    }
  }
  return Array.from(byJob.values()).sort((a, b) => b.retry_count - a.retry_count);
}

function RetryCharts({ tests }: { tests: AggregatedTest[] }) {
  const total = tests.reduce((s, t) => s + t.retry_count, 0);

  const stateCounts = tests.flatMap((t) => t.instances).reduce<Record<string, number>>((acc, r) => {
    acc[r.state] = (acc[r.state] ?? 0) + 1;
    return acc;
  }, {});
  const STATE_COLORS: Record<string, string> = { Failed: '#ef4444', Error: '#f97316' };
  const statePieData = Object.entries(stateCounts).map(([name, value]) => ({ name, value }));

  // tests is already sorted descending; recharts puts first item at top, so longest bar is first = top
  const mainBarJobs = tests.filter((t) => t.retry_count / total >= 0.1);
  const otherBarCount = tests.filter((t) => t.retry_count / total < 0.1).reduce((s, t) => s + t.retry_count, 0);
  const barData = mainBarJobs.map((t) => ({ name: t.job_name, value: t.retry_count }));
  if (otherBarCount > 0) barData.push({ name: 'Other', value: otherBarCount });

  return (
    <div className="flex flex-wrap gap-8 mb-8 items-start">
      <div className="flex-1" style={{ minWidth: 380 }}>
        <p className="text-xs font-medium text-slate-500 mb-1">Retries per job</p>
        <ResponsiveContainer width="100%" height={Math.max(260, barData.length * 28)}>
          <BarChart data={barData} layout="vertical" margin={{ right: 24 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 11 }} allowDecimals={false} />
            <YAxis type="category" dataKey="name" width={200} tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v) => [String(v ?? ''), 'retries']} />
            <Bar dataKey="value" radius={[0, 3, 3, 0]}>
              {barData.map((entry, i) => (
                <Cell key={i} fill={entry.name === 'Other' ? '#94a3b8' : '#0ea5e9'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div>
        <p className="text-xs font-medium text-slate-500 mb-1">By failure type</p>
        <PieChart width={300} height={300}>
          <Pie data={statePieData} dataKey="value" cx="50%" cy="50%" outerRadius={95}>
            {statePieData.map((entry, i) => (
              <Cell key={i} fill={STATE_COLORS[entry.name] ?? COLORS[i % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip formatter={(v) => [String(v ?? ''), 'retries']} />
          <Legend iconSize={10} wrapperStyle={{ fontSize: '11px' }} />
        </PieChart>
      </div>
    </div>
  );
}

function InstanceRows({ instances, batchBaseUrl }: { instances: RetriedTest[]; batchBaseUrl: string }) {
  return (
    <>
      {instances.map((r) => (
        <tr key={r.id} className="bg-slate-50 border-b border-slate-100 text-xs text-slate-600">
          <td className="py-1.5 pl-8 pr-4" colSpan={2}>
            <a
              href={`${batchBaseUrl}/batches/${r.batch_id}/jobs/${r.job_id}`}
              className="text-sky-700 hover:underline font-mono"
              target="_blank"
              rel="noopener noreferrer"
            >
              batch {r.batch_id} / job {r.job_id}
            </a>
            <span className="ml-2 text-slate-400">PR #{r.pr_number}</span>
            <span className="ml-2 text-slate-400">{r.source_branch}</span>
          </td>
          <td className="py-1.5 pr-4 text-right">
            <span className={r.state === 'Success' ? 'text-green-600' : 'text-red-500'}>{r.state}</span>
            {r.exit_code !== null && <span className="ml-1 text-slate-400">(exit {r.exit_code})</span>}
          </td>
          <td className="py-1.5 pr-4 text-right text-slate-400">—</td>
          <td className="py-1.5 text-slate-400">{new Date(r.retried_at).toLocaleString()}</td>
        </tr>
      ))}
    </>
  );
}

function FlakyTests({ basePath, batchBaseUrl }: { basePath: string; batchBaseUrl: string }) {
  const [tests, setTests] = useState<AggregatedTest[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

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

  function toggleExpanded(jobName: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(jobName)) {
        next.delete(jobName);
      } else {
        next.add(jobName);
      }
      return next;
    });
  }

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
      <h2 className="text-base font-semibold text-slate-700 mb-2">Leaderboard</h2>
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
              <>
                <tr
                  key={t.job_name}
                  className="border-b border-slate-100 hover:bg-slate-50 cursor-pointer select-none"
                  onClick={() => toggleExpanded(t.job_name)}
                >
                  <td className="py-2 pr-4 text-slate-400">{i + 1}</td>
                  <td className="py-2 pr-4 font-mono">
                    <span className="inline-flex items-center gap-1">
                      <ChevronRight className={`h-3.5 w-3.5 text-slate-400 transition-transform ${expanded.has(t.job_name) ? 'rotate-90' : ''}`} />
                      {t.job_name}
                    </span>
                  </td>
                  <td className="py-2 pr-4 text-right font-semibold text-sky-700">{t.retry_count}</td>
                  <td className="py-2 pr-4 text-right text-slate-600">{t.distinct_prs.size}</td>
                  <td className="py-2 text-slate-500">{new Date(t.last_retried_at).toLocaleString()}</td>
                </tr>
                {expanded.has(t.job_name) && (
                  <InstanceRows instances={t.instances} batchBaseUrl={batchBaseUrl} />
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>
      {hasMore && (
        <p className="mt-3 text-sm text-amber-600">
          Results truncated — showing most recent 500 retries only. Aggregations may be incomplete.
        </p>
      )}
      <h2 className="text-base font-semibold text-slate-700 mt-6 mb-2">Charts</h2>
      <RetryCharts tests={tests} />
    </div>
  );
}

const container = document.getElementById('flaky-tests-root');
if (container) {
  const basePath = container.dataset.basePath ?? '';
  const batchBaseUrl = container.dataset.batchBaseUrl ?? '';
  createRoot(container).render(<FlakyTests basePath={basePath} batchBaseUrl={batchBaseUrl} />);
}
