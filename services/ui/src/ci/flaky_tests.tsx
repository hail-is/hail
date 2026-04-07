import { useState, useEffect, useRef, Fragment } from 'react';
import { createRoot } from 'react-dom/client';
import { PieChart, Pie, Cell, Tooltip, Legend, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';

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
  distinct_builds: Set<number>;
  distinct_prs: Set<number>;
  last_retried_at: string;
  instances: RetriedTest[];
}

function retryHeatColor(ratio: number): string {
  if (ratio > 0.66) return 'rgb(239 68 68 / 0.25)';   // red
  if (ratio > 0.33) return 'rgb(249 115 22 / 0.25)';  // orange
  return 'rgb(253 224 71 / 0.25)';                      // yellow
}

function familyName(jobName: string): string {
  return jobName.replace(/_\d+$/, '');
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

function aggregate(rows: RetriedTest[], groupByFamily: boolean): AggregatedTest[] {
  const byJob = new Map<string, AggregatedTest>();
  for (const row of rows) {
    const rawName = row.job_name ?? '(unknown)';
    const key = groupByFamily ? familyName(rawName) : rawName;
    const existing = byJob.get(key);
    if (existing) {
      existing.retry_count += 1;
      existing.distinct_builds.add(row.batch_id);
      existing.distinct_prs.add(row.pr_number);
      if (row.retried_at > existing.last_retried_at) {
        existing.last_retried_at = row.retried_at;
      }
      existing.instances.push(row);
    } else {
      byJob.set(key, {
        job_name: key,
        retry_count: 1,
        distinct_builds: new Set([row.batch_id]),
        distinct_prs: new Set([row.pr_number]),
        last_retried_at: row.retried_at,
        instances: [row],
      });
    }
  }
  return Array.from(byJob.values()).sort((a, b) => b.retry_count - a.retry_count);
}


function RetryCharts({ tests, days }: { tests: AggregatedTest[]; days: number }) {
  const [barMetric, setBarMetric] = useState<'retries' | 'batches' | 'prs'>('batches');
  const barColor = barMetric === 'retries' ? '#a855f7' : barMetric === 'batches' ? '#0ea5e9' : '#10b981';

  const instances = tests.flatMap((t) => t.instances);

  const stateCounts = instances.reduce<Record<string, number>>((acc, r) => {
    acc[r.state] = (acc[r.state] ?? 0) + 1;
    return acc;
  }, {});
  const STATE_COLORS: Record<string, string> = { Failed: '#ef4444', Error: '#f97316' };
  const statePieData = Object.entries(stateCounts).map(([name, value]) => ({ name, value }));

  const retriedByCounts = instances.reduce<Record<string, number>>((acc, r) => {
    acc[r.retried_by] = (acc[r.retried_by] ?? 0) + 1;
    return acc;
  }, {});
  const retriedByPieData = Object.entries(retriedByCounts)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({ name, value }));

  function localDateKey(d: Date): string {
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  }
  const dayRetryCounts = new Map<string, number>();
  const dayBatchSets = new Map<string, Set<number>>();
  const dayPrSets = new Map<string, Set<number>>();
  for (const r of instances) {
    const day = localDateKey(new Date(r.retried_at));
    dayRetryCounts.set(day, (dayRetryCounts.get(day) ?? 0) + 1);
    const bs = dayBatchSets.get(day);
    if (bs) bs.add(r.batch_id); else dayBatchSets.set(day, new Set([r.batch_id]));
    const ps = dayPrSets.get(day);
    if (ps) ps.add(r.pr_number); else dayPrSets.set(day, new Set([r.pr_number]));
  }
  const barData = Array.from({ length: days }, (_, i) => {
    const d = new Date(Date.now() - (days - 1 - i) * 24 * 60 * 60 * 1000);
    const key = localDateKey(d);
    return {
      date: `${d.getMonth() + 1}/${d.getDate()}`,
      retries: dayRetryCounts.get(key) ?? 0,
      batches: dayBatchSets.get(key)?.size ?? 0,
      prs: dayPrSets.get(key)?.size ?? 0,
    };
  });

  const totalValue = barMetric === 'retries' ? instances.length
    : barMetric === 'batches' ? new Set(instances.map((r) => r.batch_id)).size
    : new Set(instances.map((r) => r.pr_number)).size;
  const totalLabel = barMetric === 'retries' ? 'Total job retries' : barMetric === 'batches' ? 'Total builds retried' : 'Total PRs affected';

  return (
    <div className="mb-8 flex flex-wrap gap-8 items-start">
      <div style={{ flexShrink: 0 }}>
        <p className="text-xs font-medium text-slate-500 mb-1">Failure types</p>
        <PieChart width={260} height={280}>
          <Pie data={statePieData} dataKey="value" cx="50%" cy="50%" outerRadius={95} isAnimationActive={false}>
            {statePieData.map((entry, i) => (
              <Cell key={i} fill={STATE_COLORS[entry.name] ?? COLORS[i % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip formatter={(v, name) => [String(v ?? ''), String(name ?? '')]} />
          <Legend iconSize={10} wrapperStyle={{ fontSize: '11px' }} />
        </PieChart>
      </div>
      <div style={{ flexShrink: 0 }}>
        <p className="text-xs font-medium text-slate-500 mb-1">Retried by</p>
        <PieChart width={260} height={280}>
          <Pie data={retriedByPieData} dataKey="value" cx="50%" cy="50%" outerRadius={95} isAnimationActive={false}>
            {retriedByPieData.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip formatter={(v, name) => [String(v ?? ''), String(name ?? '')]} />
          <Legend iconSize={10} wrapperStyle={{ fontSize: '11px' }} />
        </PieChart>
      </div>
      <div style={{ flexShrink: 0 }}>
        <div className="flex items-center gap-3 mb-1">
          <p className="text-xs font-medium text-slate-500">Retries per day</p>
          <div className="flex rounded overflow-hidden border border-slate-200 text-xs">
            {(['retries', 'batches', 'prs'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => { setBarMetric(m); }}
                className={`px-2 py-0.5 ${m === barMetric ? 'text-white' : 'bg-white text-slate-600 hover:bg-slate-50'}`}
                style={m === barMetric ? { backgroundColor: barColor } : undefined}
              >
                {m === 'retries' ? 'jobs' : m === 'batches' ? 'builds' : 'prs'}
              </button>
            ))}
          </div>
        </div>
        <BarChart width={460} height={280} data={barData} margin={{ top: 5, right: 10, left: 0, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
          <XAxis dataKey="date" tick={{ fontSize: 10 }} interval="preserveStartEnd" angle={-45} textAnchor="end" />
          <YAxis tick={{ fontSize: 10 }} allowDecimals={false} width={30} />
          <Tooltip formatter={(v) => [String(v), barMetric === 'retries' ? 'job retries' : barMetric === 'batches' ? 'builds affected' : 'PRs affected']} />
          <Bar dataKey={barMetric} fill={barColor} isAnimationActive={false} radius={[2, 2, 0, 0]} />
        </BarChart>
      </div>
      <div style={{ width: '260px', height: '295px', flexShrink: 0 }} className="flex flex-col">
        <p className="text-xs font-medium text-slate-500 mb-1">{totalLabel}</p>
        <div className="flex flex-1 items-center justify-center">
          <span className="text-8xl tracking-tight leading-none" style={{ color: barColor, fontFamily: "'Fredoka One', sans-serif" }}>
            {totalValue}
          </span>
        </div>
      </div>
    </div>
  );
}

function LeaderboardTable({ tests, maxCount, expanded, toggleExpanded, batchBaseUrl, hasMore }: {
  tests: AggregatedTest[];
  maxCount: number;
  expanded: Set<string>;
  toggleExpanded: (jobName: string) => void;
  batchBaseUrl: string;
  hasMore: boolean;
}) {
  return (
    <>
      <div className="overflow-x-auto rounded-lg border border-slate-200">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-sky-100 text-slate-600 text-xs font-semibold uppercase tracking-wider">
              <th className="px-4 py-2.5 text-left w-10">#</th>
              <th className="px-4 py-2.5 text-left">Job Name</th>
              <th className="px-4 py-2.5 text-right w-24"># Jobs</th>
              <th className="px-4 py-2.5 text-right w-24"># Builds</th>
              <th className="px-4 py-2.5 text-right w-24"># PRs</th>
              <th className="px-4 py-2.5 text-left w-40">Last Retried</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {tests.map((t, i) => (
              <Fragment key={t.job_name}>
                <tr className="relative hover:bg-sky-50 select-none">
                  <td className="px-4 py-2 text-slate-400">{i + 1}</td>
                  <td className="px-4 py-2 font-mono" style={{ background: `linear-gradient(to right, ${retryHeatColor(t.retry_count / maxCount)} ${(t.retry_count / maxCount) * 100}%, transparent ${(t.retry_count / maxCount) * 100}%)` }}>
                    <button
                      type="button"
                      className="inline-flex items-center gap-1 cursor-pointer bg-transparent border-none p-0 focus:outline-none focus-visible:underline after:absolute after:inset-0"
                      aria-expanded={expanded.has(t.job_name)}
                      onClick={() => { toggleExpanded(t.job_name); }}
                    >
                      <ChevronRight className={`h-5 w-5 text-slate-600 transition-transform ${expanded.has(t.job_name) ? 'rotate-90' : ''}`} />
                      {t.job_name}
                    </button>
                  </td>
                  <td className="px-4 py-2 text-right font-semibold" style={{ color: '#a855f7' }}>{t.retry_count}</td>
                  <td className="px-4 py-2 text-right font-semibold" style={{ color: '#0ea5e9' }}>{t.distinct_builds.size}</td>
                  <td className="px-4 py-2 text-right font-semibold" style={{ color: '#10b981' }}>{t.distinct_prs.size}</td>
                  <td className="px-4 py-2 text-slate-500">{new Date(t.last_retried_at).toLocaleString()}</td>
                </tr>
                {expanded.has(t.job_name) && (
                  <InstanceRows instances={t.instances} batchBaseUrl={batchBaseUrl} />
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
      {hasMore && (
        <p className="mt-3 text-sm text-amber-600">
          Results truncated — showing most recent 500 retries only. Aggregations may be incomplete.
        </p>
      )}
    </>
  );
}

function InstanceRows({ instances, batchBaseUrl }: { instances: RetriedTest[]; batchBaseUrl: string }) {
  return (
    <>
      {instances.map((r) => (
        <tr key={r.id} className="bg-slate-50 text-xs text-slate-600">
          <td className="py-1.5 pl-8 pr-4" colSpan={2}>
            <a
              href={`${batchBaseUrl}/batches/${r.batch_id}/jobs/${r.job_id}`}
              className="text-sky-700 hover:underline font-mono"
              target="_blank"
              rel="noopener noreferrer"
            >
              build {r.batch_id} / job {r.job_id}
            </a>
            <span className="ml-2 text-slate-400">PR #{r.pr_number}</span>
            <span className="ml-2 text-slate-400">{r.source_branch}</span>
          </td>
          <td className="py-1.5 pr-4 text-right">
            <span className={r.state === 'Success' ? 'text-green-600' : 'text-red-500'}>{r.state}</span>
            {r.exit_code !== null && <span className="ml-1 text-slate-400">(exit {r.exit_code})</span>}
          </td>
          <td className="py-1.5 pr-4 text-right text-slate-400">—</td>
          <td className="py-1.5 pr-4 text-right text-slate-400">—</td>
          <td className="py-1.5 pr-4 text-slate-400">{new Date(r.retried_at).toLocaleString()}</td>
        </tr>
      ))}
    </>
  );
}

const DAY_OPTIONS = [7, 14, 30, 90] as const;

function DaysSelector({ days, onChange }: { days: number; onChange: (newDays: number) => void }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-sm text-slate-500">History:</span>
      <div className="flex rounded-md overflow-hidden border border-slate-200 text-sm">
        {DAY_OPTIONS.map((d) => (
          <button
            key={d}
            type="button"
            onClick={() => { onChange(d); }}
            className={`px-2.5 py-0.5 ${d === days ? 'bg-sky-600 text-white' : 'bg-white text-slate-600 hover:bg-slate-50'}`}
          >
            {d}d
          </button>
        ))}
      </div>
    </div>
  );
}

function FlakyTests({ basePath, batchBaseUrl }: { basePath: string; batchBaseUrl: string }) {
  const [days, setDays] = useState(14);
  const [rawRows, setRawRows] = useState<RetriedTest[]>([]);
  const [groupByFamily, setGroupByFamily] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const hasLoadedRef = useRef(false);

  useEffect(() => {
    setError(null);
    if (!hasLoadedRef.current) {
      setLoading(true);
    }
    const after = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
    fetch(`${basePath}/api/v1alpha/retried_tests?after=${encodeURIComponent(after)}`)
      .then((resp) => {
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return resp.json() as Promise<ApiResponse>;
      })
      .then((data) => {
        setRawRows(data.rows);
        setHasMore(data.has_more);
        hasLoadedRef.current = true;
      })
      .catch((e: unknown) => { setError(e instanceof Error ? e.message : String(e)); })
      .finally(() => { setLoading(false); });
  }, [basePath, days]);

  const tests = aggregate(rawRows, groupByFamily);

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

  const controls = (
    <div className="mb-4">
      <DaysSelector days={days} onChange={(d) => { setDays(d); setExpanded(new Set()); }} />
    </div>
  );

  const groupByFamilyToggle = (
    <div className="flex items-center gap-2.5">
      <button
        type="button"
        role="switch"
        aria-checked={groupByFamily}
        onClick={() => { setGroupByFamily(v => !v); setExpanded(new Set()); }}
        className={`relative h-5 w-9 flex-shrink-0 rounded-full p-0 transition-colors duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 ${groupByFamily ? 'bg-sky-600' : 'bg-slate-300'}`}
      >
        <span
          className="absolute h-4 w-4 rounded-full bg-white shadow-sm"
          style={{ top: '2px', left: groupByFamily ? '18px' : '2px', transition: 'left 200ms ease-in-out' }}
        />
      </button>
      <button
        type="button"
        className="text-sm text-slate-600 bg-transparent border-none p-0 cursor-pointer select-none focus:outline-none focus-visible:underline"
        onClick={() => { setGroupByFamily(v => !v); setExpanded(new Set()); }}
      >
        Group test families
      </button>
    </div>
  );

  if (loading) {
    return (
      <div className="mt-4">
        {controls}
        <div className="flex items-center gap-2 text-slate-500">
          <Spinner />
          Loading&hellip;
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mt-4">
        {controls}
        <div className="text-red-600">Error: {error}</div>
      </div>
    );
  }

  if (tests.length === 0) {
    return (
      <div className="mt-4">
        {controls}
        <div className="text-slate-500">No retried tests in the last {days} days.</div>
      </div>
    );
  }

  const maxCount = tests[0]?.retry_count ?? 1;

  return (
    <div className="mt-4">
      {controls}
      <h2 className="text-base font-semibold text-slate-700 mb-2">Charts</h2>
      <RetryCharts tests={tests} days={days} />
      <div className="flex items-center gap-4 mb-2">
        <h2 className="text-base font-semibold text-slate-700">Leaderboard</h2>
        {groupByFamilyToggle}
      </div>
      <LeaderboardTable
        tests={tests}
        maxCount={maxCount}
        expanded={expanded}
        toggleExpanded={toggleExpanded}
        batchBaseUrl={batchBaseUrl}
        hasMore={hasMore}
      />
    </div>
  );
}

const container = document.getElementById('flaky-tests-root');
if (container) {
  const basePath = container.dataset.basePath ?? '';
  const batchBaseUrl = container.dataset.batchBaseUrl ?? '';
  createRoot(container).render(<FlakyTests basePath={basePath} batchBaseUrl={batchBaseUrl} />);
}
