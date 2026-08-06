import { useState, useRef, useEffect } from 'react';
import { fetchJson } from './api';
import { fmtCost } from './fmt';
import { ErrorBanner } from './shared';

interface BillingRecord {
  billing_project: string;
  user: string;
  quote_name: string;
  total_spent: number;
}


type Tab = 'by-project' | 'by-user' | 'by-bp-user' | 'by-quote' | 'by-quote-bp';
type Mode = 'billing-projects' | 'quotes' | 'just-me';

interface Props {
  basePath: string;
  isGlobalBm: boolean;
  username: string;
  initialStart: string;
  initialEnd: string;
}

function csvEscape(val: string): string {
  if (val.includes(',') || val.includes('"') || val.includes('\n')) {
    return '"' + val.replace(/"/g, '""') + '"';
  }
  return val;
}

function toCsv(rows: string[][], columns: string[]): string {
  const lines = [columns.join(',')];
  for (const row of rows) {
    lines.push(row.map(csvEscape).join(','));
  }
  return lines.join('\n');
}

function mmddyyyyToIso(s: string): string {
  const [mm, dd, yyyy] = s.split('/');
  return `${yyyy}-${mm.padStart(2, '0')}-${dd.padStart(2, '0')}`;
}

function todayIso(): string {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
}

function firstOfMonthIso(): string {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-01`;
}

function firstOfMonthMmDdYyyy(): string {
  const now = new Date();
  return `${String(now.getMonth() + 1).padStart(2, '0')}/01/${now.getFullYear()}`;
}

function buildCsv(
  records: BillingRecord[],
  tab: Tab,
  appliedStart: string,
  appliedEnd: string,
): { csv: string; filename: string } {
  const startIso = appliedStart ? mmddyyyyToIso(appliedStart) : firstOfMonthIso();
  const endIso = appliedEnd ? mmddyyyyToIso(appliedEnd) : todayIso();

  let csv: string;
  let label: string;

  if (tab === 'by-project') {
    const acc = new Map<string, number>();
    for (const r of records) acc.set(r.billing_project, (acc.get(r.billing_project) ?? 0) + r.total_spent);
    const rows = [...acc.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([bp, cost]) => [bp, String(cost)]);
    csv = toCsv(rows, ['billing_project', 'total_spent']);
    label = 'by billing project';
  } else if (tab === 'by-user') {
    const acc = new Map<string, number>();
    for (const r of records) acc.set(r.user, (acc.get(r.user) ?? 0) + r.total_spent);
    const rows = [...acc.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([u, cost]) => [u, String(cost)]);
    csv = toCsv(rows, ['user', 'total_spent']);
    label = 'by user';
  } else if (tab === 'by-quote') {
    const acc = new Map<string, number>();
    for (const r of records) acc.set(r.quote_name, (acc.get(r.quote_name) ?? 0) + r.total_spent);
    const rows = [...acc.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([q, cost]) => [q, String(cost)]);
    csv = toCsv(rows, ['quote_name', 'total_spent']);
    label = 'by quote';
  } else if (tab === 'by-quote-bp') {
    const rows = [...records]
      .sort((a, b) => a.quote_name.localeCompare(b.quote_name) || a.billing_project.localeCompare(b.billing_project))
      .map((r) => [r.quote_name, r.billing_project, String(r.total_spent)]);
    csv = toCsv(rows, ['quote_name', 'billing_project', 'total_spent']);
    label = 'by quote and billing project';
  } else {
    const rows = [...records]
      .sort((a, b) => a.billing_project.localeCompare(b.billing_project) || a.user.localeCompare(b.user))
      .map((r) => [r.billing_project, r.user, String(r.total_spent)]);
    csv = toCsv(rows, ['billing_project', 'user', 'total_spent']);
    label = 'by billing project and user';
  }

  return { csv, filename: `Hail billing export ${startIso} to ${endIso} ${label}.csv` };
}

function TabButton({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`px-4 pt-4 pb-2 text-lg hover:opacity-100 hover:cursor-pointer border-black ${active ? 'border-b opacity-100' : 'opacity-50'}`}
    >
      {label}
    </button>
  );
}

function ClipboardIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
      <polyline points="7 10 12 15 17 10"></polyline>
      <line x1="12" y1="15" x2="12" y2="3"></line>
    </svg>
  );
}

function sortVal(v: string): string | number {
  if (v.startsWith('<$')) return Number.EPSILON;
  return v.startsWith('$') ? parseFloat(v.replace(/[$,]/g, '')) || 0 : v;
}

function cmpVals(a: string | number, b: string | number): number {
  return typeof a === 'number' ? a - (b as number) : (a as string).localeCompare(b as string);
}

function SortIndicator({ active, dir }: { active: boolean; dir: 'asc' | 'desc' }) {
  return (
    <span className="material-symbols-outlined text-sm text-slate-400">
      {active ? (dir === 'asc' ? 'arrow_upward' : 'arrow_downward') : 'unfold_more'}
    </span>
  );
}

function SummaryTable({ rows, columns }: { rows: [string, string][]; columns: [string, string] }) {
  const [sortCol, setSortCol] = useState(0);
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  const handleSort = (col: number) => {
    if (sortCol === col) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    else { setSortCol(col); setSortDir(rows.length > 0 && typeof sortVal(rows[0][col]) === 'number' ? 'desc' : 'asc'); }
  };

  const sorted = [...rows].sort((a, b) => {
    const cmp = cmpVals(sortVal(a[sortCol]), sortVal(b[sortCol]));
    return sortDir === 'asc' ? cmp : -cmp;
  });

  return (
    <table className="w-full overflow-auto">
      <thead>
        <tr className="border-b bg-slate-50">
          {columns.map((col, i) => (
            <th key={col} onClick={() => handleSort(i)} className="text-left p-3 font-medium cursor-pointer select-none hover:bg-slate-100">
              <div className="flex items-center gap-1">{col}<SortIndicator active={sortCol === i} dir={sortDir} /></div>
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {sorted.map(([a, b]) => (
          <tr key={a} className="border-y">
            <td className="p-2">{a}</td>
            <td className="p-2">{b}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function TwoColumnTable({ rows, columns }: { rows: [string, string, string][]; columns: [string, string, string] }) {
  const [sortCol, setSortCol] = useState(0);
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  const handleSort = (col: number) => {
    if (sortCol === col) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    else { setSortCol(col); setSortDir(rows.length > 0 && typeof sortVal(rows[0][col]) === 'number' ? 'desc' : 'asc'); }
  };

  const sorted = [...rows].sort((a, b) => {
    const cmp = cmpVals(sortVal(a[sortCol]), sortVal(b[sortCol]));
    return sortDir === 'asc' ? cmp : -cmp;
  });

  return (
    <table className="w-full overflow-auto">
      <thead>
        <tr className="border-b bg-slate-50">
          {columns.map((col, i) => (
            <th key={col} onClick={() => handleSort(i)} className="text-left p-3 font-medium cursor-pointer select-none hover:bg-slate-100">
              <div className="flex items-center gap-1">{col}<SortIndicator active={sortCol === i} dir={sortDir} /></div>
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {sorted.map(([a, b, c], i) => (
          <tr key={i} className="border-y">
            <td className="p-2">{a}</td>
            <td className="p-2">{b}</td>
            <td className="p-2">{c}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function BillingPage({ basePath, isGlobalBm, username, initialStart, initialEnd }: Props) {
  const initialStartVal = initialStart || firstOfMonthMmDdYyyy();
  const [start, setStart] = useState(initialStartVal);
  const [end, setEnd] = useState(initialEnd);
  const [appliedStart, setAppliedStart] = useState(initialStartVal);
  const [appliedEnd, setAppliedEnd] = useState(initialEnd);
  const [records, setRecords] = useState<BillingRecord[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showQuotes, setShowQuotes] = useState(isGlobalBm);
  const [mode, setMode] = useState<Mode>('billing-projects');
  const [tab, setTab] = useState<Tab>('by-project');
  const [exportStatus, setExportStatus] = useState('');
  const exportStatusTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [managedQuoteNames, setManagedQuoteNames] = useState<Set<string> | null>(null);
  const [bpCount, setBpCount] = useState<number | null>(null);

  const isDirty = start !== appliedStart || end !== appliedEnd;

  const fetchData = async (startVal: string, endVal: string) => {
    setLoading(true);
    setError(null);
    setRecords(null);
    const params = new URLSearchParams({ start: startVal });
    if (endVal) params.set('end', endVal);
    try {
      const data = await fetchJson<BillingRecord[]>(`${basePath}/api/v1alpha/billing?${params}`);
      setRecords(data);
      setAppliedStart(startVal);
      setAppliedEnd(endVal);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchData(start, end);
    fetchJson<unknown[]>(`${basePath}/api/v1alpha/billing_projects`).then((bps) => setBpCount(bps.length)).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    fetchJson<Array<{ name: string }>>(`${basePath}/api/v1alpha/quotes`).then((quotes) => {
      const names = new Set(quotes.map((q) => q.name));
      setManagedQuoteNames(names);
      if (names.size > 0) {
        setShowQuotes(true);
        setMode('quotes');
        setTab('by-quote');
      }
    }).catch(() => {});
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleApply = async (e: React.FormEvent) => {
    e.preventDefault();
    await fetchData(start, end);
  };

  const handleUndo = () => {
    setStart(appliedStart);
    setEnd(appliedEnd);
  };

  const handleModeChange = (newMode: Mode) => {
    setMode(newMode);
    if (newMode === 'billing-projects') {
      setTab('by-project');
    } else if (newMode === 'quotes') {
      setTab('by-quote');
    } else {
      setTab('by-bp-user');
    }
  };

  const quoteRecords = records && managedQuoteNames
    ? records.filter((r) => managedQuoteNames.has(r.quote_name))
    : records;

  const justMeRecords = records ? records.filter((r) => r.user === username) : null;

  const activeRecords = mode === 'quotes' ? quoteRecords : mode === 'just-me' ? justMeRecords : records;
  const totalCost = activeRecords ? fmtCost(activeRecords.reduce((s, r) => s + r.total_spent, 0)) : null;

  const byProject: [string, string][] = records
    ? (() => {
        const acc = new Map<string, number>();
        for (const r of records) acc.set(r.billing_project, (acc.get(r.billing_project) ?? 0) + r.total_spent);
        return [...acc.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([bp, cost]): [string, string] => [bp, fmtCost(cost) || '$0']);
      })()
    : [];

  const byUser: [string, string][] = records
    ? (() => {
        const acc = new Map<string, number>();
        for (const r of records) acc.set(r.user, (acc.get(r.user) ?? 0) + r.total_spent);
        return [...acc.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([u, cost]): [string, string] => [u, fmtCost(cost) || '$0']);
      })()
    : [];

  const byBpUser: [string, string, string][] = activeRecords
    ? [...activeRecords]
        .sort((a, b) => a.billing_project.localeCompare(b.billing_project) || a.user.localeCompare(b.user))
        .map((r): [string, string, string] => [r.billing_project, r.user, fmtCost(r.total_spent) || '$0'])
    : [];

  const byQuote: [string, string][] = quoteRecords
    ? (() => {
        const acc = new Map<string, number>();
        for (const r of quoteRecords) acc.set(r.quote_name, (acc.get(r.quote_name) ?? 0) + r.total_spent);
        return [...acc.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([q, cost]): [string, string] => [q, fmtCost(cost) || '$0']);
      })()
    : [];

  const byQuoteBp: [string, string, string][] = quoteRecords
    ? [...quoteRecords]
        .sort((a, b) => a.quote_name.localeCompare(b.quote_name) || a.billing_project.localeCompare(b.billing_project))
        .map((r): [string, string, string] => [r.quote_name, r.billing_project, fmtCost(r.total_spent) || '$0'])
    : [];

  const doExport = (action: 'download' | 'copy') => {
    if (!activeRecords) return;
    const { csv, filename } = buildCsv(activeRecords, tab, appliedStart, appliedEnd);

    if (action === 'download') {
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      showExportStatus('✓ Done');
    } else {
      navigator.clipboard.writeText(csv).then(() => showExportStatus('✓ Done')).catch(() => showExportStatus('Failed'));
    }
  };

  const showExportStatus = (msg: string) => {
    setExportStatus(msg);
    if (exportStatusTimeout.current) clearTimeout(exportStatusTimeout.current);
    exportStatusTimeout.current = setTimeout(() => setExportStatus(''), 1500);
  };

  const spendContext = mode === 'billing-projects'
    ? `In your billing projects, ${appliedStart} – ${appliedEnd || 'today'}`
    : mode === 'quotes'
    ? `Across your managed quotes, ${appliedStart} – ${appliedEnd || 'today'}`
    : `Your personal spend, ${appliedStart} – ${appliedEnd || 'today'}`;

  return (
    <div className="flex flex-wrap justify-around items-start md:mt-16">
      {/* LEFT: Settings */}
      <div className="lg:basis-1/3">
        <h1 className="text-2xl font-light mb-4">Billing</h1>


        <form onSubmit={(e) => void handleApply(e)}>
          <div className="flex flex-wrap justify-between space-y-2 items-end">
            <div className="flex flex-col">
              <label className="mb-1" htmlFor="billing-start">Start</label>
              <input
                id="billing-start"
                className="border rounded p-2"
                name="start"
                type="text"
                required
                value={start}
                onChange={(e) => setStart(e.target.value)}
                placeholder="MM/DD/YYYY"
              />
            </div>
            <div className="flex flex-col">
              <label className="mb-1" htmlFor="billing-end">End (inclusive)</label>
              <input
                id="billing-end"
                className="border rounded p-2"
                name="end"
                type="text"
                value={end}
                onChange={(e) => setEnd(e.target.value)}
                placeholder="MM/DD/YYYY (optional)"
              />
            </div>
            <div className="h-1/2 flex items-center gap-3">
              <button
                type="submit"
                disabled={loading}
                className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
              >
                {loading ? 'Loading…' : 'Apply'}
              </button>
              {isDirty && (
                <button
                  type="button"
                  onClick={handleUndo}
                  className="text-sm text-zinc-500 hover:text-zinc-800 underline hover:cursor-pointer"
                >
                  Undo changes
                </button>
              )}
            </div>
          </div>
        </form>

        <div className="mt-6">
          <p className="text-sm text-zinc-500 mb-2">View spending as…</p>
          <div className="flex rounded border border-zinc-300 overflow-hidden">
            {showQuotes && managedQuoteNames !== null && managedQuoteNames.size > 0 && (
              <button
                type="button"
                onClick={() => handleModeChange('quotes')}
                className={`flex-1 px-4 py-2 text-sm hover:cursor-pointer ${mode === 'quotes' ? 'bg-blue-600 text-white' : 'bg-white text-zinc-600 hover:bg-zinc-50'}`}
              >
                … a quote manager (of {managedQuoteNames.size} quotes)
              </button>
            )}
            {bpCount !== null && bpCount > 0 && (
              <button
                type="button"
                onClick={() => handleModeChange('billing-projects')}
                className={`flex-1 px-4 py-2 text-sm border-l border-zinc-300 hover:cursor-pointer ${mode === 'billing-projects' ? 'bg-blue-600 text-white' : 'bg-white text-zinc-600 hover:bg-zinc-50'}`}
              >
                … a member (of {bpCount} billing projects)
              </button>
            )}
            <button
              type="button"
              onClick={() => handleModeChange('just-me')}
              className={`flex-1 px-4 py-2 text-sm border-l border-zinc-300 hover:cursor-pointer ${mode === 'just-me' ? 'bg-blue-600 text-white' : 'bg-white text-zinc-600 hover:bg-zinc-50'}`}
            >
              … just me
            </button>
          </div>
        </div>

        <p className="text-zinc-500 text-balance py-8">
          Start must be a date in the format MM/DD/YYYY. End is an optional date in the format
          MM/DD/YYYY. Leave End empty to include currently running batches. If End is not empty,
          then no currently running batches are included. All dates search for batches that have
          completed within that time interval (inclusive).
        </p>
      </div>

      {/* RIGHT: Results */}
      <div className={`bg-slate-100 border rounded overflow-hidden lg:basis-1/2 transition-opacity ${isDirty && !loading ? 'opacity-40 pointer-events-none' : ''}`}>
        {error && <ErrorBanner message={error} />}

        {loading && (
          <div className="flex items-center justify-center p-12">
            <span className="text-3xl font-light text-sky-600">Loading&hellip;</span>
          </div>
        )}

        {!loading && records && (
          <>
            <div className="m-4 p-4 bg-white border rounded">
              <p className="text-xs text-zinc-400 uppercase tracking-wide font-medium mb-1">Total spend</p>
              <p className="text-zinc-400 text-sm mb-2">{spendContext}</p>
              <p className="text-2xl font-bold">{totalCost}</p>
            </div>

            <div className="bg-white">
              <div className="flex border-b text-lg flex-wrap items-center">
                {mode === 'billing-projects' && <TabButton label="By Billing Project" active={tab === 'by-project'} onClick={() => setTab('by-project')} />}
                {mode === 'billing-projects' && isGlobalBm && <TabButton label="By User" active={tab === 'by-user'} onClick={() => setTab('by-user')} />}
                {mode === 'billing-projects' && <TabButton label="By Billing Project and User" active={tab === 'by-bp-user'} onClick={() => setTab('by-bp-user')} />}
                {mode === 'quotes' && <TabButton label="By Quote" active={tab === 'by-quote'} onClick={() => setTab('by-quote')} />}
                {mode === 'quotes' && <TabButton label="By Quote and Billing Project" active={tab === 'by-quote-bp'} onClick={() => setTab('by-quote-bp')} />}
                {mode === 'just-me' && <TabButton label="By Billing Project and User" active={tab === 'by-bp-user'} onClick={() => setTab('by-bp-user')} />}
                <div className="ml-auto flex items-center gap-1 px-3 pb-1">
                  {exportStatus && <span className="text-sm text-zinc-400">{exportStatus}</span>}
                  <button
                    onClick={() => doExport('copy')}
                    title="Copy to clipboard"
                    className="p-2 text-zinc-400 hover:text-zinc-700 hover:cursor-pointer"
                  >
                    <ClipboardIcon />
                  </button>
                  <button
                    onClick={() => doExport('download')}
                    title="Download CSV"
                    className="p-2 text-zinc-400 hover:text-zinc-700 hover:cursor-pointer"
                  >
                    <DownloadIcon />
                  </button>
                </div>
              </div>

              {tab === 'by-project' && <SummaryTable rows={byProject} columns={['Billing Project', 'Cost']} />}
              {tab === 'by-user' && <SummaryTable rows={byUser} columns={['User', 'Cost']} />}
              {tab === 'by-bp-user' && <TwoColumnTable rows={byBpUser} columns={['Billing Project', 'User', 'Cost']} />}
              {tab === 'by-quote' && <SummaryTable rows={byQuote} columns={['Quote', 'Cost']} />}
              {tab === 'by-quote-bp' && <TwoColumnTable rows={byQuoteBp} columns={['Quote', 'Billing Project', 'Cost']} />}
            </div>
          </>
        )}

        {!loading && !records && !error && (
          <div className="p-8 text-slate-500 text-sm">No results.</div>
        )}
      </div>
    </div>
  );
}
