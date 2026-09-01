import { useState, useEffect, useCallback } from 'react';
import { AutoRefreshBar } from '../shared/AutoRefreshBar';
import { SpinnerIcon } from '../shared/SpinnerIcon';
import { formatDuration, formatTime } from '../shared/timeUtils';
import { hasPermission } from '../shared/authUtils';

const REFRESH_INTERVAL_MS = 30_000;


// ── Types ────────────────────────────────────────────────────────────────────

interface InstanceStateCounts {
  pending?: number;
  active?: number;
  inactive?: number;
  deleted?: number;
}

interface KnownFeatureFlags {
  compact_billing_tables?: boolean;
  oms_agent?: boolean;
  dockerhub_proxy?: boolean;
}

interface InstCollSummary {
  name: string;
  all_versions_instances_by_state: InstanceStateCounts;
  all_versions_cores_mcpu_by_state: InstanceStateCounts;
  max_live_instances: number;
  max_instances: number;
  schedulable_free_cores_mcpu: number;
  schedulable_cores_mcpu: number;
}

interface GlobalStats {
  n_instances_by_state: InstanceStateCounts;
  cores_mcpu_by_state: InstanceStateCounts;
  schedulable_free_cores_mcpu: number;
  schedulable_cores_mcpu: number;
}

interface DriverData {
  instance_id: string;
  frozen: boolean;
  ready_cores_mcpu: number;
  feature_flags: KnownFeatureFlags;
  pools: InstCollSummary[];
  jpim: InstCollSummary;
  global_stats: GlobalStats;
}

interface InstanceData {
  name: string;
  inst_coll_name: string;
  location: string;
  version: number;
  state: string;
  free_cores_mcpu: number;
  cores_mcpu: number;
  failed_request_count: number;
  time_created_ms: number | null;
  last_updated_ms: number | null;
}

type SortDir = 'asc' | 'desc';
type InstanceSortCol =
  | 'name' | 'inst_coll_name' | 'location' | 'version' | 'state'
  | 'free_cores_mcpu' | 'failed_request_count' | 'time_created_ms' | 'age' | 'last_updated_ms';

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatAgo(ms: number | null): string {
  const d = formatDuration(ms);
  return d ? `${d} ago` : '';
}

function pctFree(free: number, total: number): string {
  return total !== 0 ? `${((free * 100) / total).toFixed(1)}%` : '';
}

async function apiFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const method = (init?.method ?? 'GET').toUpperCase();
  const csrfHeaders: Record<string, string> = {};
  if (method !== 'GET' && method !== 'HEAD') {
    const token = document.head.querySelector('meta[name="csrf"]')?.getAttribute('value');
    if (token) csrfHeaders['X-CSRF-Token'] = token;
  }
  const resp = await fetch(url, {
    credentials: 'same-origin',
    ...init,
    headers: { ...csrfHeaders, ...init?.headers },
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(`HTTP ${resp.status}${text ? ': ' + text : ''}`);
  }
  return resp.json() as Promise<T>;
}

// ── Instance state icon ───────────────────────────────────────────────────────

function InstanceStateIcon({ state }: { state: string }): JSX.Element {
  if (state === 'pending') {
    return <SpinnerIcon className="text-sky-600 inline" />;
  }
  const { icon, color } = (() => {
    switch (state) {
      case 'active':   return { icon: 'check', color: 'text-green-600' };
      case 'inactive': return { icon: 'close', color: 'text-orange-500' };
      case 'deleted':  return { icon: 'close', color: 'text-red-600' };
      default:         return { icon: 'schedule', color: 'text-zinc-400' };
    }
  })();
  return <span className={`material-symbols-outlined text-base leading-none ${color}`}>{icon}</span>;
}

// ── Table primitives ──────────────────────────────────────────────────────────

const TH_BASE = 'whitespace-nowrap px-2.5 py-0.5 text-white bg-zinc-500 font-normal text-left border-r border-zinc-400 last:border-r-0';
const TD_BASE = 'px-2.5 py-0.5';

function Th({ children, className = '', colSpan }: { children?: React.ReactNode; className?: string; colSpan?: number }): JSX.Element {
  return <th colSpan={colSpan} className={`${TH_BASE} ${className}`}>{children}</th>;
}

function Td({ children, className = '' }: { children?: React.ReactNode; className?: string }): JSX.Element {
  return <td className={`${TD_BASE} ${className}`}>{children}</td>;
}

function TdNum({ children }: { children?: React.ReactNode }): JSX.Element {
  return <td className={`${TD_BASE} text-right tabular-nums`}>{children}</td>;
}

function FootTd({ children, className = '' }: { children?: React.ReactNode; className?: string }): JSX.Element {
  return <td className={`${TH_BASE} font-bold ${className}`}>{children}</td>;
}

function DataTr({ children }: { children: React.ReactNode }): JSX.Element {
  return <tr className="even:bg-zinc-100 hover:bg-zinc-200">{children}</tr>;
}

// ── Sortable column header ────────────────────────────────────────────────────

function SortTh({ col, label, sortCol, sortDir, onSort }: {
  col: string;
  label: string;
  sortCol: string;
  sortDir: SortDir;
  onSort: (_col: string) => void;
}): JSX.Element {
  const active = sortCol === col;
  return (
    <th
      className={TH_BASE}
      aria-sort={active ? (sortDir === 'asc' ? 'ascending' : 'descending') : 'none'}
    >
      <button
        className="inline-flex items-center gap-1 w-full hover:text-zinc-200"
        onClick={() => { onSort(col); }}
      >
        {label}
        <span className={`text-xs ${active ? '' : 'opacity-40'}`} aria-hidden>
          {active ? (sortDir === 'asc' ? '↑' : '↓') : '↑↓'}
        </span>
      </button>
    </th>
  );
}

// ── Instance Collections table ────────────────────────────────────────────────

type TipState = { text: string; x: number; y: number } | null;

function useTip(): [TipState, (text: string) => (e: React.MouseEvent) => void, () => void] {
  const [tip, setTip] = useState<TipState>(null);
  const onEnter = (text: string) => (e: React.MouseEvent) => setTip({ text, x: e.clientX, y: e.clientY });
  const onLeave = () => setTip(null);
  return [tip, onEnter, onLeave];
}

function FloatingTip({ tip }: { tip: TipState }): JSX.Element | null {
  if (!tip) return null;
  return (
    <div
      className="fixed z-50 pointer-events-none bg-zinc-800 text-white text-xs px-2 py-1 rounded shadow whitespace-nowrap"
      style={{ left: tip.x + 12, top: tip.y - 32 }}
    >
      {tip.text}
    </div>
  );
}

function PoolUsageBar({ ic, maxInstances }: { ic: InstCollSummary; maxInstances: number }): JSX.Element {
  const [tip, onEnter, onLeave] = useTip();
  if (maxInstances === 0 || ic.max_instances === 0) return <td className={TD_BASE} />;
  const pending  = ic.all_versions_instances_by_state['pending'] ?? 0;
  const active   = ic.all_versions_instances_by_state['active'] ?? 0;
  const inactive = ic.all_versions_instances_by_state['inactive'] ?? 0;
  const deleted  = ic.all_versions_instances_by_state['deleted'] ?? 0;
  const used   = pending + active + inactive + deleted;
  const unused = Math.max(0, ic.max_instances - used);
  const barPct = (ic.max_instances / maxInstances) * 100;
  const pct = (n: number) => ` (${((n / ic.max_instances) * 100).toFixed(1)}%)`;
  const segments: [number, string, string][] = [
    [pending,  'bg-sky-300',    `Pending: ${pending}${pct(pending)}`],
    [active,   'bg-green-500',  `Active: ${active}${pct(active)}`],
    [inactive, 'bg-orange-400', `Inactive: ${inactive}${pct(inactive)}`],
    [deleted,  'bg-red-400',    `Deleted: ${deleted}${pct(deleted)}`],
  ];
  return (
    <td className={`${TD_BASE} w-44`}>
      <div className="h-3.5 w-40">
        <div className="flex h-full bg-zinc-300 rounded overflow-hidden" style={{ width: `${barPct}%` }}>
          {segments.map(([value, color, label], i) => {
            const pct = (value / ic.max_instances) * 100;
            return pct > 0 ? (
              <div key={i} className={`${color} h-full flex-shrink-0`} style={{ width: `${pct}%` }}
                onMouseEnter={onEnter(label)} onMouseLeave={onLeave} />
            ) : null;
          })}
          {unused > 0 && (
            <div className="flex-1 h-full"
              onMouseEnter={onEnter(`Unused: ${unused}${pct(unused)}`)} onMouseLeave={onLeave} />
          )}
        </div>
      </div>
      <FloatingTip tip={tip} />
    </td>
  );
}

function CoreUtilBar({ ic }: { ic: InstCollSummary }): JSX.Element {
  const [tip, onEnter, onLeave] = useTip();
  const pending  = ic.all_versions_cores_mcpu_by_state['pending'] ?? 0;
  const activeAll = ic.all_versions_cores_mcpu_by_state['active'] ?? 0;
  const inactive = ic.all_versions_cores_mcpu_by_state['inactive'] ?? 0;
  const deleted  = ic.all_versions_cores_mcpu_by_state['deleted'] ?? 0;
  const free     = ic.schedulable_free_cores_mcpu;
  const activeUsed = Math.max(0, activeAll - free);
  const total = pending + activeAll + inactive + deleted;
  const cores = (mcpu: number) => (mcpu / 1000).toLocaleString(undefined, { maximumFractionDigits: 1 });
  const pct   = (mcpu: number) => total > 0 ? ` (${((mcpu / total) * 100).toFixed(1)}%)` : '';
  const label = (name: string, mcpu: number) => `${name}: ${cores(mcpu)} cores${pct(mcpu)}`;
  const segments: [number, string, string][] = [
    [pending,    'bg-sky-300',    label('Pending', pending)],
    [free,       'bg-blue-500',   label('Active (free)', free)],
    [activeUsed, 'bg-green-500',  label('Active (used)', activeUsed)],
    [inactive,   'bg-orange-400', label('Inactive', inactive)],
    [deleted,    'bg-red-400',    label('Deleted', deleted)],
  ];
  return (
    <td className={`${TD_BASE} w-44`}>
      <div className="flex h-3.5 w-40 bg-zinc-300 rounded overflow-hidden">
        {total === 0 ? null : segments.map(([value, color, label], i) => {
          const pct = (value / total) * 100;
          return pct > 0 ? (
            <div key={i} className={`${color} h-full flex-shrink-0`} style={{ width: `${pct}%` }}
              onMouseEnter={onEnter(label)} onMouseLeave={onLeave} />
          ) : null;
        })}
      </div>
      <FloatingTip tip={tip} />
    </td>
  );
}

function InstCollRow({ ic, href, showSchedulable, maxInstances }: {
  ic: InstCollSummary;
  href: string;
  showSchedulable: boolean;
  maxInstances: number;
}): JSX.Element {
  return (
    <DataTr>
      <Td><a href={href} className="text-sky-700 hover:underline">{ic.name}</a></Td>
      <TdNum>{ic.all_versions_instances_by_state.pending ?? 0}</TdNum>
      <TdNum>{ic.all_versions_instances_by_state.active ?? 0}</TdNum>
      <TdNum>{ic.all_versions_instances_by_state.inactive ?? 0}</TdNum>
      <TdNum>{ic.all_versions_instances_by_state.deleted ?? 0}</TdNum>
      <TdNum>{ic.max_instances}</TdNum>
      <Td />
      <TdNum>{(ic.all_versions_cores_mcpu_by_state.pending ?? 0) / 1000}</TdNum>
      <TdNum>{(ic.all_versions_cores_mcpu_by_state.active ?? 0) / 1000}</TdNum>
      <TdNum>{(ic.all_versions_cores_mcpu_by_state.inactive ?? 0) / 1000}</TdNum>
      <TdNum>{(ic.all_versions_cores_mcpu_by_state.deleted ?? 0) / 1000}</TdNum>
      <Td />
      {showSchedulable ? (
        <>
          <TdNum>{ic.schedulable_free_cores_mcpu / 1000}</TdNum>
          <TdNum>{ic.schedulable_cores_mcpu / 1000}</TdNum>
          <TdNum>{pctFree(ic.schedulable_free_cores_mcpu, ic.schedulable_cores_mcpu)}</TdNum>
        </>
      ) : (
        <><TdNum /><TdNum /><TdNum /></>
      )}
      <PoolUsageBar ic={ic} maxInstances={maxInstances} />
      <CoreUtilBar ic={ic} />
    </DataTr>
  );
}

function InstCollsTable({ pools, jpim, globalStats, basePath }: {
  pools: InstCollSummary[];
  jpim: InstCollSummary;
  globalStats: GlobalStats;
  basePath: string;
}): JSX.Element {
  const allIcs = [...pools, jpim];
  const maxInstances = Math.max(1, ...allIcs.map((ic) => ic.max_instances));
  return (
    <div className="overflow-x-auto">
      <table className="min-w-[600px] border-collapse">
        <thead>
          <tr>
            <Th colSpan={1}>Name</Th>
            <Th colSpan={5}>Instances</Th>
            <Th />
            <Th colSpan={4}>Cores</Th>
            <Th />
            <Th colSpan={3}>Schedulable Cores</Th>
            <Th colSpan={2}>Utilization</Th>
          </tr>
          <tr>
            <Th />
            <Th>Pending</Th><Th>Active</Th><Th>Inactive</Th><Th>Deleted</Th><Th>Max</Th>
            <Th />
            <Th>Pending</Th><Th>Active</Th><Th>Inactive</Th><Th>Deleted</Th>
            <Th />
            <Th>Free</Th><Th>Total</Th><Th>% Free</Th>
            <Th>Pool Usage</Th><Th>Core Utilization (%)</Th>
          </tr>
        </thead>
        <tbody>
          {pools.map((p) => (
            <InstCollRow key={p.name} ic={p} href={`${basePath}/inst_coll/pool/${p.name}`} showSchedulable maxInstances={maxInstances} />
          ))}
          <InstCollRow ic={jpim} href={`${basePath}/inst_coll/jpim`} showSchedulable={false} maxInstances={maxInstances} />
        </tbody>
        <tfoot>
          <tr>
            <FootTd>Total</FootTd>
            <FootTd className="text-right tabular-nums">{globalStats.n_instances_by_state.pending ?? 0}</FootTd>
            <FootTd className="text-right tabular-nums">{globalStats.n_instances_by_state.active ?? 0}</FootTd>
            <FootTd className="text-right tabular-nums">{globalStats.n_instances_by_state.inactive ?? 0}</FootTd>
            <FootTd className="text-right tabular-nums">{globalStats.n_instances_by_state.deleted ?? 0}</FootTd>
            <FootTd /><FootTd />
            <FootTd className="text-right tabular-nums">{(globalStats.cores_mcpu_by_state.pending ?? 0) / 1000}</FootTd>
            <FootTd className="text-right tabular-nums">{(globalStats.cores_mcpu_by_state.active ?? 0) / 1000}</FootTd>
            <FootTd className="text-right tabular-nums">{(globalStats.cores_mcpu_by_state.inactive ?? 0) / 1000}</FootTd>
            <FootTd className="text-right tabular-nums">{(globalStats.cores_mcpu_by_state.deleted ?? 0) / 1000}</FootTd>
            <FootTd />
            <FootTd className="text-right tabular-nums">{globalStats.schedulable_free_cores_mcpu / 1000}</FootTd>
            <FootTd className="text-right tabular-nums">{globalStats.schedulable_cores_mcpu / 1000}</FootTd>
            <FootTd className="text-right tabular-nums">{pctFree(globalStats.schedulable_free_cores_mcpu, globalStats.schedulable_cores_mcpu)}</FootTd>
            <FootTd /><FootTd />
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

// ── Instances table ───────────────────────────────────────────────────────────

function InstancesTable({ instances }: { instances: InstanceData[] }): JSX.Element {
  const [filter, setFilter] = useState('');
  const [sortCol, setSortCol] = useState<InstanceSortCol>('time_created_ms');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  const handleSort = (col: string) => {
    if (sortCol === col) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortCol(col as InstanceSortCol);
      setSortDir('asc');
    }
  };

  const filtered = filter
    ? instances.filter((i) => {
        const q = filter.toLowerCase();
        return (
          i.name.toLowerCase().includes(q) ||
          i.inst_coll_name.toLowerCase().includes(q) ||
          i.location.toLowerCase().includes(q) ||
          i.state.toLowerCase().includes(q)
        );
      })
    : instances;

  const sorted = [...filtered].sort((a, b) => {
    let av: string | number;
    let bv: string | number;
    switch (sortCol) {
      case 'name': av = a.name; bv = b.name; break;
      case 'inst_coll_name': av = a.inst_coll_name; bv = b.inst_coll_name; break;
      case 'location': av = a.location; bv = b.location; break;
      case 'version': av = a.version; bv = b.version; break;
      case 'state': av = a.state; bv = b.state; break;
      case 'free_cores_mcpu': av = a.free_cores_mcpu; bv = b.free_cores_mcpu; break;
      case 'failed_request_count': av = a.failed_request_count; bv = b.failed_request_count; break;
      case 'time_created_ms': av = a.time_created_ms ?? 0; bv = b.time_created_ms ?? 0; break;
      case 'age': av = a.time_created_ms ?? 0; bv = b.time_created_ms ?? 0; break;
      case 'last_updated_ms': av = a.last_updated_ms ?? 0; bv = b.last_updated_ms ?? 0; break;
      default: av = ''; bv = '';
    }
    if (av < bv) return sortDir === 'asc' ? -1 : 1;
    if (av > bv) return sortDir === 'asc' ? 1 : -1;
    return 0;
  });

  const shProps = { sortCol, sortDir, onSort: handleSort };

  return (
    <>
      <input
        type="text"
        placeholder="Filter by name, collection, location, or state…"
        value={filter}
        onChange={(e) => { setFilter(e.target.value); }}
        className="mb-2 px-3 py-1.5 border border-zinc-300 rounded text-sm w-full max-w-md"
      />
      <div className="w-fit max-w-full overflow-auto max-h-[600px] border border-zinc-200 rounded">
        <table className="border-collapse">
          <thead className="sticky top-0 z-10">
            <tr>
              <SortTh col="name" label="Name" {...shProps} />
              <SortTh col="inst_coll_name" label="Instance Collection" {...shProps} />
              <SortTh col="location" label="Location" {...shProps} />
              <SortTh col="version" label="Version" {...shProps} />
              <SortTh col="state" label="State" {...shProps} />
              <SortTh col="free_cores_mcpu" label="Free Cores" {...shProps} />
              <SortTh col="failed_request_count" label="Failed Requests" {...shProps} />
              <SortTh col="time_created_ms" label="Time Created" {...shProps} />
              <SortTh col="age" label="Age" {...shProps} />
              <SortTh col="last_updated_ms" label="Last Updated" {...shProps} />
              <Th>Logs</Th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((inst) => (
              <DataTr key={inst.name}>
                <Td>{inst.name}</Td>
                <Td>{inst.inst_coll_name}</Td>
                <Td>{inst.location}</Td>
                <TdNum>{inst.version}</TdNum>
                <Td><span className="inline-flex items-center gap-1"><InstanceStateIcon state={inst.state} />{inst.state}</span></Td>
                <TdNum>{inst.free_cores_mcpu / 1000} / {inst.cores_mcpu / 1000}</TdNum>
                <TdNum>{inst.failed_request_count}</TdNum>
                <Td className="font-mono">{formatTime(inst.time_created_ms)}</Td>
                <Td>{formatDuration(inst.time_created_ms)}</Td>
                <Td>{formatAgo(inst.last_updated_ms)}</Td>
                <Td>
                  <a
                    target="_blank"
                    rel="noreferrer"
                    className="text-sky-700 hover:underline inline-flex items-center gap-0.5"
                    href={`https://console.cloud.google.com/logs/query;query=%22${inst.name}%22%0A-protoPayload.serviceName%3D%22logging.googleapis.com%22%0A-protoPayload.serviceName%3D%22securitycenter.googleapis.com%22`}
                  >
                    Logs
                    <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>open_in_new</span>
                  </a>
                </Td>
              </DataTr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

// ── Feature Flags section ─────────────────────────────────────────────────────

function FeatureFlagsSection({
  flags, canUpdate, basePath, onFlagsUpdated,
}: {
  flags: KnownFeatureFlags;
  canUpdate: boolean;
  basePath: string;
  onFlagsUpdated: (_flags: KnownFeatureFlags) => void;
}): JSX.Element {
  const [localFlags, setLocalFlags] = useState<KnownFeatureFlags | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const display = localFlags ?? flags;

  const handleSave = async () => {
    if (!localFlags) return;
    setSaving(true);
    setError(null);
    try {
      const result = await apiFetch<KnownFeatureFlags>(`${basePath}/api/v1alpha/feature_flags`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(localFlags),
      });
      onFlagsUpdated(result);
      setLocalFlags(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  if (!canUpdate) {
    return (
      <div className="ml-4 space-y-1 text-sm">
        <div>compact_billing_tables: <span className="font-mono">{String(flags.compact_billing_tables ?? false)}</span></div>
        <div>oms_agent: <span className="font-mono">{String(flags.oms_agent ?? false)}</span></div>
        <div>dockerhub_proxy: <span className="font-mono">{String(flags.dockerhub_proxy ?? false)}</span></div>
      </div>
    );
  }

  return (
    <div className="ml-4">
      <label className="flex items-center gap-2 mb-1 cursor-pointer text-sm w-fit">
        <input type="checkbox" checked={display.compact_billing_tables ?? false}
          onChange={(e) => { setLocalFlags({ ...(localFlags ?? flags), compact_billing_tables: e.target.checked }); }} />
        compact_billing_tables
      </label>
      <label className="flex items-center gap-2 mb-1 cursor-pointer text-sm w-fit">
        <input type="checkbox" checked={display.oms_agent ?? false}
          onChange={(e) => { setLocalFlags({ ...(localFlags ?? flags), oms_agent: e.target.checked }); }} />
        oms_agent
      </label>
      <label className="flex items-center gap-2 mb-1 cursor-pointer text-sm w-fit">
        <input type="checkbox" checked={display.dockerhub_proxy ?? false}
          onChange={(e) => { setLocalFlags({ ...(localFlags ?? flags), dockerhub_proxy: e.target.checked }); }} />
        dockerhub_proxy
      </label>
      {error && <p className="text-red-600 text-sm mt-1">{error}</p>}
      <button
        className="mt-2 px-2 py-0.5 border border-zinc-900 rounded text-sm cursor-pointer hover:text-zinc-500 hover:border-zinc-500 disabled:opacity-40"
        disabled={saving || localFlags === null}
        onClick={() => { void handleSave(); }}
      >
        {saving ? 'Saving…' : 'Update'}
      </button>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export function DriverDashboard({ basePath }: {
  basePath: string;
}): JSX.Element {
  const [driver, setDriver] = useState<DriverData | null>(null);
  const [instances, setInstances] = useState<InstanceData[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [countdownKey, setCountdownKey] = useState(0);
  const [autoRefresh, setAutoRefreshState] = useState<boolean>(() => {
    try { return localStorage.getItem('batch_driver.dashboard.autoRefresh') !== 'false'; }
    catch { return true; }
  });
  const [frozenSaving, setFrozenSaving] = useState(false);
  const [frozenError, setFrozenError] = useState<string | null>(null);

  const setAutoRefresh = useCallback((v: boolean) => {
    setAutoRefreshState(v);
    try { localStorage.setItem('batch_driver.dashboard.autoRefresh', String(v)); }
    catch { /* ignore */ }
  }, []);

  const fetchData = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    try {
      const driverData = await apiFetch<DriverData>(`${basePath}/api/v1alpha/driver`);

      const poolFetches = driverData.pools.map((p) =>
        apiFetch<{ items: InstanceData[] }>(`${basePath}/api/v1alpha/inst_coll/pool/${p.name}/instances`),
      );
      const instanceResponses = await Promise.all([
        ...poolFetches,
        apiFetch<{ items: InstanceData[] }>(`${basePath}/api/v1alpha/inst_coll/jpim/instances`),
      ]);

      setDriver(driverData);
      setInstances(instanceResponses.flatMap((r) => r.items));
      setError(null);
      if (isRefresh) setCountdownKey((k) => k + 1);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
      if (isRefresh) setRefreshing(false);
    }
  }, [basePath]);

  useEffect(() => { void fetchData(false); }, [fetchData]);

  useEffect(() => {
    if (!autoRefresh) return;
    let cancelled = false;
    const schedule = (delay: number): ReturnType<typeof setTimeout> =>
      setTimeout(async () => {
        if (!cancelled) {
          await fetchData(true);
          if (!cancelled) schedule(REFRESH_INTERVAL_MS);
        }
      }, delay);
    const id = schedule(REFRESH_INTERVAL_MS);
    return () => { cancelled = true; clearTimeout(id); };
  }, [autoRefresh, fetchData]);

  const handleFreeze = async (freeze: boolean) => {
    setFrozenSaving(true);
    setFrozenError(null);
    try {
      const data = await apiFetch<{ frozen: boolean }>(`${basePath}/api/v1alpha/driver`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frozen: freeze }),
      });
      setDriver((d) => (d ? { ...d, frozen: data.frozen } : d));
    } catch (e) {
      setFrozenError(String(e));
    } finally {
      setFrozenSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center mt-24">
        <span className="text-5xl font-light text-sky-600">Loading&hellip;</span>
      </div>
    );
  }

  if (error || !driver) {
    return <div className="mt-8 text-red-600">Error loading driver status: {error ?? 'unknown error'}</div>;
  }

  return (
    <div className="pb-8">
      <div className="mb-4">
        <AutoRefreshBar
          autoRefresh={autoRefresh}
          onToggle={setAutoRefresh}
          countdownKey={countdownKey}
          refreshing={refreshing}
          intervalMs={REFRESH_INTERVAL_MS}
        />
      </div>

      <h1 className="text-2xl font-semibold mt-4 mb-2">Globals</h1>
      <div className="ml-4 space-y-1 text-sm">
        <div>instance ID: <span className="font-mono">{driver.instance_id}</span></div>
        <div>ready cores: <span className="font-mono">{driver.ready_cores_mcpu / 1000}</span></div>
        <div>frozen: <span className="font-mono">{String(driver.frozen)}</span></div>
      </div>
      {hasPermission('update_deployed_system_state') && (
        <div className="mt-2 ml-4">
          {frozenError && <p className="text-red-600 text-sm mb-1">{frozenError}</p>}
          <button
            className="px-2 py-0.5 border border-red-600 text-red-600 rounded text-sm cursor-pointer hover:text-red-400 hover:border-red-400 disabled:opacity-40"
            disabled={frozenSaving}
            onClick={() => { void handleFreeze(!driver.frozen); }}
          >
            {frozenSaving ? 'Saving…' : (driver.frozen ? 'Unfreeze' : 'Freeze')}
          </button>
        </div>
      )}

      <h1 className="text-2xl font-semibold mt-6 mb-2">Feature Flags</h1>
      <FeatureFlagsSection
        flags={driver.feature_flags}
        canUpdate={hasPermission('update_deployed_system_state')}
        basePath={basePath}
        onFlagsUpdated={(updated) => { setDriver((d) => (d ? { ...d, feature_flags: updated } : d)); }}
      />

      <h1 className="text-2xl font-semibold mt-6 mb-2">Instance Collections</h1>
      <InstCollsTable
        pools={driver.pools}
        jpim={driver.jpim}
        globalStats={driver.global_stats}
        basePath={basePath}
      />

      <h1 className="text-2xl font-semibold mt-6 mb-2">Instances</h1>
      <p className="text-sm text-zinc-500 mb-2">Total: {instances.length}</p>
      <InstancesTable instances={instances} />
    </div>
  );
}
