import { useState, useEffect, useCallback, useMemo } from 'react';
import { BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, ScatterChart, Scatter, ZAxis } from 'recharts';
import { useLegendToggle } from '../shared/useLegendToggle';
import { Panel } from '../shared/Panel';
import { RatioRow } from '../shared/RatioRow';
import { ToggleSwitch } from '../shared/ToggleSwitch';
import { MiniPieChart, type PieSlice } from '../shared/MiniPieChart';
import { PresetChips, ScatterPresetChips } from '../shared/PresetChips';
import { fmt, pct, makeYDollarFormatter } from './components/format';
import { computeStats, computeRegression, toPctRows } from './components/statsUtils';
import { CostRow } from './components/CostRow';
import { ChartTooltip } from './components/ChartTooltip';
import { StatsDisplay, RegressionStatsDisplay, statsReferenceLines } from './components/StatsDisplay';
import { createHailApi, type HailApi } from './components/hailApi';

// --- Types ---

// GCP service name for the GKE cluster management fee line item
const GKE_SERVICE_NAME = 'Kubernetes Engine';

// Cap how many months' worth of trend data we fetch concurrently (2 requests/month) to avoid
// spiking load on the monitoring/batch APIs when a wide date range is selected.
const TRENDS_FETCH_BATCH_SIZE = 6;

interface CloudCosts {
  user_compute: number;
  other_compute: number;
  k8s: number;
  total: number;
  // other_compute sub-breakdown
  batch_test: number;
  batch_dev: number;
  unknown: number;
  // k8s sub-breakdown
  k8s_nodes: number;
  k8s_mgmt: number;
  // non-compute GCP services (excludes Kubernetes Engine), service → total cost
  non_compute_services: Map<string, number>;
  // non-compute services broken down by SKU: service → sku → cost
  overhead_by_sku: Map<string, Map<string, number>>;
  // user_compute sub-breakdown by SKU product name
  user_compute_by_product: Map<string, number>;
}

interface UserBilling {
  total: number;
  resource_cost: number;
  service_fee_cost: number;
  resource_by_type: Map<string, number>;
  billing_by_project: Map<string, number>;
  billing_project_count: number;
  billing_project_concentration: number;
}

interface MonthDataPoint {
  month: string;
  cloud_total: number;
  user_compute: number;
  other_compute: number;
  k8s: number;
  batch_test: number;
  batch_dev: number;
  unknown: number;
  k8s_nodes: number;
  k8s_mgmt: number;
  non_compute_services: Map<string, number>;
  overhead_by_sku: Map<string, Map<string, number>>;
  user_compute_by_product: Map<string, number>;
  resource_by_type: Map<string, number>;
  billing_by_project: Map<string, number>;
  billing_project_count: number;
  billing_project_concentration: number;
  user_billing: number;
  service_fees: number;
  resource_cost: number;
  profit: number;
  svc_fee_overhead_pct: number | null;
  resource_billing_pct: number | null;
  svc_fee_bill_pct: number | null;
  overhead_cloud_pct: number | null;
  overhead_resource_pct: number | null;
}

// --- Helpers ---

function currentMonthParam(): string {
  const now = new Date();
  return `${String(now.getMonth() + 1).padStart(2, '0')}/${now.getFullYear()}`;
}

function monthParamToInputValue(param: string): string {
  const [mm, yyyy] = param.split('/');
  return `${yyyy}-${mm}`;
}

function inputValueToMonthParam(value: string): string {
  const [yyyy, mm] = value.split('-');
  return `${mm}/${yyyy}`;
}

function monthToDateRange(param: string): { start: string; end: string } {
  const [mm, yyyy] = param.split('/');
  const lastDay = new Date(parseInt(yyyy, 10), parseInt(mm, 10), 0).getDate();
  const pad = (n: number) => String(n).padStart(2, '0');
  return { start: `${mm}/01/${yyyy}`, end: `${mm}/${pad(lastDay)}/${yyyy}` };
}

function parseCostStr(s: string): number {
  if (!s || s.startsWith('<')) return 0;
  return parseFloat(s.replace(/[$,]/g, '')) || 0;
}

function fmtCoreHours(v: number): string {
  return Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)}k` : `${Math.round(v)}`;
}

function shiftMonthParam(param: string, delta: number): string {
  const [mm, yyyy] = param.split('/');
  const d = new Date(parseInt(yyyy, 10), parseInt(mm, 10) - 1 + delta, 1);
  return `${String(d.getMonth() + 1).padStart(2, '0')}/${d.getFullYear()}`;
}

function monthParamToLabel(param: string): string {
  const [mm, yyyy] = param.split('/');
  return new Date(parseInt(yyyy, 10), parseInt(mm, 10) - 1, 1).toLocaleString('en-US', { month: 'short', year: 'numeric' });
}

function monthsBetween(start: string, end: string): string[] {
  const toOrd = (p: string) => { const [mm, yyyy] = p.split('/'); return parseInt(yyyy, 10) * 12 + parseInt(mm, 10); };
  const months: string[] = [];
  let current = start;
  while (toOrd(current) <= toOrd(end)) {
    months.push(current);
    current = shiftMonthParam(current, 1);
  }
  return months;
}

// --- Custom ratio field helpers ---

interface FieldGroup { group: string; fields: { id: string; label: string }[] }

function buildFieldGroups(products: string[], overheadServices: string[], overheadSkusByService: Map<string, string[]>, resourceTypes: string[], billingProjects: string[]): FieldGroup[] {
  return [
    {
      group: 'Cloud Costs',
      fields: [
        { id: 'cloud/total', label: 'Cloud Costs / Total' },
        { id: 'cloud/user_compute', label: 'Cloud Costs / Compute (user-driven)' },
        ...products.map(p => ({ id: `cloud/user_compute/${p}`, label: `Cloud Costs / Compute (user-driven) / ${p}` })),
        { id: 'cloud/other_compute', label: 'Cloud Costs / Compute (other)' },
        { id: 'cloud/batch_test', label: 'Cloud Costs / Compute (other) / CI/test batches' },
        { id: 'cloud/batch_dev', label: 'Cloud Costs / Compute (other) / Dev batches' },
        { id: 'cloud/unknown', label: 'Cloud Costs / Compute (other) / Unknown/unlabeled' },
        { id: 'cloud/k8s', label: 'Cloud Costs / K8s' },
        { id: 'cloud/k8s_nodes', label: 'Cloud Costs / K8s / Nodes' },
        { id: 'cloud/k8s_mgmt', label: 'Cloud Costs / K8s / Management' },
        ...overheadServices.flatMap(s => [
          { id: `cloud/svc/${s}`, label: `Cloud Costs / ${s}` },
          ...(overheadSkusByService.get(s) ?? []).map(sku => ({ id: `cloud/svc/${s}/${sku}`, label: `Cloud Costs / ${s} / ${sku}` })),
        ]),
      ],
    },
    {
      group: 'User Billing',
      fields: [
        { id: 'billing/total', label: 'User Billing / Total' },
        { id: 'billing/resource_cost', label: 'User Billing / Resource cost' },
        ...resourceTypes.map(r => ({ id: `billing/resource/${r}`, label: `User Billing / Resource cost / ${r}` })),
        { id: 'billing/service_fees', label: 'User Billing / Service fees' },
        { id: 'billing/project_count', label: 'User Billing / Billing Projects / Active count' },
        { id: 'billing/project_concentration', label: 'User Billing / Billing Projects / Concentration' },
        ...billingProjects.map(p => ({ id: `billing/project/${p}`, label: `User Billing / Billing Projects / ${p}` })),
      ],
    },
    {
      group: 'Margin Analysis',
      fields: [
        { id: 'margin/profit', label: 'Margin Analysis / Profit ($)' },
        { id: 'margin/margin_pct', label: 'Margin Analysis / Margin %' },
      ],
    },
    {
      group: 'Usage',
      fields: [
        { id: 'derived/core_hours', label: 'Usage / Core hours' },
      ],
    },
  ];
}

function fieldLabel(id: string, groups: FieldGroup[]): string {
  for (const g of groups) {
    const f = g.fields.find(f => f.id === id);
    if (f) return f.label;
  }
  return id;
}

function resolveMonthly(id: string, c: CloudCosts, b: UserBilling): number {
  if (id === 'cloud/total') return c.total;
  if (id === 'cloud/user_compute') return c.user_compute;
  if (id.startsWith('cloud/user_compute/')) return c.user_compute_by_product.get(id.slice(19)) ?? 0;
  if (id === 'cloud/other_compute') return c.other_compute;
  if (id === 'cloud/batch_test') return c.batch_test;
  if (id === 'cloud/batch_dev') return c.batch_dev;
  if (id === 'cloud/unknown') return c.unknown;
  if (id === 'cloud/k8s') return c.k8s;
  if (id === 'cloud/k8s_nodes') return c.k8s_nodes;
  if (id === 'cloud/k8s_mgmt') return c.k8s_mgmt;
  if (id.startsWith('cloud/svc/')) {
    const rest = id.slice('cloud/svc/'.length);
    const slash = rest.indexOf('/');
    if (slash === -1) return c.non_compute_services.get(rest) ?? 0;
    return c.overhead_by_sku.get(rest.slice(0, slash))?.get(rest.slice(slash + 1)) ?? 0;
  }
  if (id === 'billing/total') return b.total;
  if (id === 'billing/resource_cost') return b.resource_cost;
  if (id.startsWith('billing/resource/')) return b.resource_by_type.get(id.slice(17)) ?? 0;
  if (id === 'billing/service_fees') return b.service_fee_cost;
  if (id === 'billing/project_count') return b.billing_project_count;
  if (id === 'billing/project_concentration') return b.billing_project_concentration;
  if (id.startsWith('billing/project/')) return b.billing_by_project.get(id.slice(16)) ?? 0;
  if (id === 'margin/profit') return b.total - c.total;
  if (id === 'margin/margin_pct') return c.total === 0 ? 0 : ((b.total - c.total) / c.total) * 100;
  if (id === 'derived/core_hours') return b.service_fee_cost * 100;
  return 0;
}

function resolveTrend(id: string, p: MonthDataPoint): number {
  if (id === 'cloud/total') return p.cloud_total;
  if (id === 'cloud/user_compute') return p.user_compute;
  if (id.startsWith('cloud/user_compute/')) return p.user_compute_by_product.get(id.slice(19)) ?? 0;
  if (id === 'cloud/other_compute') return p.other_compute;
  if (id === 'cloud/batch_test') return p.batch_test;
  if (id === 'cloud/batch_dev') return p.batch_dev;
  if (id === 'cloud/unknown') return p.unknown;
  if (id === 'cloud/k8s') return p.k8s;
  if (id === 'cloud/k8s_nodes') return p.k8s_nodes;
  if (id === 'cloud/k8s_mgmt') return p.k8s_mgmt;
  if (id.startsWith('cloud/svc/')) {
    const rest = id.slice('cloud/svc/'.length);
    const slash = rest.indexOf('/');
    if (slash === -1) return p.non_compute_services.get(rest) ?? 0;
    return p.overhead_by_sku.get(rest.slice(0, slash))?.get(rest.slice(slash + 1)) ?? 0;
  }
  if (id === 'billing/total') return p.user_billing;
  if (id === 'billing/resource_cost') return p.resource_cost;
  if (id.startsWith('billing/resource/')) return p.resource_by_type.get(id.slice(17)) ?? 0;
  if (id === 'billing/service_fees') return p.service_fees;
  if (id === 'billing/project_count') return p.billing_project_count;
  if (id === 'billing/project_concentration') return p.billing_project_concentration;
  if (id.startsWith('billing/project/')) return p.billing_by_project.get(id.slice(16)) ?? 0;
  if (id === 'margin/profit') return p.profit;
  if (id === 'margin/margin_pct') return p.cloud_total === 0 ? 0 : (p.profit / p.cloud_total) * 100;
  if (id === 'derived/core_hours') return p.service_fees * 100;
  return 0;
}

// --- API fetchers ---

async function fetchCloudCosts(api: HailApi, period: string): Promise<CloudCosts> {
  const data = await api.monitoring.billing.get(period);

  const breakdown = data.compute_cost_breakdown ?? [];
  const byService = data.cost_by_service ?? [];
  const bySkuLabel = data.cost_by_sku_label ?? [];

  const costs: CloudCosts = { user_compute: 0, other_compute: 0, k8s: 0, total: 0, batch_test: 0, batch_dev: 0, unknown: 0, k8s_nodes: 0, k8s_mgmt: 0, non_compute_services: new Map(), overhead_by_sku: new Map(), user_compute_by_product: new Map() };
  for (const row of breakdown) {
    const cost = parseCostStr(row.cost);
    if (row.source === 'batch-production') costs.user_compute += cost;
    else if (row.source === 'k8s') { costs.k8s_nodes += cost; costs.k8s += cost; }
    else {
      costs.other_compute += cost;
      if (row.source === 'batch-test') costs.batch_test += cost;
      else if (row.source === 'batch-dev') costs.batch_dev += cost;
      else costs.unknown += cost;
    }
  }
  for (const row of byService) {
    const cost = parseCostStr(row.cost);
    if (row.service === 'Compute Engine') continue;
    if (row.service === GKE_SERVICE_NAME) { costs.k8s_mgmt += cost; costs.k8s += cost; }
    else { costs.non_compute_services.set(row.service, (costs.non_compute_services.get(row.service) ?? 0) + cost); }
  }
  for (const row of bySkuLabel) {
    if (row.source === 'batch-production') {
      const cost = parseCostStr(row.cost);
      costs.user_compute_by_product.set(row.sku_description, (costs.user_compute_by_product.get(row.sku_description) ?? 0) + cost);
    } else if (row.source === null && row.service_description !== GKE_SERVICE_NAME) {
      const cost = parseCostStr(row.cost);
      const skus = costs.overhead_by_sku.get(row.service_description) ?? new Map<string, number>();
      skus.set(row.sku_description, (skus.get(row.sku_description) ?? 0) + cost);
      costs.overhead_by_sku.set(row.service_description, skus);
    }
  }
  const cloudTotal = byService.reduce((sum, row) => sum + parseCostStr(row.cost), 0);
  costs.total = cloudTotal;
  return costs;
}

async function fetchUserBilling(api: HailApi, period: string): Promise<UserBilling> {
  const { start, end } = monthToDateRange(period);
  const rows = await api.batch.billingBreakdown.get(start, end);

  let total = 0;
  let service_fee_cost = 0;
  const resource_by_type = new Map<string, number>();
  const billing_by_project = new Map<string, number>();
  for (const row of rows) {
    total += row.cost;
    billing_by_project.set(row.billing_project, (billing_by_project.get(row.billing_project) ?? 0) + row.cost);
    if (row.resource.startsWith('service-fee')) {
      service_fee_cost += row.cost;
    } else {
      const lastSlash = row.resource.lastIndexOf('/');
      const key = lastSlash !== -1 && /^\d+$/.test(row.resource.slice(lastSlash + 1))
        ? row.resource.slice(0, lastSlash)
        : row.resource;
      resource_by_type.set(key, (resource_by_type.get(key) ?? 0) + row.cost);
    }
  }
  const billing_project_count = billing_by_project.size;
  const projectCosts = [...billing_by_project.values()];
  const projectTotal = projectCosts.reduce((a, b) => a + b, 0);
  const hhi = projectTotal > 0 ? projectCosts.reduce((s, c) => s + (c / projectTotal) ** 2, 0) : 0;
  const billing_project_concentration = billing_project_count > 1
    ? (hhi - 1 / billing_project_count) / (1 - 1 / billing_project_count)
    : billing_project_count === 1 ? 1 : 0;
  return { total, resource_cost: total - service_fee_cost, service_fee_cost, resource_by_type, billing_by_project, billing_project_count, billing_project_concentration };
}

// --- Components ---

function CustomRatioPicker({ fieldGroups, num, den, onNumChange, onDenChange }: {
  fieldGroups: FieldGroup[];
  num: string; den: string;
  onNumChange: (_v: string) => void;
  onDenChange: (_v: string) => void;
}) {
  const selectClass = 'text-xs border border-zinc-300 rounded px-2 py-1 bg-white text-zinc-600 focus:outline-none focus:ring-2 focus:ring-sky-400 flex-1 min-w-0';
  const renderOptions = () => fieldGroups.map(g => (
    <optgroup key={g.group} label={g.group}>
      {g.fields.map(f => <option key={f.id} value={f.id}>{f.label}</option>)}
    </optgroup>
  ));
  return (
    <div className="flex items-center gap-2 py-2 flex-wrap">
      <span className="text-xs text-zinc-500 shrink-0">Custom:</span>
      <select className={selectClass} value={num} onChange={e => { onNumChange(e.target.value); }}>{renderOptions()}</select>
      <span className="text-xs text-zinc-400 shrink-0">as % of</span>
      <select className={selectClass} value={den} onChange={e => { onDenChange(e.target.value); }}>{renderOptions()}</select>
    </div>
  );
}

// --- Preset quick-links ---

const RATIO_PRESETS: { label: string; num: string; den: string }[] = [
  { label: 'Resource billing as % of user-driven compute',  num: 'billing/resource_cost',  den: 'cloud/user_compute' },
  { label: 'Approximate pool utilization',           num: 'billing/resource/compute/n1-preemptible/us-central1', den: 'cloud/user_compute/Spot Preemptible N1 Predefined Instance Core running in Americas' },
  { label: 'Service fees as % of user bill',          num: 'billing/service_fees',   den: 'billing/total' },
  { label: 'K8s as % of cloud costs',                num: 'cloud/k8s',              den: 'cloud/total' },
  { label: 'User billing as % of cloud costs',       num: 'billing/total',          den: 'cloud/total' },
];

const SCATTER_PRESETS: { label: string; x: string; y: string }[] = [
  { label: 'Core hours vs Profit',             x: 'derived/core_hours', y: 'margin/profit' },
  { label: 'Cloud total vs Profit',            x: 'cloud/total',        y: 'margin/profit' },
  { label: 'Compute (user-driven) vs Resource billing', x: 'cloud/user_compute', y: 'billing/resource_cost' },
  { label: 'Cloud total vs User billing',      x: 'cloud/total',        y: 'billing/total' },
  { label: 'Compute (user-driven) vs Margin %',         x: 'cloud/user_compute', y: 'margin/margin_pct' },
];

// --- Main component ---

interface CostAnalysisProps { monitoringBaseUrl: string; batchBaseUrl: string }

export function CostAnalysis({ monitoringBaseUrl, batchBaseUrl }: CostAnalysisProps) {
  const api = useMemo(() => createHailApi({ monitoring: monitoringBaseUrl, batch: batchBaseUrl }), [monitoringBaseUrl, batchBaseUrl]);

  const [tab, setTab] = useState<'monthly' | 'trends'>(() => {
    const p = new URLSearchParams(window.location.search).get('tab');
    return p === 'trends' ? 'trends' : 'monthly';
  });

  const changeTab = useCallback((t: 'monthly' | 'trends') => {
    setTab(t);
    const url = new URL(window.location.href);
    url.searchParams.set('tab', t);
    window.history.replaceState(null, '', url.toString());
  }, []);
  const [cloudView, setCloudView] = useState<string>('summary');
  const [billingView, setBillingView] = useState<'summary' | 'resource_usage' | 'billing_project'>('summary');
  const cloudCostsToggle = useLegendToggle(['user_compute', 'other_compute', 'k8s'] as const);
  const otherComputeToggle = useLegendToggle(['batch_test', 'batch_dev', 'unknown'] as const);
  const k8sToggle = useLegendToggle(['k8s_nodes', 'k8s_mgmt'] as const);
  const billingToggle = useLegendToggle(['resource_cost', 'service_fees'] as const);
  const [cloudShowPct, setCloudShowPct] = useState(false);
  const [billingShowPct, setBillingShowPct] = useState(false);
  const [timePeriod, setTimePeriod] = useState(() => new URLSearchParams(window.location.search).get('month') ?? currentMonthParam());
  const [cloudCosts, setCloudCosts] = useState<CloudCosts | null>(null);
  const [userBilling, setUserBilling] = useState<UserBilling | null>(null);
  const [cloudError, setCloudError] = useState<string | null>(null);
  const [billingError, setBillingError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [trendData, setTrendData] = useState<MonthDataPoint[]>([]);
  const [customRatioNum, setCustomRatioNum] = useState('billing/resource_cost');
  const [customRatioDen, setCustomRatioDen] = useState('cloud/user_compute');
  const [scatterX, setScatterX] = useState('derived/core_hours');
  const [scatterY, setScatterY] = useState('margin/profit');
  const [showRegression, setShowRegression] = useState(false);
  const [trendsLoading, setTrendsLoading] = useState(false);
  const [trendsError, setTrendsError] = useState<string | null>(null);
  const [trendsStart, setTrendsStart] = useState(() => new URLSearchParams(window.location.search).get('trends_start') ?? shiftMonthParam(currentMonthParam(), -12));
  const [trendsEnd, setTrendsEnd] = useState(() => new URLSearchParams(window.location.search).get('trends_end') ?? shiftMonthParam(currentMonthParam(), -1));

  const [compareTimePeriod, setCompareTimePeriod] = useState<string | null>(() => new URLSearchParams(window.location.search).get('comparison_month'));
  const [compareCloudCosts, setCompareCloudCosts] = useState<CloudCosts | null>(null);
  const [compareUserBilling, setCompareUserBilling] = useState<UserBilling | null>(null);
  const [compareCloudError, setCompareCloudError] = useState<string | null>(null);
  const [compareBillingError, setCompareBillingError] = useState<string | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);

  const { overheadServices, overheadSkusByService } = useMemo(() => {
    const svcMax = new Map<string, number>();
    const skuMax = new Map<string, Map<string, number>>();
    const trackSvc = (svc: string, val: number) => { svcMax.set(svc, Math.max(svcMax.get(svc) ?? 0, val)); };
    const trackSku = (svc: string, sku: string, val: number) => {
      const skus = skuMax.get(svc) ?? new Map<string, number>();
      skus.set(sku, Math.max(skus.get(sku) ?? 0, val));
      skuMax.set(svc, skus);
    };
    const processCloud = (c: CloudCosts) => {
      for (const [svc, val] of c.non_compute_services) trackSvc(svc, val);
      for (const [svc, skus] of c.overhead_by_sku) {
        for (const [sku, val] of skus) trackSku(svc, sku, val);
      }
    };
    if (cloudCosts) processCloud(cloudCosts);
    if (compareCloudCosts) processCloud(compareCloudCosts);
    for (const d of trendData) {
      for (const [svc, val] of d.non_compute_services) trackSvc(svc, val);
      for (const [svc, skus] of d.overhead_by_sku) {
        for (const [sku, val] of skus) trackSku(svc, sku, val);
      }
    }
    const services = [...svcMax.keys()].filter(s => (svcMax.get(s) ?? 0) >= 10).sort();
    const skusByService = new Map<string, string[]>();
    for (const svc of services) {
      const skus = skuMax.get(svc) ?? new Map<string, number>();
      skusByService.set(svc, [...skus.keys()].filter(k => (skus.get(k) ?? 0) >= 10).sort());
    }
    return { overheadServices: services, overheadSkusByService: skusByService };
  }, [cloudCosts, compareCloudCosts, trendData]);

  const { billingResources, billingResourcesHasOther } = useMemo(() => {
    const maxes = new Map<string, number>();
    for (const d of trendData) {
      for (const [r, v] of d.resource_by_type) maxes.set(r, Math.max(maxes.get(r) ?? 0, v));
    }
    if (userBilling) {
      for (const [r, v] of userBilling.resource_by_type) maxes.set(r, Math.max(maxes.get(r) ?? 0, v));
    }
    const resources = [...maxes.keys()].filter(r => (maxes.get(r) ?? 0) >= 10).sort();
    const hasOther = [...maxes.values()].some(v => v < 10);
    return { billingResources: resources, billingResourcesHasOther: hasOther };
  }, [trendData, userBilling]);

  const brOther = useCallback(
    (byType: Map<string, number>) =>
      [...byType].filter(([r]) => !billingResources.includes(r)).reduce((s, [, v]) => s + v, 0),
    [billingResources]
  );

  const { billingProjects, billingProjectsHasOther } = useMemo(() => {
    const maxes = new Map<string, number>();
    for (const d of trendData) {
      for (const [p, v] of d.billing_by_project) maxes.set(p, Math.max(maxes.get(p) ?? 0, v));
    }
    if (userBilling) {
      for (const [p, v] of userBilling.billing_by_project) maxes.set(p, Math.max(maxes.get(p) ?? 0, v));
    }
    const projects = [...maxes.keys()].filter(p => (maxes.get(p) ?? 0) >= 10).sort();
    const hasOther = [...maxes.values()].some(v => v < 10);
    return { billingProjects: projects, billingProjectsHasOther: hasOther };
  }, [trendData, userBilling]);

  const bpOther = useCallback(
    (byProject: Map<string, number>) =>
      [...byProject].filter(([p]) => !billingProjects.includes(p)).reduce((s, [, v]) => s + v, 0),
    [billingProjects]
  );

  const { userComputeProducts, userComputeHasOther } = useMemo(() => {
    const maxes = new Map<string, number>();
    for (const d of trendData) {
      for (const [p, v] of d.user_compute_by_product) maxes.set(p, Math.max(maxes.get(p) ?? 0, v));
    }
    const products = [...maxes.keys()].filter(p => (maxes.get(p) ?? 0) >= 10).sort();
    const hasOther = [...maxes.values()].some(v => v < 10);
    return { userComputeProducts: products, userComputeHasOther: hasOther };
  }, [trendData]);

  const ucOther = useCallback(
    (byProduct: Map<string, number>) =>
      [...byProduct].filter(([p]) => !userComputeProducts.includes(p)).reduce((s, [, v]) => s + v, 0),
    [userComputeProducts]
  );

  const fieldGroups = useMemo<FieldGroup[]>(() => {
    const productMax = new Map<string, number>();
    const resourceMax = new Map<string, number>();
    const projectMax = new Map<string, number>();
    const track = (maxes: Map<string, number>, entries: Map<string, number>) => {
      for (const [k, v] of entries) maxes.set(k, Math.max(maxes.get(k) ?? 0, v));
    };
    if (cloudCosts) {
      track(productMax, cloudCosts.user_compute_by_product);
    }
    if (userBilling) {
      track(resourceMax, userBilling.resource_by_type);
      track(projectMax, userBilling.billing_by_project);
    }
    if (compareUserBilling) track(projectMax, compareUserBilling.billing_by_project);
    trendData.forEach(d => {
      track(productMax, d.user_compute_by_product);
      track(resourceMax, d.resource_by_type);
      track(projectMax, d.billing_by_project);
    });
    const over10 = (maxes: Map<string, number>) => [...maxes.keys()].filter(k => (maxes.get(k) ?? 0) >= 10).sort();
    return buildFieldGroups(over10(productMax), overheadServices, overheadSkusByService, over10(resourceMax), over10(projectMax));
  }, [cloudCosts, userBilling, compareUserBilling, trendData, overheadServices, overheadSkusByService]);

  const cloudSeriesKeys = useMemo(() => {
    if (cloudView === 'user_compute') return [...userComputeProducts, ...(userComputeHasOther ? ['(Other)'] : [])];
    if (cloudView === 'other_compute') return ['batch_test', 'batch_dev', 'unknown'];
    if (cloudView === 'k8s') return ['k8s_nodes', 'k8s_mgmt'];
    if (overheadServices.includes(cloudView)) return overheadSkusByService.get(cloudView) ?? [];
    return ['user_compute', 'other_compute', 'k8s', ...overheadServices];
  }, [cloudView, userComputeProducts, userComputeHasOther, overheadServices, overheadSkusByService]);

  const cloudBaseData = useMemo(() => {
    if (cloudView === 'user_compute')
      return trendData.map(d => ({ month: d.month, ...Object.fromEntries(userComputeProducts.map(p => [p, d.user_compute_by_product.get(p) ?? 0])), ...(userComputeHasOther ? { '(Other)': ucOther(d.user_compute_by_product) } : {}) }));
    if (cloudView === 'other_compute' || cloudView === 'k8s') return trendData as unknown[];
    if (overheadServices.includes(cloudView)) {
      const skus = overheadSkusByService.get(cloudView) ?? [];
      return trendData.map(d => ({ month: d.month, ...Object.fromEntries(skus.map(sku => [sku, d.overhead_by_sku.get(cloudView)?.get(sku) ?? 0])) }));
    }
    // summary: flatten overhead services into data
    return trendData.map(d => ({
      month: d.month,
      user_compute: d.user_compute,
      other_compute: d.other_compute,
      k8s: d.k8s,
      ...Object.fromEntries(overheadServices.map(svc => [svc, d.non_compute_services.get(svc) ?? 0])),
    }));
  }, [cloudView, trendData, userComputeProducts, userComputeHasOther, ucOther, overheadServices, overheadSkusByService]);

  const cloudChartData = useMemo(
    () => cloudShowPct ? toPctRows(cloudBaseData as Record<string, unknown>[], cloudSeriesKeys) : cloudBaseData,
    [cloudShowPct, cloudBaseData, cloudSeriesKeys]
  );

  const billingSeriesKeys = useMemo(() => {
    if (billingView === 'resource_usage') return [...billingResources, ...(billingResourcesHasOther ? ['(Other)'] : [])];
    if (billingView === 'billing_project') return [...billingProjects, ...(billingProjectsHasOther ? ['(Other)'] : [])];
    return ['resource_cost', 'service_fees'];
  }, [billingView, billingResources, billingResourcesHasOther, billingProjects, billingProjectsHasOther]);

  const billingBaseData = useMemo(() => {
    if (billingView === 'resource_usage')
      return trendData.map(d => ({ month: d.month, ...Object.fromEntries(billingResources.map(r => [r, d.resource_by_type.get(r) ?? 0])), ...(billingResourcesHasOther ? { '(Other)': brOther(d.resource_by_type) } : {}) }));
    if (billingView === 'billing_project')
      return trendData.map(d => ({ month: d.month, ...Object.fromEntries(billingProjects.map(p => [p, d.billing_by_project.get(p) ?? 0])), ...(billingProjectsHasOther ? { '(Other)': bpOther(d.billing_by_project) } : {}) }));
    return trendData as unknown[];
  }, [billingView, trendData, billingResources, billingResourcesHasOther, brOther, billingProjects, billingProjectsHasOther, bpOther]);

  const billingChartData = useMemo(
    () => billingShowPct ? toPctRows(billingBaseData as Record<string, unknown>[], billingSeriesKeys) : billingBaseData,
    [billingShowPct, billingBaseData, billingSeriesKeys]
  );

  const OVERHEAD_PALETTE = ['#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd', '#818cf8', '#4f46e5', '#7c3aed', '#9333ea', '#a855f7', '#c026d3'];
  const overheadServiceColor = (svc: string) => OVERHEAD_PALETTE[overheadServices.indexOf(svc) % OVERHEAD_PALETTE.length];

  const overheadServicesMonthly = useMemo(() => {
    if (!cloudCosts) return [] as string[];
    return [...cloudCosts.non_compute_services].filter(([, v]) => v >= 10).sort(([, a], [, b]) => b - a).map(([s]) => s);
  }, [cloudCosts]);

  const overheadAllKeys = useMemo(() => {
    if (overheadServices.includes(cloudView)) return overheadSkusByService.get(cloudView) ?? [];
    return [...overheadServices];
  }, [cloudView, overheadServices, overheadSkusByService]);
  const overheadToggle = useLegendToggle(overheadAllKeys);
  const onOverheadLegendClick = overheadToggle.onLegendClick;
  const isOverheadHidden = overheadToggle.isHidden;
  const setOverheadHidden = overheadToggle.setHidden;

  const onSummaryCloudLegendClick = useCallback(
    (e: { dataKey?: string | number | ((_obj: unknown) => unknown) }, index: number, event: { shiftKey: boolean }) => {
      if (typeof e.dataKey !== 'string') return;
      const key = e.dataKey;
      const fixedKeys = ['user_compute', 'other_compute', 'k8s'] as const;
      const isFixed = (fixedKeys as readonly string[]).includes(key);
      if (event.shiftKey) {
        // shift-click: toggle just this item in its own group
        if (isFixed) cloudCostsToggle.onLegendClick(e, index, event);
        else onOverheadLegendClick(e, index, event);
      } else {
        // click: solo across both groups, or restore all if already soloed
        const visibleFixed = fixedKeys.filter(k => !cloudCostsToggle.isHidden(k));
        const visibleOverhead = overheadServices.filter(k => !isOverheadHidden(k));
        const isSolo = visibleFixed.length + visibleOverhead.length === 1 &&
          (visibleFixed[0] === key || visibleOverhead[0] === key);
        if (isSolo) {
          cloudCostsToggle.setHidden(new Set());
          setOverheadHidden(new Set());
        } else {
          cloudCostsToggle.setHidden(new Set(fixedKeys.filter(k => k !== key)));
          setOverheadHidden(new Set(overheadServices.filter(k => k !== key)));
        }
      }
    },
    [cloudCostsToggle, onOverheadLegendClick, overheadServices, isOverheadHidden, setOverheadHidden]
  );

  const BILLING_RESOURCE_PALETTE = ['#10b981', '#059669', '#34d399', '#047857', '#6ee7b7', '#065f46', '#a7f3d0', '#14b8a6', '#0d9488', '#2dd4bf'];
  const billingResourceColor = (r: string) => r === '(Other)' ? '#9ca3af' : BILLING_RESOURCE_PALETTE[billingResources.indexOf(r) % BILLING_RESOURCE_PALETTE.length];

  const BILLING_PROJECT_PALETTE = ['#f97316', '#ea580c', '#fb923c', '#c2410c', '#fdba74', '#9a3412', '#fed7aa', '#d97706', '#fbbf24', '#b45309'];
  const billingProjectColor = (p: string) => p === '(Other)' ? '#9ca3af' : BILLING_PROJECT_PALETTE[billingProjects.indexOf(p) % BILLING_PROJECT_PALETTE.length];
  // Literal Tailwind classes matching BILLING_PROJECT_PALETTE by index — a dynamic `text-[${hex}]`
  // arbitrary-value class isn't discoverable by Tailwind's static content scan, so it never gets
  // generated into the compiled CSS. Keep this in sync with BILLING_PROJECT_PALETTE above.
  const BILLING_PROJECT_TEXT_CLASSES = ['text-orange-500', 'text-orange-600', 'text-orange-400', 'text-orange-700', 'text-orange-300', 'text-orange-800', 'text-orange-200', 'text-amber-600', 'text-amber-400', 'text-amber-700'];

  const billingProjectAllKeys = useMemo(
    () => [...billingProjects, ...(billingProjectsHasOther ? ['(Other)'] : [])],
    [billingProjects, billingProjectsHasOther]
  );
  const billingProjectToggle = useLegendToggle(billingProjectAllKeys);
  const onBillingProjectLegendClick = billingProjectToggle.onLegendClick;
  const isBillingProjectHidden = billingProjectToggle.isHidden;

  const billingResourceAllKeys = useMemo(
    () => [...billingResources, ...(billingResourcesHasOther ? ['(Other)'] : [])],
    [billingResources, billingResourcesHasOther]
  );
  const billingResourceToggle = useLegendToggle(billingResourceAllKeys);
  const onBillingResourceLegendClick = billingResourceToggle.onLegendClick;
  const isBillingResourceHidden = billingResourceToggle.isHidden;

  const USER_COMPUTE_PALETTE = ['#0ea5e9', '#38bdf8', '#7dd3fc', '#bae6fd', '#0284c7', '#0369a1', '#075985', '#0c4a6e', '#22d3ee', '#06b6d4'];
  const userComputeProductColor = (p: string) => p === '(Other)' ? '#9ca3af' : USER_COMPUTE_PALETTE[userComputeProducts.indexOf(p) % USER_COMPUTE_PALETTE.length];

  const userComputeAllKeys = useMemo(
    () => [...userComputeProducts, ...(userComputeHasOther ? ['(Other)'] : [])],
    [userComputeProducts, userComputeHasOther]
  );
  const userComputeToggle = useLegendToggle(userComputeAllKeys);
  const onUserComputeLegendClick = userComputeToggle.onLegendClick;
  const isUserComputeHidden = userComputeToggle.isHidden;

  const cloudYMax = Math.max(
    0,
    ...trendData.map(d =>
      cloudView === 'user_compute'
        ? userComputeProducts.reduce((s, p) => s + (isUserComputeHidden(p) ? 0 : (d.user_compute_by_product.get(p) ?? 0)), 0) +
          (userComputeHasOther && !isUserComputeHidden('(Other)') ? ucOther(d.user_compute_by_product) : 0)
        : cloudView === 'other_compute'
          ? (otherComputeToggle.isHidden('batch_test') ? 0 : d.batch_test) +
            (otherComputeToggle.isHidden('batch_dev') ? 0 : d.batch_dev) +
            (otherComputeToggle.isHidden('unknown') ? 0 : d.unknown)
          : cloudView === 'k8s'
            ? (k8sToggle.isHidden('k8s_nodes') ? 0 : d.k8s_nodes) +
              (k8sToggle.isHidden('k8s_mgmt') ? 0 : d.k8s_mgmt)
            : overheadServices.includes(cloudView)
              ? (overheadSkusByService.get(cloudView) ?? []).reduce((s, sku) => s + (isOverheadHidden(sku) ? 0 : d.overhead_by_sku.get(cloudView)?.get(sku) ?? 0), 0)
              : (cloudCostsToggle.isHidden('user_compute') ? 0 : d.user_compute) +
                (cloudCostsToggle.isHidden('other_compute') ? 0 : d.other_compute) +
                (cloudCostsToggle.isHidden('k8s') ? 0 : d.k8s) +
                overheadServices.reduce((s, svc) => s + (isOverheadHidden(svc) ? 0 : (d.non_compute_services.get(svc) ?? 0)), 0)
    ),
  );
  const billingYMax = Math.max(
    0,
    ...trendData.map(d =>
      billingView === 'resource_usage'
        ? billingResources.reduce((s, r) => s + (isBillingResourceHidden(r) ? 0 : (d.resource_by_type.get(r) ?? 0)), 0) +
          (billingResourcesHasOther && !isBillingResourceHidden('(Other)') ? brOther(d.resource_by_type) : 0)
        : billingView === 'billing_project'
          ? billingProjects.reduce((s, p) => s + (isBillingProjectHidden(p) ? 0 : (d.billing_by_project.get(p) ?? 0)), 0) +
            (billingProjectsHasOther && !isBillingProjectHidden('(Other)') ? bpOther(d.billing_by_project) : 0)
          : (billingToggle.isHidden('resource_cost') ? 0 : d.resource_cost) +
            (billingToggle.isHidden('service_fees') ? 0 : d.service_fees)
    ),
  );

  const profitYExtent = Math.max(0, ...trendData.map(d => Math.abs(d.profit)));

  const cloudStats = computeStats(trendData.map(d =>
    cloudView === 'user_compute'
      ? userComputeProducts.reduce((s, p) => s + (isUserComputeHidden(p) ? 0 : (d.user_compute_by_product.get(p) ?? 0)), 0) +
        (userComputeHasOther && !isUserComputeHidden('(Other)') ? ucOther(d.user_compute_by_product) : 0)
      : cloudView === 'other_compute'
        ? (otherComputeToggle.isHidden('batch_test') ? 0 : d.batch_test) +
          (otherComputeToggle.isHidden('batch_dev') ? 0 : d.batch_dev) +
          (otherComputeToggle.isHidden('unknown') ? 0 : d.unknown)
        : cloudView === 'k8s'
          ? (k8sToggle.isHidden('k8s_nodes') ? 0 : d.k8s_nodes) +
            (k8sToggle.isHidden('k8s_mgmt') ? 0 : d.k8s_mgmt)
          : overheadServices.includes(cloudView)
            ? (overheadSkusByService.get(cloudView) ?? []).reduce((s, sku) => s + (isOverheadHidden(sku) ? 0 : d.overhead_by_sku.get(cloudView)?.get(sku) ?? 0), 0)
            : (cloudCostsToggle.isHidden('user_compute') ? 0 : d.user_compute) +
              (cloudCostsToggle.isHidden('other_compute') ? 0 : d.other_compute) +
              (cloudCostsToggle.isHidden('k8s') ? 0 : d.k8s) +
              overheadServices.reduce((s, svc) => s + (isOverheadHidden(svc) ? 0 : (d.non_compute_services.get(svc) ?? 0)), 0)
  ));
  const cloudSeriesStats: Record<string, { mean: number; std: number } | null> = cloudView === 'user_compute'
    ? {
        ...Object.fromEntries(userComputeProducts.map(p => [p, computeStats(trendData.map(d => d.user_compute_by_product.get(p) ?? 0))])),
        ...(userComputeHasOther ? { '(Other)': computeStats(trendData.map(d => ucOther(d.user_compute_by_product))) } : {}),
      }
    : cloudView === 'other_compute'
      ? {
          batch_test: computeStats(trendData.map(d => d.batch_test)),
          batch_dev: computeStats(trendData.map(d => d.batch_dev)),
          unknown: computeStats(trendData.map(d => d.unknown)),
        }
      : cloudView === 'k8s'
        ? {
            k8s_nodes: computeStats(trendData.map(d => d.k8s_nodes)),
            k8s_mgmt: computeStats(trendData.map(d => d.k8s_mgmt)),
          }
        : overheadServices.includes(cloudView)
          ? Object.fromEntries((overheadSkusByService.get(cloudView) ?? []).map(sku => [sku, computeStats(trendData.map(d => d.overhead_by_sku.get(cloudView)?.get(sku) ?? 0))]))
          : {
              user_compute: computeStats(trendData.map(d => d.user_compute)),
              other_compute: computeStats(trendData.map(d => d.other_compute)),
              k8s: computeStats(trendData.map(d => d.k8s)),
              ...Object.fromEntries(overheadServices.map(svc => [svc, computeStats(trendData.map(d => d.non_compute_services.get(svc) ?? 0))])),
            };
  const billingStats = computeStats(trendData.map(d =>
    billingView === 'resource_usage'
      ? billingResources.reduce((s, r) => s + (isBillingResourceHidden(r) ? 0 : (d.resource_by_type.get(r) ?? 0)), 0) +
        (billingResourcesHasOther && !isBillingResourceHidden('(Other)') ? brOther(d.resource_by_type) : 0)
      : billingView === 'billing_project'
        ? billingProjects.reduce((s, p) => s + (isBillingProjectHidden(p) ? 0 : (d.billing_by_project.get(p) ?? 0)), 0) +
          (billingProjectsHasOther && !isBillingProjectHidden('(Other)') ? bpOther(d.billing_by_project) : 0)
        : (billingToggle.isHidden('resource_cost') ? 0 : d.resource_cost) +
          (billingToggle.isHidden('service_fees') ? 0 : d.service_fees)
  ));
  const billingSeriesStats: Record<string, { mean: number; std: number } | null> = billingView === 'resource_usage'
    ? {
        ...Object.fromEntries(billingResources.map(r => [r, computeStats(trendData.map(d => d.resource_by_type.get(r) ?? 0))])),
        ...(billingResourcesHasOther ? { '(Other)': computeStats(trendData.map(d => brOther(d.resource_by_type))) } : {}),
      }
    : billingView === 'billing_project'
      ? {
          ...Object.fromEntries(billingProjects.map(p => [p, computeStats(trendData.map(d => d.billing_by_project.get(p) ?? 0))])),
          ...(billingProjectsHasOther ? { '(Other)': computeStats(trendData.map(d => bpOther(d.billing_by_project))) } : {}),
        }
      : {
          resource_cost: computeStats(trendData.map(d => d.resource_cost)),
          service_fees: computeStats(trendData.map(d => d.service_fees)),
        };
  const rowNum = (row: Map<string, unknown>, k: string) => { const v = row.get(k); return typeof v === 'number' ? v : 0; };
  const toRowMaps = (rows: unknown[]) => (rows as Record<string, unknown>[]).map(row => new Map(Object.entries(row)));

  const cloudPctRows = toRowMaps(cloudChartData);
  const cloudPctStats = computeStats(cloudPctRows.map(row =>
    cloudView === 'user_compute'
      ? [...userComputeProducts.filter(p => !isUserComputeHidden(p)), ...(userComputeHasOther && !isUserComputeHidden('(Other)') ? ['(Other)'] : [])].reduce((s, k) => s + rowNum(row, k), 0)
      : cloudView === 'other_compute'
        ? (['batch_test', 'batch_dev', 'unknown'] as const).filter(k => !otherComputeToggle.isHidden(k)).reduce((s, k) => s + rowNum(row, k), 0)
        : cloudView === 'k8s'
          ? (['k8s_nodes', 'k8s_mgmt'] as const).filter(k => !k8sToggle.isHidden(k)).reduce((s, k) => s + rowNum(row, k), 0)
          : overheadServices.includes(cloudView)
            ? (overheadSkusByService.get(cloudView) ?? []).filter(s => !isOverheadHidden(s)).reduce((s, k) => s + rowNum(row, k), 0)
            : [...['user_compute', 'other_compute', 'k8s'].filter(k => !cloudCostsToggle.isHidden(k)), ...overheadServices.filter(s => !isOverheadHidden(s))].reduce((s, k) => s + rowNum(row, k), 0)
  ));
  const cloudPctSeriesStats: Record<string, { mean: number; std: number } | null> = cloudView === 'user_compute'
    ? { ...Object.fromEntries(userComputeProducts.map(p => [p, computeStats(cloudPctRows.map(row => rowNum(row, p)))])), ...(userComputeHasOther ? { '(Other)': computeStats(cloudPctRows.map(row => rowNum(row, '(Other)'))) } : {}) }
    : cloudView === 'other_compute'
      ? { batch_test: computeStats(cloudPctRows.map(row => rowNum(row, 'batch_test'))), batch_dev: computeStats(cloudPctRows.map(row => rowNum(row, 'batch_dev'))), unknown: computeStats(cloudPctRows.map(row => rowNum(row, 'unknown'))) }
      : cloudView === 'k8s'
        ? { k8s_nodes: computeStats(cloudPctRows.map(row => rowNum(row, 'k8s_nodes'))), k8s_mgmt: computeStats(cloudPctRows.map(row => rowNum(row, 'k8s_mgmt'))) }
        : overheadServices.includes(cloudView)
          ? Object.fromEntries((overheadSkusByService.get(cloudView) ?? []).map(sku => [sku, computeStats(cloudPctRows.map(row => rowNum(row, sku)))]))
          : { user_compute: computeStats(cloudPctRows.map(row => rowNum(row, 'user_compute'))), other_compute: computeStats(cloudPctRows.map(row => rowNum(row, 'other_compute'))), k8s: computeStats(cloudPctRows.map(row => rowNum(row, 'k8s'))), ...Object.fromEntries(overheadServices.map(svc => [svc, computeStats(cloudPctRows.map(row => rowNum(row, svc)))])) };

  const billingPctRows = toRowMaps(billingChartData);
  const billingPctStats = computeStats(billingPctRows.map(row =>
    billingView === 'resource_usage'
      ? [...billingResources.filter(r => !isBillingResourceHidden(r)), ...(billingResourcesHasOther && !isBillingResourceHidden('(Other)') ? ['(Other)'] : [])].reduce((s, k) => s + rowNum(row, k), 0)
      : billingView === 'billing_project'
        ? [...billingProjects.filter(p => !isBillingProjectHidden(p)), ...(billingProjectsHasOther && !isBillingProjectHidden('(Other)') ? ['(Other)'] : [])].reduce((s, k) => s + rowNum(row, k), 0)
        : (['resource_cost', 'service_fees'] as const).filter(k => !billingToggle.isHidden(k)).reduce((s, k) => s + rowNum(row, k), 0)
  ));
  const billingPctSeriesStats: Record<string, { mean: number; std: number } | null> = billingView === 'resource_usage'
    ? { ...Object.fromEntries(billingResources.map(r => [r, computeStats(billingPctRows.map(row => rowNum(row, r)))])), ...(billingResourcesHasOther ? { '(Other)': computeStats(billingPctRows.map(row => rowNum(row, '(Other)'))) } : {}) }
    : billingView === 'billing_project'
      ? { ...Object.fromEntries(billingProjects.map(p => [p, computeStats(billingPctRows.map(row => rowNum(row, p)))])), ...(billingProjectsHasOther ? { '(Other)': computeStats(billingPctRows.map(row => rowNum(row, '(Other)'))) } : {}) }
      : { resource_cost: computeStats(billingPctRows.map(row => rowNum(row, 'resource_cost'))), service_fees: computeStats(billingPctRows.map(row => rowNum(row, 'service_fees'))) };

  const profitStats = computeStats(trendData.map(d => d.profit));
  const coreHoursData = useMemo(() => trendData.map(d => ({ month: d.month, core_hours: d.service_fees * 100 })), [trendData]);
  const coreHoursStats = useMemo(() => computeStats(coreHoursData.map(d => d.core_hours)), [coreHoursData]);
  const coreHoursExtent = Math.max(0, ...coreHoursData.map(d => d.core_hours));

  const customRatioChartData = useMemo(
    () => trendData.map(d => {
      const den = resolveTrend(customRatioDen, d);
      const value = den === 0 ? null : (resolveTrend(customRatioNum, d) / den) * 100;
      return { month: d.month, value };
    }),
    [trendData, customRatioNum, customRatioDen]
  );
  const customRatioStats = useMemo(
    () => computeStats(customRatioChartData.map(p => p.value).filter((v): v is number => v !== null)),
    [customRatioChartData]
  );
  const scatterChartData = useMemo(
    () => trendData.map(d => ({ month: d.month, x: resolveTrend(scatterX, d), y: resolveTrend(scatterY, d) })),
    [trendData, scatterX, scatterY]
  );
  const scatterRegression = useMemo(() => computeRegression(scatterChartData), [scatterChartData]);
  const regressionLineData = useMemo(() => {
    if (!scatterRegression || scatterChartData.length < 2) return [];
    const xs = scatterChartData.map(d => d.x);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    return [
      { x: xMin, y: scatterRegression.slope * xMin + scatterRegression.intercept },
      { x: xMax, y: scatterRegression.slope * xMax + scatterRegression.intercept },
    ];
  }, [scatterRegression, scatterChartData]);

  const fetchData = useCallback(async (period: string) => {
    setLoading(true);
    setCloudCosts(null);
    setUserBilling(null);
    setCloudError(null);
    setBillingError(null);

    const [cloudResult, billingResult] = await Promise.allSettled([
      fetchCloudCosts(api, period),
      fetchUserBilling(api, period),
    ]);

    if (cloudResult.status === 'fulfilled') setCloudCosts(cloudResult.value);
    else setCloudError(cloudResult.reason instanceof Error ? cloudResult.reason.message : 'Failed to load cloud costs.');

    if (billingResult.status === 'fulfilled') setUserBilling(billingResult.value);
    else setBillingError(billingResult.reason instanceof Error ? billingResult.reason.message : 'Failed to load user billing.');

    setLoading(false);
  }, [api]);

  useEffect(() => { void fetchData(timePeriod); }, [fetchData, timePeriod]);

  useEffect(() => {
    const url = new URL(window.location.href);
    url.searchParams.set('month', timePeriod);
    if (compareTimePeriod) url.searchParams.set('comparison_month', compareTimePeriod);
    else url.searchParams.delete('comparison_month');
    url.searchParams.set('trends_start', trendsStart);
    url.searchParams.set('trends_end', trendsEnd);
    window.history.replaceState(null, '', url.toString());
  }, [timePeriod, compareTimePeriod, trendsStart, trendsEnd]);

  useEffect(() => {
    if (!compareTimePeriod) {
      setCompareCloudCosts(null);
      setCompareUserBilling(null);
      setCompareCloudError(null);
      setCompareBillingError(null);
      return;
    }
    setCompareLoading(true);
    setCompareCloudCosts(null);
    setCompareUserBilling(null);
    setCompareCloudError(null);
    setCompareBillingError(null);
    void Promise.allSettled([
      fetchCloudCosts(api, compareTimePeriod),
      fetchUserBilling(api, compareTimePeriod),
    ]).then(([cloudResult, billingResult]) => {
      if (cloudResult.status === 'fulfilled') setCompareCloudCosts(cloudResult.value);
      else setCompareCloudError(cloudResult.reason instanceof Error ? cloudResult.reason.message : 'Failed to load cloud costs.');
      if (billingResult.status === 'fulfilled') setCompareUserBilling(billingResult.value);
      else setCompareBillingError(billingResult.reason instanceof Error ? billingResult.reason.message : 'Failed to load user billing.');
      setCompareLoading(false);
    });
  }, [compareTimePeriod, api]);

  const fetchTrendPoint = useCallback(async (m: string): Promise<MonthDataPoint> => {
    // Deliberately uses Promise.all (not allSettled): a failed fetch for any month must fail
    // the whole trend load rather than being silently recorded as a $0 month.
    const [c, b] = await Promise.all([
      fetchCloudCosts(api, m),
      fetchUserBilling(api, m),
    ]);
    const overhead = c.other_compute + [...c.non_compute_services.values()].reduce((a, bv) => a + bv, 0);
    return {
      month: monthParamToLabel(m),
      cloud_total: c.total,
      user_compute: c.user_compute,
      other_compute: c.other_compute,
      batch_test: c.batch_test,
      batch_dev: c.batch_dev,
      unknown: c.unknown,
      k8s: c.k8s,
      k8s_nodes: c.k8s_nodes,
      k8s_mgmt: c.k8s_mgmt,
      non_compute_services: c.non_compute_services,
      overhead_by_sku: c.overhead_by_sku,
      user_compute_by_product: c.user_compute_by_product,
      resource_by_type: b.resource_by_type,
      billing_by_project: b.billing_by_project,
      billing_project_count: b.billing_project_count,
      billing_project_concentration: b.billing_project_concentration,
      user_billing: b.total,
      service_fees: b.service_fee_cost,
      resource_cost: b.resource_cost,
      profit: b.total - c.total,
      svc_fee_overhead_pct: overhead > 0 ? (b.service_fee_cost / overhead) * 100 : null,
      resource_billing_pct: c.user_compute > 0 ? (b.resource_cost / c.user_compute) * 100 : null,
      svc_fee_bill_pct: b.total > 0 ? (b.service_fee_cost / b.total) * 100 : null,
      overhead_cloud_pct: c.total > 0 ? (overhead / c.total) * 100 : null,
      overhead_resource_pct: b.resource_cost > 0 ? (overhead / b.resource_cost) * 100 : null,
    };
  }, [api]);

  const fetchTrends = useCallback(async (start: string, end: string) => {
    setTrendsLoading(true);
    setTrendsError(null);
    const months = monthsBetween(start, end);
    // Fetch a bounded number of months concurrently at a time (2 requests/month) rather than
    // firing every request in the range at once, which could spike load on the backend APIs
    // for a wide date range.
    const points: MonthDataPoint[] = [];
    try {
      for (let i = 0; i < months.length; i += TRENDS_FETCH_BATCH_SIZE) {
        const batch = months.slice(i, i + TRENDS_FETCH_BATCH_SIZE);
        points.push(...await Promise.all(batch.map(fetchTrendPoint)));
      }
      setTrendData(points);
    } catch (e) {
      setTrendData([]);
      setTrendsError(e instanceof Error ? e.message : 'Failed to load trend data.');
    }
    setTrendsLoading(false);
  }, [fetchTrendPoint]);

  const renderCloudBody = (
    costs: CloudCosts | null,
    err: string | null,
    ldg: boolean,
    period: string,
    compact: boolean,
    baseCosts?: CloudCosts | null,
  ) => {
    if (err) return <p className="text-red-500 text-sm py-2">{err}</p>;
    if (!costs) return ldg ? null : <p className="text-zinc-400 text-sm py-2">No data for {period}.</p>;

    const d = (v: number, baseV: number) => baseCosts != null ? v - baseV : undefined;

    const mProducts = [...costs.user_compute_by_product]
      .filter(([, v]) => v >= 10).sort(([, a], [, b]) => b - a).map(([p]) => p);
    const hasOtherP = [...costs.user_compute_by_product.values()].some(v => v < 10);
    const ucOtherCost = [...costs.user_compute_by_product]
      .filter(([p]) => !mProducts.includes(p)).reduce((s, [, v]) => s + v, 0);
    const mOverheadSvcs = [...costs.non_compute_services].filter(([, v]) => v >= 10).sort(([, a], [, b]) => b - a).map(([s]) => s);

    const rows = (() => {
      if (cloudView === 'summary') return (
        <>
          {([
            { label: 'Compute (user-driven)', value: costs.user_compute, baseV: baseCosts?.user_compute ?? 0 },
            { label: 'Compute (other)', value: costs.other_compute, baseV: baseCosts?.other_compute ?? 0 },
            { label: 'K8s', value: costs.k8s, baseV: baseCosts?.k8s ?? 0 },
            ...mOverheadSvcs.map(svc => ({ label: svc, value: costs.non_compute_services.get(svc) ?? 0, baseV: baseCosts?.non_compute_services.get(svc) ?? 0 })),
          ]).sort((a, b) => b.value - a.value).map(r => (
            <CostRow key={r.label} label={r.label} value={r.value} pctStr={pct(r.value, costs.total)} delta={d(r.value, r.baseV)} />
          ))}
          <CostRow label="Total" value={costs.total} bold delta={d(costs.total, baseCosts?.total ?? 0)} />
        </>
      );
      if (cloudView === 'user_compute') return (
        <>
          {mProducts.map(p => {
            const v = costs.user_compute_by_product.get(p) ?? 0;
            return <CostRow key={p} label={p} value={v} pctStr={pct(v, costs.user_compute)} indent delta={d(v, baseCosts?.user_compute_by_product.get(p) ?? 0)} />;
          })}
          {hasOtherP && ucOtherCost > 0 && (() => {
            const baseOther = baseCosts ? [...baseCosts.user_compute_by_product].filter(([p]) => !mProducts.includes(p)).reduce((s, [, v]) => s + v, 0) : 0;
            return <CostRow label="(Other)" value={ucOtherCost} pctStr={pct(ucOtherCost, costs.user_compute)} indent delta={d(ucOtherCost, baseOther)} />;
          })()}
          <CostRow label="Compute (user-driven) total" value={costs.user_compute} bold delta={d(costs.user_compute, baseCosts?.user_compute ?? 0)} />
        </>
      );
      if (cloudView === 'other_compute') return (
        <>
          {([
            { label: 'CI / test batches', value: costs.batch_test, baseV: baseCosts?.batch_test ?? 0 },
            { label: 'Dev batches', value: costs.batch_dev, baseV: baseCosts?.batch_dev ?? 0 },
            { label: 'Unknown / unlabeled', value: costs.unknown, baseV: baseCosts?.unknown ?? 0 },
          ] as const).slice().sort((a, b) => b.value - a.value).map(r => (
            <CostRow key={r.label} label={r.label} value={r.value} pctStr={pct(r.value, costs.other_compute)} indent delta={d(r.value, r.baseV)} />
          ))}
          <CostRow label="Compute (other) total" value={costs.other_compute} bold delta={d(costs.other_compute, baseCosts?.other_compute ?? 0)} />
        </>
      );
      if (cloudView === 'k8s') return (
        <>
          {([
            { label: 'Compute nodes', value: costs.k8s_nodes, baseV: baseCosts?.k8s_nodes ?? 0 },
            { label: 'Management fee', value: costs.k8s_mgmt, baseV: baseCosts?.k8s_mgmt ?? 0 },
          ] as const).slice().sort((a, b) => b.value - a.value).map(r => (
            <CostRow key={r.label} label={r.label} value={r.value} pctStr={pct(r.value, costs.k8s)} indent delta={d(r.value, r.baseV)} />
          ))}
          <CostRow label="K8s total" value={costs.k8s} bold delta={d(costs.k8s, baseCosts?.k8s ?? 0)} />
        </>
      );
      if (overheadServicesMonthly.includes(cloudView)) {
        const skuMap = costs.overhead_by_sku.get(cloudView) ?? new Map<string, number>();
        const mSkus = [...skuMap].filter(([, v]) => v >= 10).sort(([, a], [, b]) => b - a).map(([s]) => s);
        const hasOtherSku = [...skuMap.values()].some(v => v < 10);
        const svcTotal = costs.non_compute_services.get(cloudView) ?? 0;
        const otherSkuCost = [...skuMap].filter(([s]) => !mSkus.includes(s)).reduce((a, [, v]) => a + v, 0);
        return (
          <>
            {mSkus.map(sku => {
              const v = skuMap.get(sku) ?? 0;
              return <CostRow key={sku} label={sku} value={v} pctStr={pct(v, svcTotal)} indent delta={d(v, baseCosts?.overhead_by_sku.get(cloudView)?.get(sku) ?? 0)} />;
            })}
            {hasOtherSku && otherSkuCost > 0 && (() => {
              const baseSkuMap = baseCosts?.overhead_by_sku.get(cloudView) ?? new Map<string, number>();
              const baseOther = baseCosts ? [...baseSkuMap].filter(([s]) => !mSkus.includes(s)).reduce((a, [, v]) => a + v, 0) : 0;
              return <CostRow label="(Other)" value={otherSkuCost} pctStr={pct(otherSkuCost, svcTotal)} indent delta={d(otherSkuCost, baseOther)} />;
            })()}
            <CostRow label={`${cloudView} total`} value={svcTotal} bold delta={d(svcTotal, baseCosts?.non_compute_services.get(cloudView) ?? 0)} />
          </>
        );
      }
      return null;
    })();

    const pieData: PieSlice[] = (() => {
      if (cloudView === 'summary') return [
        { name: 'Compute (user-driven)', value: costs.user_compute, fill: '#0ea5e9' },
        { name: 'Compute (other)', value: costs.other_compute, fill: '#f59e0b' },
        { name: 'K8s', value: costs.k8s, fill: '#10b981' },
        ...mOverheadSvcs.map((svc, i) => ({ name: svc, value: costs.non_compute_services.get(svc) ?? 0, fill: OVERHEAD_PALETTE[i % OVERHEAD_PALETTE.length] })),
      ];
      if (cloudView === 'user_compute') return [
        ...mProducts.map((p, i) => ({ name: p, value: costs.user_compute_by_product.get(p) ?? 0, fill: USER_COMPUTE_PALETTE[i % USER_COMPUTE_PALETTE.length] })),
        ...(hasOtherP ? [{ name: '(Other)', value: ucOtherCost, fill: '#9ca3af' }] : []),
      ];
      if (cloudView === 'other_compute') return [
        { name: 'CI / test batches', value: costs.batch_test, fill: '#f59e0b' },
        { name: 'Dev batches', value: costs.batch_dev, fill: '#fcd34d' },
        { name: 'Unknown / unlabeled', value: costs.unknown, fill: '#fef3c7' },
      ];
      if (cloudView === 'k8s') return [
        { name: 'Compute nodes', value: costs.k8s_nodes, fill: '#059669' },
        { name: 'Management fee', value: costs.k8s_mgmt, fill: '#6ee7b7' },
      ];
      if (overheadServicesMonthly.includes(cloudView)) {
        const skuMap = costs.overhead_by_sku.get(cloudView) ?? new Map<string, number>();
        const mSkus = [...skuMap].filter(([, v]) => v >= 10).sort(([, a], [, b]) => b - a).map(([s]) => s);
        const otherSkuCost = [...skuMap].filter(([s]) => !mSkus.includes(s)).reduce((a, [, v]) => a + v, 0);
        return [
          ...mSkus.map((sku, i) => ({ name: sku, value: skuMap.get(sku) ?? 0, fill: OVERHEAD_PALETTE[i % OVERHEAD_PALETTE.length] })),
          ...(otherSkuCost > 0 ? [{ name: '(Other)', value: otherSkuCost, fill: '#9ca3af' }] : []),
        ];
      }
      return [];
    })();

    return (
      <div className="flex items-center gap-4">
        <div className="flex-1">{rows}</div>
        <div className={`${compact ? 'w-28' : 'w-40'} flex-shrink-0`}>
          <MiniPieChart data={pieData} size={compact ? 'sm' : 'md'} format={fmt} />
        </div>
      </div>
    );
  };

  const renderBillingBody = (
    billing: UserBilling | null,
    err: string | null,
    ldg: boolean,
    period: string,
    compact: boolean,
    baseBilling?: UserBilling | null,
  ) => {
    if (err) return <p className="text-amber-600 text-sm py-2">{err}</p>;
    if (!billing) return ldg ? null : <p className="text-zinc-400 text-sm py-2">No data for {period}.</p>;

    const d = (v: number, baseV: number) => baseBilling != null ? v - baseV : undefined;

    const mResources = [...billing.resource_by_type]
      .filter(([, v]) => v >= 10).sort(([, a], [, b]) => b - a).map(([r]) => r);
    const hasOtherR = [...billing.resource_by_type.values()].some(v => v < 10);
    const brOtherCost = [...billing.resource_by_type]
      .filter(([r]) => !mResources.includes(r)).reduce((s, [, v]) => s + v, 0);

    const mProjects = [...billing.billing_by_project]
      .filter(([, v]) => v >= 10).sort(([, a], [, b]) => b - a).map(([p]) => p);
    const hasOtherBP = [...billing.billing_by_project.values()].some(v => v < 10);
    const bpOtherCost = [...billing.billing_by_project]
      .filter(([p]) => !mProjects.includes(p)).reduce((s, [, v]) => s + v, 0);

    const rows = billingView === 'summary' ? (
      <>
        <CostRow label="Resource usage" value={billing.resource_cost} pctStr={pct(billing.resource_cost, billing.total)} delta={d(billing.resource_cost, baseBilling?.resource_cost ?? 0)} />
        <CostRow label="Service fees" value={billing.service_fee_cost} pctStr={pct(billing.service_fee_cost, billing.total)} delta={d(billing.service_fee_cost, baseBilling?.service_fee_cost ?? 0)} />
        <CostRow label="Total" value={billing.total} bold delta={d(billing.total, baseBilling?.total ?? 0)} />
      </>
    ) : billingView === 'billing_project' ? (
      <>
        {mProjects.map((p, i) => {
          const v = billing.billing_by_project.get(p) ?? 0;
          return <CostRow key={p} label={p} value={v} pctStr={pct(v, billing.total)} indent colorClass={BILLING_PROJECT_TEXT_CLASSES[i % BILLING_PROJECT_TEXT_CLASSES.length]} delta={d(v, baseBilling?.billing_by_project.get(p) ?? 0)} />;
        })}
        {hasOtherBP && bpOtherCost > 0 && (() => {
          const baseOther = baseBilling ? [...baseBilling.billing_by_project].filter(([p]) => !mProjects.includes(p)).reduce((s, [, v]) => s + v, 0) : 0;
          return <CostRow label="(Other)" value={bpOtherCost} pctStr={pct(bpOtherCost, billing.total)} indent delta={d(bpOtherCost, baseOther)} />;
        })()}
        <CostRow label="Total" value={billing.total} bold delta={d(billing.total, baseBilling?.total ?? 0)} />
        <div className="flex justify-between py-2 border-b border-zinc-100 text-xs text-zinc-500">
          <span>Active billing projects</span>
          <span className="tabular-nums font-medium text-zinc-700">{billing.billing_project_count}</span>
        </div>
        <div className="flex justify-between py-2 border-b border-zinc-100 last:border-0 text-xs text-zinc-500">
          <span title="Normalized Herfindahl-Hirschman Index: 0 = perfectly equal across projects, 1 = one project has all billing" className="cursor-help underline decoration-dotted decoration-zinc-400">Concentration index</span>
          <span className="tabular-nums font-medium text-zinc-700">{billing.billing_project_concentration.toFixed(3)}</span>
        </div>
      </>
    ) : (
      <>
        {mResources.map(r => {
          const v = billing.resource_by_type.get(r) ?? 0;
          return <CostRow key={r} label={r} value={v} pctStr={pct(v, billing.resource_cost)} indent delta={d(v, baseBilling?.resource_by_type.get(r) ?? 0)} />;
        })}
        {hasOtherR && brOtherCost > 0 && (() => {
          const baseOther = baseBilling ? [...baseBilling.resource_by_type].filter(([r]) => !mResources.includes(r)).reduce((s, [, v]) => s + v, 0) : 0;
          return <CostRow label="(Other)" value={brOtherCost} pctStr={pct(brOtherCost, billing.resource_cost)} indent delta={d(brOtherCost, baseOther)} />;
        })()}
        <CostRow label="Resource usage total" value={billing.resource_cost} bold delta={d(billing.resource_cost, baseBilling?.resource_cost ?? 0)} />
      </>
    );

    const pieData: PieSlice[] = billingView === 'summary'
      ? [
          { name: 'Resource usage', value: billing.resource_cost, fill: '#10b981' },
          { name: 'Service fees', value: billing.service_fee_cost, fill: '#6ee7b7' },
        ]
      : billingView === 'billing_project'
        ? [
            ...mProjects.map((p, i) => ({ name: p, value: billing.billing_by_project.get(p) ?? 0, fill: BILLING_PROJECT_PALETTE[i % BILLING_PROJECT_PALETTE.length] })),
            ...(hasOtherBP ? [{ name: '(Other)', value: bpOtherCost, fill: '#9ca3af' }] : []),
          ]
        : [
            ...mResources.map((r, i) => ({ name: r, value: billing.resource_by_type.get(r) ?? 0, fill: BILLING_RESOURCE_PALETTE[i % BILLING_RESOURCE_PALETTE.length] })),
            ...(hasOtherR ? [{ name: '(Other)', value: brOtherCost, fill: '#9ca3af' }] : []),
          ];

    return (
      <div className="flex items-center gap-4">
        <div className="flex-1">{rows}</div>
        <div className={`${compact ? 'w-28' : 'w-40'} flex-shrink-0`}>
          <MiniPieChart data={pieData} size={compact ? 'sm' : 'md'} format={fmt} />
        </div>
      </div>
    );
  };

  const renderMarginBody = (costs: CloudCosts | null, billing: UserBilling | null, baseCosts?: CloudCosts | null, baseBilling?: UserBilling | null) => {
    if (!costs || !billing) return <p className="text-zinc-400 text-sm py-2">No data.</p>;
    const netVal = billing.total - costs.total;
    const baseNetVal = baseCosts && baseBilling ? baseBilling.total - baseCosts.total : null;
    return (
      <>
        <CostRow label="Net (billed − cloud)" value={netVal} bold colorClass={netVal >= 0 ? 'text-emerald-600' : 'text-red-600'} delta={baseNetVal != null ? netVal - baseNetVal : undefined} />
        <div className="flex justify-between py-2 border-b border-zinc-100">
          <span className="text-zinc-700 font-semibold">Margin %</span>
          <span className={`tabular-nums font-semibold ${netVal >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>{pct(netVal, costs.total)}</span>
        </div>
      </>
    );
  };

  const renderFixedRatios = (costs: CloudCosts | null, billing: UserBilling | null) => {
    if (!costs || !billing) return <p className="text-zinc-400 text-sm py-2">No data.</p>;
    return (
      <>
        <RatioRow label="Compute (user-driven) as % of cloud" value={pct(costs.user_compute, costs.total)} />
        <RatioRow label="Resource billing as % of user-driven compute" value={pct(billing.resource_cost, costs.user_compute)} />
        <RatioRow label="Service fees as % of user billing" value={pct(billing.service_fee_cost, billing.total)} />
        <RatioRow label="Service fees as % of overhead" value={pct(billing.service_fee_cost, costs.other_compute + [...costs.non_compute_services.values()].reduce((a, b) => a + b, 0))} />
      </>
    );
  };

  const net = userBilling && cloudCosts ? userBilling.total - cloudCosts.total : null;

  const tabClass = (t: typeof tab) =>
    `px-4 py-2 text-sm font-medium border-b-2 transition-colors ${tab === t
      ? 'border-sky-500 text-sky-600'
      : 'border-transparent text-zinc-500 hover:text-zinc-700'}`;

  return (
    <div className="px-4 py-6 space-y-6">
      <h1 className="text-2xl font-light text-zinc-800">Cost Analysis</h1>

      <div className="flex border-b border-zinc-200">
        <button type="button" className={tabClass('monthly')} onClick={() => { changeTab('monthly'); }}>Monthly Breakdown</button>
        <button type="button" className={tabClass('trends')} onClick={() => { changeTab('trends'); }}>Trends</button>
      </div>

      {tab === 'monthly' && (
        <>
          {compareTimePeriod !== null ? (
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center gap-2">
                <button type="button" onClick={() => { setTimePeriod(p => shiftMonthParam(p, -1)); }} className="px-2 py-1 text-sm border border-zinc-300 rounded hover:bg-zinc-100">‹</button>
                <input
                  type="month"
                  className="border border-zinc-300 rounded px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-sky-400"
                  value={monthParamToInputValue(timePeriod)}
                  onChange={e => { if (e.target.value) setTimePeriod(inputValueToMonthParam(e.target.value)); }}
                />
                <button type="button" onClick={() => { setTimePeriod(p => shiftMonthParam(p, 1)); }} className="px-2 py-1 text-sm border border-zinc-300 rounded hover:bg-zinc-100">›</button>
                {loading && <span className="text-xs text-zinc-400 animate-pulse">Loading…</span>}
              </div>
              <div className="flex items-center gap-2">
                <button type="button" onClick={() => { setCompareTimePeriod(p => p ? shiftMonthParam(p, -1) : p); }} className="px-2 py-1 text-sm border border-zinc-300 rounded hover:bg-zinc-100">‹</button>
                <input
                  type="month"
                  className="border border-zinc-300 rounded px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-sky-400"
                  value={monthParamToInputValue(compareTimePeriod)}
                  onChange={e => { if (e.target.value) setCompareTimePeriod(inputValueToMonthParam(e.target.value)); }}
                />
                <button type="button" onClick={() => { setCompareTimePeriod(p => p ? shiftMonthParam(p, 1) : p); }} className="px-2 py-1 text-sm border border-zinc-300 rounded hover:bg-zinc-100">›</button>
                {compareLoading && <span className="text-xs text-zinc-400 animate-pulse">Loading…</span>}
                <button
                  type="button"
                  onClick={() => { setCompareTimePeriod(null); }}
                  className="ml-1 p-1 rounded text-zinc-400 hover:text-zinc-600 hover:bg-zinc-100"
                  title="Remove comparison"
                >
                  <svg aria-hidden="true" className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <label htmlFor="cost-analysis-month" className="text-sm text-zinc-500">Month</label>
              <button type="button" onClick={() => { setTimePeriod(p => shiftMonthParam(p, -1)); }} className="px-2 py-1 text-sm border border-zinc-300 rounded hover:bg-zinc-100">‹</button>
              <input
                id="cost-analysis-month"
                type="month"
                className="border border-zinc-300 rounded px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-sky-400"
                value={monthParamToInputValue(timePeriod)}
                onChange={e => { if (e.target.value) setTimePeriod(inputValueToMonthParam(e.target.value)); }}
              />
              <button type="button" onClick={() => { setTimePeriod(p => shiftMonthParam(p, 1)); }} className="px-2 py-1 text-sm border border-zinc-300 rounded hover:bg-zinc-100">›</button>
              {loading && <span className="text-xs text-zinc-400 animate-pulse">Loading…</span>}
              <button
                type="button"
                onClick={() => { setCompareTimePeriod(shiftMonthParam(timePeriod, -1)); }}
                className="ml-2 flex items-center gap-1.5 text-sm text-sky-600 hover:text-sky-700 border border-sky-200 hover:border-sky-400 rounded px-3 py-1.5 bg-sky-50 hover:bg-sky-100 transition-colors"
              >
                <svg aria-hidden="true" className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 3a1 1 0 011 1v5h5a1 1 0 110 2h-5v5a1 1 0 11-2 0v-5H4a1 1 0 110-2h5V4a1 1 0 011-1z" clipRule="evenodd" />
                </svg>
                Add month to compare
              </button>
            </div>
          )}

          <Panel title="Cloud Costs" viewSelector={
            <select
              className="text-xs border border-zinc-300 rounded px-2 py-1 bg-white text-zinc-600 focus:outline-none focus:ring-2 focus:ring-sky-400"
              value={cloudView}
              onChange={e => { setCloudView(e.target.value); }}
            >
              <option value="summary">Summary</option>
              <option value="user_compute">Compute (user-driven)</option>
              <option value="other_compute">Compute (other)</option>
              <option value="k8s">K8s</option>
              {overheadServicesMonthly.map(svc => <option key={svc} value={svc}>{svc}</option>)}
            </select>
          }>
            {compareTimePeriod !== null ? (
              <div className="grid grid-cols-2 divide-x divide-zinc-100">
                <div className="pr-4">{renderCloudBody(cloudCosts, cloudError, loading, timePeriod, true)}</div>
                <div className="pl-4">{renderCloudBody(compareCloudCosts, compareCloudError, compareLoading, compareTimePeriod, true, cloudCosts)}</div>
              </div>
            ) : renderCloudBody(cloudCosts, cloudError, loading, timePeriod, false)}
          </Panel>

          <Panel title="User Billing" viewSelector={
            <select
              className="text-xs border border-zinc-300 rounded px-2 py-1 bg-white text-zinc-600 focus:outline-none focus:ring-2 focus:ring-sky-400"
              value={billingView}
              onChange={e => { setBillingView(e.target.value as typeof billingView); }}
            >
              <option value="summary">Summary</option>
              <option value="resource_usage">Resource usage</option>
              <option value="billing_project">By billing project</option>
            </select>
          }>
            {compareTimePeriod !== null ? (
              <div className="grid grid-cols-2 divide-x divide-zinc-100">
                <div className="pr-4">{renderBillingBody(userBilling, billingError, loading, timePeriod, true)}</div>
                <div className="pl-4">{renderBillingBody(compareUserBilling, compareBillingError, compareLoading, compareTimePeriod, true, userBilling)}</div>
              </div>
            ) : renderBillingBody(userBilling, billingError, loading, timePeriod, false)}
          </Panel>

          {(userBilling ?? compareTimePeriod !== null) && (
            <Panel title="Usage">
              {compareTimePeriod !== null ? (
                <div className="grid grid-cols-2 divide-x divide-zinc-100">
                  <div className="pr-4">
                    {userBilling
                      ? <RatioRow label="Core hours" value={fmtCoreHours(userBilling.service_fee_cost * 100)} />
                      : <p className="text-zinc-400 text-sm py-2">—</p>}
                  </div>
                  <div className="pl-4">
                    {compareUserBilling ? (() => {
                      const cmpHrs = compareUserBilling.service_fee_cost * 100;
                      const deltaHrs = userBilling ? cmpHrs - userBilling.service_fee_cost * 100 : null;
                      const deltaStr = deltaHrs != null ? ` (${deltaHrs >= 0 ? '+' : ''}${fmtCoreHours(deltaHrs)})` : '';
                      return <RatioRow label="Core hours" value={`${fmtCoreHours(cmpHrs)}${deltaStr}`} />;
                    })() : <p className="text-zinc-400 text-sm py-2">—</p>}
                  </div>
                </div>
              ) : (
                userBilling && <RatioRow label="Core hours" value={fmtCoreHours(userBilling.service_fee_cost * 100)} />
              )}
            </Panel>
          )}

          {(net !== null || compareTimePeriod !== null) && (
            <>
              <Panel title="Margin Analysis">
                {compareTimePeriod !== null ? (
                  <div className="grid grid-cols-2 divide-x divide-zinc-100">
                    <div className="pr-4">{renderMarginBody(cloudCosts, userBilling)}</div>
                    <div className="pl-4">{renderMarginBody(compareCloudCosts, compareUserBilling, cloudCosts, userBilling)}</div>
                  </div>
                ) : (
                  net !== null && cloudCosts && userBilling && renderMarginBody(cloudCosts, userBilling)
                )}
              </Panel>
              <Panel title="Ratios">
                {compareTimePeriod !== null ? (
                  <>
                    <div className="grid grid-cols-2 divide-x divide-zinc-100">
                      <div className="pr-4">{renderFixedRatios(cloudCosts, userBilling)}</div>
                      <div className="pl-4">{renderFixedRatios(compareCloudCosts, compareUserBilling)}</div>
                    </div>
                    <div className="border-t border-zinc-100 pt-1 mt-1">
                      <CustomRatioPicker fieldGroups={fieldGroups} num={customRatioNum} den={customRatioDen} onNumChange={setCustomRatioNum} onDenChange={setCustomRatioDen} />
                      <div className="grid grid-cols-2 divide-x divide-zinc-100">
                        <div className="pr-4">
                          <RatioRow
                            label={`${fieldLabel(customRatioNum, fieldGroups)} as % of ${fieldLabel(customRatioDen, fieldGroups)}`}
                            value={cloudCosts && userBilling ? pct(resolveMonthly(customRatioNum, cloudCosts, userBilling), resolveMonthly(customRatioDen, cloudCosts, userBilling)) : '—'}
                          />
                        </div>
                        <div className="pl-4">
                          <RatioRow
                            label={`${fieldLabel(customRatioNum, fieldGroups)} as % of ${fieldLabel(customRatioDen, fieldGroups)}`}
                            value={compareCloudCosts && compareUserBilling ? pct(resolveMonthly(customRatioNum, compareCloudCosts, compareUserBilling), resolveMonthly(customRatioDen, compareCloudCosts, compareUserBilling)) : '—'}
                          />
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  net !== null && cloudCosts && userBilling && (
                    <>
                      {renderFixedRatios(cloudCosts, userBilling)}
                      <div className="border-t border-zinc-100 pt-1">
                        <CustomRatioPicker fieldGroups={fieldGroups} num={customRatioNum} den={customRatioDen} onNumChange={setCustomRatioNum} onDenChange={setCustomRatioDen} />
                        <RatioRow
                          label={`${fieldLabel(customRatioNum, fieldGroups)} as % of ${fieldLabel(customRatioDen, fieldGroups)}`}
                          value={pct(resolveMonthly(customRatioNum, cloudCosts, userBilling), resolveMonthly(customRatioDen, cloudCosts, userBilling))}
                        />
                      </div>
                    </>
                  )
                )}
              </Panel>
            </>
          )}
        </>
      )}

      {tab === 'trends' && (
          <div className="flex items-center gap-2">
            <label htmlFor="cost-analysis-trends-start" className="text-sm text-zinc-500">From</label>
            <input
              id="cost-analysis-trends-start"
              type="month"
              className="border border-zinc-300 rounded px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-sky-400"
              value={monthParamToInputValue(trendsStart)}
              onChange={e => { if (e.target.value) setTrendsStart(inputValueToMonthParam(e.target.value)); }}
            />
            <label htmlFor="cost-analysis-trends-end" className="text-sm text-zinc-500">to</label>
            <input
              id="cost-analysis-trends-end"
              type="month"
              className="border border-zinc-300 rounded px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-sky-400"
              value={monthParamToInputValue(trendsEnd)}
              onChange={e => { if (e.target.value) setTrendsEnd(inputValueToMonthParam(e.target.value)); }}
            />
            <button
              type="button"
              onClick={() => void fetchTrends(trendsStart, trendsEnd)}
              disabled={trendsLoading}
              className="px-4 py-1.5 text-sm font-medium rounded border border-sky-500 bg-sky-500 text-white hover:bg-sky-600 hover:border-sky-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {trendsLoading ? 'Loading…' : 'Compare'}
            </button>
          </div>
      )}

      {tab === 'trends' && (
        trendsLoading ? (
          <p className="text-zinc-400 text-sm py-4 text-center animate-pulse">Loading…</p>
        ) : trendsError ? (
          <p className="text-red-400 text-sm py-8 text-center">Failed to load trend data ({trendsError}). Please try again.</p>
        ) : trendData.length === 0 ? (
          <p className="text-zinc-400 text-sm py-8 text-center">Select a date range above and press Compare to load data.</p>
        ) : (
          <div>
            <Panel title="Cloud Costs" collapsible viewSelector={
              <div className="flex items-center gap-3">
                <ToggleSwitch checked={cloudShowPct} onChange={setCloudShowPct} />
                <select
                  className="text-xs border border-zinc-300 rounded px-2 py-1 bg-white text-zinc-600 focus:outline-none focus:ring-2 focus:ring-sky-400"
                  value={cloudView}
                  onChange={e => { setCloudView(e.target.value); }}
                >
                  <option value="summary">Summary</option>
                  <option value="user_compute">Compute (user-driven)</option>
                  <option value="other_compute">Compute (other)</option>
                  <option value="k8s">K8s</option>
                  {overheadServices.map(svc => <option key={svc} value={svc}>{svc}</option>)}
                </select>
              </div>
            }>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={cloudChartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
                  <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                  <YAxis
                    tickFormatter={cloudShowPct ? ((v: number) => `${v.toFixed(0)}%`) : makeYDollarFormatter(cloudYMax)}
                    tick={{ fontSize: 11 }}
                    width={56}
                    domain={cloudShowPct ? [0, 100] : [0, cloudYMax]}
                  />
                  <Tooltip content={(p) => <ChartTooltip {...p} stats={cloudShowPct ? cloudPctStats : cloudStats} seriesStats={new Map(Object.entries(cloudShowPct ? cloudPctSeriesStats : cloudSeriesStats))} format={cloudShowPct ? (v => `${v.toFixed(1)}%`) : fmt} stacked threshold={cloudShowPct ? undefined : 10} />} />
                  {cloudView === 'summary' ? (
                    <>
                      <Legend onClick={onSummaryCloudLegendClick} wrapperStyle={{ cursor: 'pointer' }} />
                      {statsReferenceLines(cloudShowPct ? cloudPctStats : cloudStats, 0, cloudShowPct ? 100 : cloudYMax)}
                      <Bar dataKey="user_compute" name="Compute (user-driven)" stackId="a" fill="#0ea5e9" hide={cloudCostsToggle.isHidden('user_compute')} />
                      <Bar dataKey="other_compute" name="Compute (other)" stackId="a" fill="#f59e0b" hide={cloudCostsToggle.isHidden('other_compute')} />
                      <Bar dataKey="k8s" name="K8s" stackId="a" fill="#10b981" hide={cloudCostsToggle.isHidden('k8s')} />
                      {overheadServices.map(svc => (
                        <Bar key={svc} dataKey={svc} name={svc} stackId="a" fill={overheadServiceColor(svc)} hide={isOverheadHidden(svc)} />
                      ))}
                    </>
                  ) : cloudView === 'user_compute' ? (
                    <>
                      <Legend onClick={onUserComputeLegendClick} wrapperStyle={{ cursor: 'pointer' }} />
                      {statsReferenceLines(cloudShowPct ? cloudPctStats : cloudStats, 0, cloudShowPct ? 100 : cloudYMax)}
                      {userComputeProducts.map(p => (
                        <Bar key={p} dataKey={p} name={p} stackId="a" fill={userComputeProductColor(p)} hide={isUserComputeHidden(p)} />
                      ))}
                      {userComputeHasOther && <Bar dataKey="(Other)" name="(Other)" stackId="a" fill="#9ca3af" hide={isUserComputeHidden('(Other)')} />}
                    </>
                  ) : cloudView === 'other_compute' ? (
                    <>
                      <Legend onClick={otherComputeToggle.onLegendClick} wrapperStyle={{ cursor: 'pointer' }} />
                      {statsReferenceLines(cloudShowPct ? cloudPctStats : cloudStats, 0, cloudShowPct ? 100 : cloudYMax)}
                      <Bar dataKey="batch_test" name="CI / test batches" stackId="a" fill="#f59e0b" hide={otherComputeToggle.isHidden('batch_test')} />
                      <Bar dataKey="batch_dev" name="Dev batches" stackId="a" fill="#fcd34d" hide={otherComputeToggle.isHidden('batch_dev')} />
                      <Bar dataKey="unknown" name="Unknown / unlabeled" stackId="a" fill="#fef3c7" hide={otherComputeToggle.isHidden('unknown')} />
                    </>
                  ) : cloudView === 'k8s' ? (
                    <>
                      <Legend onClick={k8sToggle.onLegendClick} wrapperStyle={{ cursor: 'pointer' }} />
                      {statsReferenceLines(cloudShowPct ? cloudPctStats : cloudStats, 0, cloudShowPct ? 100 : cloudYMax)}
                      <Bar dataKey="k8s_nodes" name="Compute nodes" stackId="a" fill="#059669" hide={k8sToggle.isHidden('k8s_nodes')} />
                      <Bar dataKey="k8s_mgmt" name="Management fee" stackId="a" fill="#6ee7b7" hide={k8sToggle.isHidden('k8s_mgmt')} />
                    </>
                  ) : overheadServices.includes(cloudView) ? (
                    <>
                      <Legend onClick={onOverheadLegendClick} wrapperStyle={{ cursor: 'pointer' }} />
                      {statsReferenceLines(cloudShowPct ? cloudPctStats : cloudStats, 0, cloudShowPct ? 100 : cloudYMax)}
                      {(overheadSkusByService.get(cloudView) ?? []).map(sku => (
                        <Bar key={sku} dataKey={sku} name={sku} stackId="a" fill={overheadServiceColor(sku)} hide={isOverheadHidden(sku)} />
                      ))}
                    </>
                  ) : null}
                </BarChart>
              </ResponsiveContainer>
              <StatsDisplay stats={cloudShowPct ? cloudPctStats : cloudStats} format={cloudShowPct ? (v => `${v.toFixed(1)}%`) : fmt} />
            </Panel>

            <div className="h-10" />
            <Panel title="Billing Charges" collapsible viewSelector={
              <div className="flex items-center gap-3">
                <ToggleSwitch checked={billingShowPct} onChange={setBillingShowPct} />
                <select
                  className="text-xs border border-zinc-300 rounded px-2 py-1 bg-white text-zinc-600 focus:outline-none focus:ring-2 focus:ring-sky-400"
                  value={billingView}
                  onChange={e => { setBillingView(e.target.value as typeof billingView); }}
                >
                  <option value="summary">Summary</option>
                  <option value="resource_usage">Resource usage</option>
                  <option value="billing_project">By billing project</option>
                </select>
              </div>
            }>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={billingChartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
                  <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                  <YAxis
                    tickFormatter={billingShowPct ? ((v: number) => `${v.toFixed(0)}%`) : makeYDollarFormatter(billingYMax)}
                    tick={{ fontSize: 11 }}
                    width={56}
                    domain={billingShowPct ? [0, 100] : [0, billingYMax]}
                  />
                  <Tooltip content={(p) => <ChartTooltip {...p} stats={billingShowPct ? billingPctStats : billingStats} seriesStats={new Map(Object.entries(billingShowPct ? billingPctSeriesStats : billingSeriesStats))} format={billingShowPct ? (v => `${v.toFixed(1)}%`) : fmt} stacked threshold={billingShowPct ? undefined : 10} />} />
                  {billingView === 'summary' ? (
                    <>
                      <Legend onClick={billingToggle.onLegendClick} wrapperStyle={{ cursor: 'pointer' }} />
                      {statsReferenceLines(billingShowPct ? billingPctStats : billingStats, 0, billingShowPct ? 100 : billingYMax)}
                      <Bar dataKey="resource_cost" name="Resource charges" stackId="a" fill="#10b981" hide={billingToggle.isHidden('resource_cost')} />
                      <Bar dataKey="service_fees" name="Service fees" stackId="a" fill="#6ee7b7" hide={billingToggle.isHidden('service_fees')} />
                    </>
                  ) : billingView === 'billing_project' ? (
                    <>
                      <Legend onClick={onBillingProjectLegendClick} wrapperStyle={{ cursor: 'pointer' }} />
                      {statsReferenceLines(billingShowPct ? billingPctStats : billingStats, 0, billingShowPct ? 100 : billingYMax)}
                      {billingProjects.map(p => (
                        <Bar key={p} dataKey={p} name={p} stackId="a" fill={billingProjectColor(p)} hide={isBillingProjectHidden(p)} />
                      ))}
                      {billingProjectsHasOther && <Bar dataKey="(Other)" name="(Other)" stackId="a" fill="#9ca3af" hide={isBillingProjectHidden('(Other)')} />}
                    </>
                  ) : (
                    <>
                      <Legend onClick={onBillingResourceLegendClick} wrapperStyle={{ cursor: 'pointer' }} />
                      {statsReferenceLines(billingShowPct ? billingPctStats : billingStats, 0, billingShowPct ? 100 : billingYMax)}
                      {billingResources.map(r => (
                        <Bar key={r} dataKey={r} name={r} stackId="a" fill={billingResourceColor(r)} hide={isBillingResourceHidden(r)} />
                      ))}
                      {billingResourcesHasOther && <Bar dataKey="(Other)" name="(Other)" stackId="a" fill="#9ca3af" hide={isBillingResourceHidden('(Other)')} />}
                    </>
                  )}
                </BarChart>
              </ResponsiveContainer>
              <StatsDisplay stats={billingShowPct ? billingPctStats : billingStats} format={billingShowPct ? (v => `${v.toFixed(1)}%`) : fmt} />
            </Panel>

            <div className="h-10" />
            <Panel title="Profit" collapsible>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={trendData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
                  <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                  <YAxis tickFormatter={makeYDollarFormatter(profitYExtent)} tick={{ fontSize: 11 }} width={56} />
                  <Tooltip content={(p) => <ChartTooltip {...p} stats={profitStats} format={fmt} />} />
                  <ReferenceLine y={0} stroke="#52525b" strokeWidth={1.5} />
                  {statsReferenceLines(profitStats, -Infinity, Infinity)}
                  <Bar dataKey="profit" name="Profit">
                    {trendData.map(d => <Cell key={d.month} fill={d.profit >= 0 ? '#10b981' : '#ef4444'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <StatsDisplay stats={profitStats} format={fmt} />
            </Panel>

            <div className="h-10" />
            <Panel title="Usage" collapsible>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={coreHoursData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
                  <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                  <YAxis tickFormatter={(v: number) => fmtCoreHours(v)} tick={{ fontSize: 11 }} width={56} domain={[0, coreHoursExtent]} />
                  <Tooltip content={(p) => <ChartTooltip {...p} stats={coreHoursStats} format={fmtCoreHours} />} />
                  {statsReferenceLines(coreHoursStats, 0, Infinity)}
                  <Bar dataKey="core_hours" name="Core hours" fill="#818cf8" />
                </BarChart>
              </ResponsiveContainer>
              <StatsDisplay stats={coreHoursStats} format={fmtCoreHours} />
            </Panel>

            <div className="h-10" />
            <Panel title="Ratios" collapsible>
              <PresetChips
                presets={RATIO_PRESETS}
                activeNum={customRatioNum}
                activeDen={customRatioDen}
                onSelect={(num, den) => { setCustomRatioNum(num); setCustomRatioDen(den); }}
              />
              <CustomRatioPicker fieldGroups={fieldGroups} num={customRatioNum} den={customRatioDen} onNumChange={setCustomRatioNum} onDenChange={setCustomRatioDen} />
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={customRatioChartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
                  <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                  <YAxis tickFormatter={(v: number) => `${v.toFixed(0)}%`} tick={{ fontSize: 11 }} width={48} domain={[0, 'auto']} />
                  <Tooltip content={(p) => <ChartTooltip {...p} stats={customRatioStats} format={v => `${v.toFixed(1)}%`} />} />
                  {statsReferenceLines(customRatioStats, -Infinity, Infinity)}
                  <Bar dataKey="value" name={`${fieldLabel(customRatioNum, fieldGroups)} as % of ${fieldLabel(customRatioDen, fieldGroups)}`} fill="#22d3ee" />
                </BarChart>
              </ResponsiveContainer>
              <StatsDisplay stats={customRatioStats} format={v => `${v.toFixed(1)}%`} />
            </Panel>

            <div className="h-10" />
            <Panel title="Scatter Plot" collapsible>
              <ScatterPresetChips
                presets={SCATTER_PRESETS}
                activeX={scatterX}
                activeY={scatterY}
                onSelect={(x, y) => { setScatterX(x); setScatterY(y); }}
              />
              <div className="flex items-center gap-2 py-2 flex-wrap">
                <span className="text-xs text-zinc-500 shrink-0">X axis:</span>
                <select
                  className="text-xs border border-zinc-300 rounded px-2 py-1 bg-white text-zinc-600 focus:outline-none focus:ring-2 focus:ring-sky-400 flex-1 min-w-0"
                  value={scatterX}
                  onChange={e => { setScatterX(e.target.value); }}
                >
                  {fieldGroups.map(g => (
                    <optgroup key={g.group} label={g.group}>
                      {g.fields.map(f => <option key={f.id} value={f.id}>{f.label}</option>)}
                    </optgroup>
                  ))}
                </select>
                <span className="text-xs text-zinc-500 shrink-0">Y axis:</span>
                <select
                  className="text-xs border border-zinc-300 rounded px-2 py-1 bg-white text-zinc-600 focus:outline-none focus:ring-2 focus:ring-sky-400 flex-1 min-w-0"
                  value={scatterY}
                  onChange={e => { setScatterY(e.target.value); }}
                >
                  {fieldGroups.map(g => (
                    <optgroup key={g.group} label={g.group}>
                      {g.fields.map(f => <option key={f.id} value={f.id}>{f.label}</option>)}
                    </optgroup>
                  ))}
                </select>
                <ToggleSwitch checked={showRegression} onChange={setShowRegression} label="regression line" />
              </div>
              {(() => {
                const fmtDollar = (v: number) => Math.abs(v) >= 1000 ? `$${(v / 1000).toFixed(1)}k` : `$${Math.round(v)}`;
                const fmtPct = (v: number) => `${v.toFixed(1)}%`;
                const fieldFmt = (id: string) => id === 'margin/margin_pct' ? fmtPct : id === 'derived/core_hours' ? fmtCoreHours : fmtDollar;
                const fmtX = fieldFmt(scatterX);
                const fmtY = fieldFmt(scatterY);
                return <>
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart margin={{ top: 16, right: 32, left: 0, bottom: 16 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis
                    type="number"
                    dataKey="x"
                    name={fieldLabel(scatterX, fieldGroups)}
                    tickFormatter={fmtX}
                    tick={{ fontSize: 11 }}
                    label={{ value: fieldLabel(scatterX, fieldGroups), position: 'insideBottom', offset: -8, fontSize: 11, fill: '#71717a' }}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    name={fieldLabel(scatterY, fieldGroups)}
                    tickFormatter={fmtY}
                    tick={{ fontSize: 11 }}
                    width={56}
                  />
                  <ZAxis range={[40, 40]} />
                  <Tooltip
                    content={({ payload }: { payload?: readonly { payload?: unknown }[] }) => {
                      if (!payload?.length) return null;
                      const d = payload[0].payload as { month: string; x: number; y: number };
                      const predicted = scatterRegression ? scatterRegression.slope * d.x + scatterRegression.intercept : null;
                      return (
                        <div className="rounded border border-zinc-200 bg-white px-3 py-2 text-xs shadow">
                          <div className="font-semibold text-zinc-700 mb-1">{d.month}</div>
                          <div className="text-zinc-500">{fieldLabel(scatterX, fieldGroups)}: <span className="text-zinc-800 font-medium">{fmtX(d.x)}</span></div>
                          <div className="text-zinc-500">{fieldLabel(scatterY, fieldGroups)}: <span className="text-zinc-800 font-medium">{fmtY(d.y)}</span></div>
                          {predicted !== null && (
                            <div className="text-zinc-400 mt-1 pt-1 border-t border-zinc-100">
                              Predicted: <span className="text-zinc-800 font-medium">{fmtY(predicted)}</span>
                              {(() => { const delta = d.y - predicted; return <span className={`ml-2 font-medium ${delta >= 0 ? 'text-emerald-600' : 'text-red-500'}`}>(Δ {delta >= 0 ? '+' : ''}{fmtY(delta)})</span>; })()}
                            </div>
                          )}
                        </div>
                      );
                    }}
                  />
                  <Scatter
                    data={scatterChartData}
                    fill="#0ea5e9"
                    shape={(props: { cx?: number; cy?: number; payload?: { month: string } }) => {
                      const { cx = 0, cy = 0, payload } = props;
                      return (
                        <g>
                          <circle cx={cx} cy={cy} r={5} fill="#0ea5e9" fillOpacity={0.85} />
                          <text x={cx + 7} y={cy + 4} fontSize={10} fill="#52525b">{payload?.month}</text>
                        </g>
                      );
                    }}
                  />
                  {showRegression && regressionLineData.length === 2 && (
                    <Scatter
                      data={regressionLineData}
                      line={{ stroke: '#f59e0b', strokeWidth: 2, strokeDasharray: '6 3' }}
                      shape={() => <g />}
                      legendType="none"
                      tooltipType="none"
                      isAnimationActive={false}
                    />
                  )}
                </ScatterChart>
              </ResponsiveContainer>
                  <RegressionStatsDisplay
                    reg={scatterRegression}
                    xLabel={fieldLabel(scatterX, fieldGroups)}
                    yLabel={fieldLabel(scatterY, fieldGroups)}
                    fmtX={fmtX}
                    fmtY={fmtY}
                  />
                </>;
              })()}
            </Panel>
          </div>
        )
      )}

      {tab === 'trends' && trendData.length > 0 && (
        <div className="mt-6 flex items-start gap-2 rounded-md border border-sky-200 bg-sky-50 px-4 py-3 text-sm text-sky-800">
          <span className="mt-0.5 shrink-0 text-sky-400">ℹ</span>
          <span>
            <strong>Chart legend:</strong> click a series name to isolate it; click again to restore all.
            Hold <kbd className="rounded border border-sky-300 bg-white px-1 py-0.5 font-mono text-xs">Shift</kbd> and click to toggle a series on or off individually.
          </span>
        </div>
      )}
    </div>
  );
}
