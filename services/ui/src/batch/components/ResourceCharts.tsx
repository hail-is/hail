import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  XAxisTickContentProps,
} from 'recharts';
import { useState } from 'react';
import { ToggleSwitch } from '../../shared/ToggleSwitch';
import { formatClockTick, formatElapsed, formatTime } from '../../shared/timeUtils';

// orient='split' format from pandas DataFrame.to_dict()
export interface SplitDataFrame {
  columns: string[];
  index: number[];
  data: (number | null)[][];
}

export type ResourceUsageData = Record<string, SplitDataFrame | null>;

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
  return `${bytes} B`;
}

// Match the colors used in the legacy Plotly charts in front_end.py:
//   colors = {'input': 'red', 'main': 'green', 'output': 'blue'}
const CONTAINER_COLORS = new Map<string, string>([
  ['input', '#ef4444'],
  ['main', '#22c55e'],
  ['output', '#3b82f6'],
]);
const FALLBACK_COLORS = ['#a855f7', '#f97316', '#06b6d4'];

function containerColor(name: string, index: number): string {
  return CONTAINER_COLORS.get(name) ?? FALLBACK_COLORS.at(index % FALLBACK_COLORS.length)!;
}

interface ContainerSeries {
  name: string;
  color: string;
  // t is epoch ms, or seconds since the earliest sample when relativeTime is set
  data: { t: number; value: number | null }[];
}

// Recharts doesn't break tick labels on newlines, so render each line as its own tspan
function ClockTick({ x, y, payload, index }: XAxisTickContentProps): JSX.Element {
  const lines = formatClockTick(payload.value as number | Date, index).split('\n');
  return (
    <text x={x} y={y} textAnchor="middle" fontSize={10} fill="#666">
      {lines.map((line, i) => (
        <tspan key={i} x={x} dy={i === 0 ? 10 : 12}>{line}</tspan>
      ))}
    </text>
  );
}

interface MetricChartProps {
  title: string;
  containers: ContainerSeries[];
  valueFormatter: (_v: number) => string;
  tickFormatter: (_v: number) => string;
  xDomain: [number, number];
  relativeTime: boolean;
}

function MetricChart({ title, containers, valueFormatter, tickFormatter, xDomain, relativeTime }: MetricChartProps): JSX.Element | null {
  if (containers.every((c) => c.data.length === 0)) return null;

  return (
    <div>
      <div className="text-xs text-zinc-400 mb-1">{title}</div>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="t"
            type="number"
            scale={relativeTime ? 'linear' : 'time'}
            domain={xDomain}
            interval="preserveStart"
            // Fixed so the plot area doesn't shift when toggling; fits the two-line first clock tick
            height={42}
            {...(relativeTime
              ? { tickFormatter: (v: number) => formatElapsed(v), tick: { fontSize: 10 } }
              : { tick: ClockTick })}
          />
          <YAxis tickFormatter={tickFormatter} tick={{ fontSize: 10 }} />
          <Tooltip
            formatter={(v) => (typeof v === 'number' ? valueFormatter(v) : String(v))}
            labelFormatter={(t) => (relativeTime ? `+${formatElapsed(Number(t))}` : formatTime(Number(t)))}
          />
          <Legend />
          {containers.map(({ name, color, data }) => (
            <Line key={name} data={data} type="monotone" dataKey="value" dot={false} stroke={color} name={name} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

interface Props { data: ResourceUsageData }

export function ResourceCharts({ data }: Props): JSX.Element {
  const [relativeTime, setRelativeTime] = useState(false);
  const containers = Object.entries(data).filter(([, df]) => df !== null) as [string, SplitDataFrame][];

  if (containers.length === 0) {
    return <div className="text-zinc-400 text-sm py-4">No resource usage data available.</div>;
  }

  // Compute a global time base so all containers share the same x-axis origin.
  let globalBaseTime = Infinity;
  for (const [, df] of containers) {
    const timeIdx = df.columns.indexOf('time_msecs');
    if (timeIdx >= 0 && df.data.length > 0) {
      const t = df.data[0].at(timeIdx)!;
      if (t < globalBaseTime) globalBaseTime = t;
    }
  }
  if (!isFinite(globalBaseTime)) globalBaseTime = 0;

  let globalMaxTime = -Infinity;
  for (const [, df] of containers) {
    const timeIdx = df.columns.indexOf('time_msecs');
    if (timeIdx >= 0 && df.data.length > 0) {
      const t = df.data[df.data.length - 1].at(timeIdx)!;
      if (t > globalMaxTime) globalMaxTime = t;
    }
  }
  if (!isFinite(globalMaxTime)) globalMaxTime = globalBaseTime;

  const toX = (ms: number): number => (relativeTime ? (ms - globalBaseTime) / 1000 : ms);
  const xDomain: [number, number] = [toX(globalBaseTime), toX(globalMaxTime)];

  function buildSeries(df: SplitDataFrame, colName: string): Array<{ t: number; value: number | null }> {
    const timeIdx = df.columns.indexOf('time_msecs');
    const valIdx = df.columns.indexOf(colName);
    if (valIdx < 0) return [];
    return df.data.map((row) => ({
      t: toX(row.at(timeIdx) ?? globalBaseTime),
      value: row.at(valIdx) ?? null,
    }));
  }

  function makeContainers(colName: string): ContainerSeries[] {
    return containers.map(([name, df], i) => ({
      name,
      color: containerColor(name, i),
      data: buildSeries(df, colName),
    }));
  }

  const hasIoStorage = containers.some(([, df]) => df.columns.includes('io_storage_in_bytes'));

  return (
    <div>
      <div className="flex mb-2">
        <ToggleSwitch checked={relativeTime} onChange={setRelativeTime} label="Relative time" />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MetricChart
          title="CPU"
          containers={makeContainers('cpu_usage')}
          valueFormatter={(v) => `${(v * 100).toFixed(1)}%`}
          tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          xDomain={xDomain}
          relativeTime={relativeTime}
        />
        <MetricChart
          title="Memory"
          containers={makeContainers('memory_in_bytes')}
          valueFormatter={formatBytes}
          tickFormatter={formatBytes}
          xDomain={xDomain}
          relativeTime={relativeTime}
        />
        <MetricChart
          title="Network Download"
          containers={makeContainers('network_bandwidth_download_in_bytes_per_second')}
          valueFormatter={(v) => `${formatBytes(v)}/s`}
          tickFormatter={formatBytes}
          xDomain={xDomain}
          relativeTime={relativeTime}
        />
        <MetricChart
          title="Network Upload"
          containers={makeContainers('network_bandwidth_upload_in_bytes_per_second')}
          valueFormatter={(v) => `${formatBytes(v)}/s`}
          tickFormatter={formatBytes}
          xDomain={xDomain}
          relativeTime={relativeTime}
        />
        <MetricChart
          title="Storage (Container Overlay)"
          containers={makeContainers('non_io_storage_in_bytes')}
          valueFormatter={formatBytes}
          tickFormatter={formatBytes}
          xDomain={xDomain}
          relativeTime={relativeTime}
        />
        {hasIoStorage && (
          <MetricChart
            title="Storage (Mounted Drive at /io)"
            containers={makeContainers('io_storage_in_bytes')}
            valueFormatter={formatBytes}
            tickFormatter={formatBytes}
            xDomain={xDomain}
            relativeTime={relativeTime}
          />
        )}
      </div>
    </div>
  );
}
