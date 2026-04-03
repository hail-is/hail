import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

// orient='split' format from pandas DataFrame.to_dict()
export type SplitDataFrame = {
  columns: string[];
  index: number[];
  data: (number | null)[][];
};

export type ResourceUsageData = Record<string, SplitDataFrame | null>;

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
  return `${bytes} B`;
}

function toChartData(df: SplitDataFrame): Record<string, number | null>[] {
  const colIdx: Record<string, number> = {};
  df.columns.forEach((c, i) => {
    colIdx[c] = i;
  });
  return df.data.map((row) => {
    const point: Record<string, number | null> = {};
    df.columns.forEach((col, i) => {
      point[col] = row[i];
    });
    return point;
  });
}

type StageChartProps = { stage: string; df: SplitDataFrame };

function StageChart({ stage, df }: StageChartProps): JSX.Element {
  const data = toChartData(df);

  const timeCol = 'time_msecs';
  const baseTime = (data[0]?.[timeCol] as number) ?? 0;
  const chartData = data.map((d) => ({
    ...d,
    t_s: (((d[timeCol] as number) ?? 0) - baseTime) / 1000,
  }));

  return (
    <div className="mb-6">
      <h4 className="text-sm font-medium text-zinc-600 mb-2 capitalize">{stage}</h4>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div>
          <div className="text-xs text-zinc-400 mb-1">CPU</div>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="t_s" tickFormatter={(v) => `${v}s`} tick={{ fontSize: 10 }} />
              <YAxis tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v) => typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : String(v)} />
              <Line type="monotone" dataKey="cpu_usage" dot={false} stroke="#0ea5e9" name="CPU" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div>
          <div className="text-xs text-zinc-400 mb-1">Memory</div>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="t_s" tickFormatter={(v) => `${v}s`} tick={{ fontSize: 10 }} />
              <YAxis tickFormatter={formatBytes} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v) => typeof v === 'number' ? formatBytes(v) : String(v)} />
              <Line
                type="monotone"
                dataKey="memory_in_bytes"
                dot={false}
                stroke="#8b5cf6"
                name="Memory"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div>
          <div className="text-xs text-zinc-400 mb-1">Network Upload</div>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="t_s" tickFormatter={(v) => `${v}s`} tick={{ fontSize: 10 }} />
              <YAxis tickFormatter={formatBytes} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v) => typeof v === 'number' ? `${formatBytes(v)}/s` : String(v)} />
              <Line
                type="monotone"
                dataKey="network_bandwidth_upload_in_bytes_per_second"
                dot={false}
                stroke="#10b981"
                name="Upload"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div>
          <div className="text-xs text-zinc-400 mb-1">Network Download</div>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="t_s" tickFormatter={(v) => `${v}s`} tick={{ fontSize: 10 }} />
              <YAxis tickFormatter={formatBytes} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v) => typeof v === 'number' ? `${formatBytes(v)}/s` : String(v)} />
              <Line
                type="monotone"
                dataKey="network_bandwidth_download_in_bytes_per_second"
                dot={false}
                stroke="#f59e0b"
                name="Download"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div>
          <div className="text-xs text-zinc-400 mb-1">Storage</div>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="t_s" tickFormatter={(v) => `${v}s`} tick={{ fontSize: 10 }} />
              <YAxis tickFormatter={formatBytes} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v) => typeof v === 'number' ? formatBytes(v) : String(v)} />
              <Legend />
              <Line
                type="monotone"
                dataKey="non_io_storage_in_bytes"
                dot={false}
                stroke="#ef4444"
                name="Non-IO"
              />
              <Line
                type="monotone"
                dataKey="io_storage_in_bytes"
                dot={false}
                stroke="#6366f1"
                name="IO"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

type Props = { data: ResourceUsageData };

export function ResourceCharts({ data }: Props): JSX.Element {
  const stages = Object.entries(data).filter(([, df]) => df !== null) as [string, SplitDataFrame][];

  if (stages.length === 0) {
    return <div className="text-zinc-400 text-sm py-4">No resource usage data available.</div>;
  }

  return (
    <div>
      {stages.map(([stage, df]) => (
        <StageChart key={stage} stage={stage} df={df} />
      ))}
    </div>
  );
}
