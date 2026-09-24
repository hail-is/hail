import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts';

export interface PieSlice { name: string; value: number; fill: string }

export function MiniPieChart({ data, size = 'md', format = String }: { data: PieSlice[]; size?: 'sm' | 'md'; format?: (_v: number) => string }) {
  const height = size === 'sm' ? 110 : 156;
  const innerR = size === 'sm' ? 28 : 42;
  const outerR = size === 'sm' ? 46 : 64;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie data={data} cx="50%" cy="50%" innerRadius={innerR} outerRadius={outerR} dataKey="value" paddingAngle={2}>
          {data.map(d => <Cell key={d.name} fill={d.fill} />)}
        </Pie>
        <Tooltip formatter={(v) => typeof v === 'number' ? format(v) : ''} />
      </PieChart>
    </ResponsiveContainer>
  );
}
