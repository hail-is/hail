import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts';
import { fmt } from './format';

export interface PieSlice { name: string; value: number; fill: string }

export function MiniPieChart({ data, size = 'md' }: { data: PieSlice[]; size?: 'sm' | 'md' }) {
  const height = size === 'sm' ? 110 : 156;
  const innerR = size === 'sm' ? 28 : 42;
  const outerR = size === 'sm' ? 46 : 64;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie data={data} cx="50%" cy="50%" innerRadius={innerR} outerRadius={outerR} dataKey="value" paddingAngle={2}>
          {data.map((d, i) => <Cell key={i} fill={d.fill} />)}
        </Pie>
        <Tooltip formatter={(v) => typeof v === 'number' ? fmt(v) : ''} />
      </PieChart>
    </ResponsiveContainer>
  );
}
