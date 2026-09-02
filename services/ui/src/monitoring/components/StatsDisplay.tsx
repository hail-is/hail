import { ReferenceLine } from 'recharts';
import { StatLabel } from '../../shared/StatLabel';
import { RegressionResult } from './statsUtils';

export function StatsDisplay({ stats, format }: { stats: { mean: number; std: number } | null; format: (v: number) => string }) {
  if (!stats) return null;
  const { mean, std } = stats;
  const cv = mean !== 0 ? (std / mean) * 100 : null;
  return (
    <div className="flex gap-6 mt-2 pt-2 border-t border-zinc-100 text-xs tabular-nums text-zinc-500">
      <span>
        <StatLabel label="mean (μ)" tooltip="Average value across all months shown." />
        {' '}<span className="text-zinc-700 font-medium">{format(mean)}</span>
      </span>
      <span>
        <StatLabel label="std dev (σ)" tooltip="Standard deviation — how much individual months typically deviate from the mean. A higher value means more month-to-month variability." />
        {' '}<span className="text-zinc-700 font-medium">{format(std)}</span>
      </span>
      {cv !== null && (
        <span>
          <StatLabel label="CV" tooltip="Coefficient of Variation (σ ÷ μ × 100) — relative variability as a percentage of the mean. Useful for comparing volatility across metrics of different scales. Under ~15% is generally stable; over ~30% suggests high variability." />
          {' '}<span className="text-zinc-700 font-medium">{cv.toFixed(1)}%</span>
        </span>
      )}
    </div>
  );
}

export function RegressionStatsDisplay({ reg, xLabel, yLabel, fmtX, fmtY }: {
  reg: RegressionResult | null;
  xLabel: string; yLabel: string;
  fmtX: (v: number) => string; fmtY: (v: number) => string;
}) {
  if (!reg) return null;
  const slopeTooltip = `For every 1-unit increase in ${xLabel}, ${yLabel} changes by this amount on average.`;
  const yInterceptTooltip = `Predicted value of ${yLabel} when ${xLabel} is zero.`;
  const xIntercept = reg.slope !== 0 ? -reg.intercept / reg.slope : null;
  const xInterceptTooltip = `Value of ${xLabel} at which the regression line predicts ${yLabel} reaches zero.`;
  const r2Tooltip = 'R² (coefficient of determination) — how well the regression line fits the data. 1.0 = perfect fit; 0 = no linear relationship.';
  return (
    <div className="flex gap-6 mt-2 pt-2 border-t border-zinc-100 text-xs tabular-nums text-zinc-500">
      <span>
        <StatLabel label="slope" tooltip={slopeTooltip} />
        {' '}<span className="text-zinc-700 font-medium">{fmtY(reg.slope)}/{fmtX(1).replace('$', '')}</span>
      </span>
      <span>
        <StatLabel label="y-intercept" tooltip={yInterceptTooltip} />
        {' '}<span className="text-zinc-700 font-medium">{fmtY(reg.intercept)}</span>
      </span>
      {xIntercept !== null && (
        <span>
          <StatLabel label="x-intercept" tooltip={xInterceptTooltip} />
          {' '}<span className="text-zinc-700 font-medium">{fmtX(xIntercept)}</span>
        </span>
      )}
      <span>
        <StatLabel label="R²" tooltip={r2Tooltip} />
        {' '}<span className="text-zinc-700 font-medium">{reg.r2.toFixed(3)}</span>
      </span>
    </div>
  );
}

export function statsReferenceLines(stats: { mean: number; std: number } | null, yMin: number, yMax: number) {
  if (!stats) return null;
  const { mean, std } = stats;
  return [
    { y: mean - 2 * std, label: 'μ−2σ', solid: false, alpha: 0.35 },
    { y: mean - std,     label: 'μ−σ',  solid: false, alpha: 0.55 },
    { y: mean,           label: 'μ',    solid: true,  alpha: 0.8  },
    { y: mean + std,     label: 'μ+σ',  solid: false, alpha: 0.55 },
    { y: mean + 2 * std, label: 'μ+2σ', solid: false, alpha: 0.35 },
  ]
    .filter(e => e.y >= yMin && e.y <= yMax)
    .map(e => (
      <ReferenceLine
        key={e.label}
        y={e.y}
        stroke="#818cf8"
        strokeOpacity={e.alpha}
        strokeWidth={e.solid ? 1.5 : 1}
        strokeDasharray={e.solid ? undefined : '4 3'}
        label={{ value: e.label, position: 'insideTopRight', fontSize: 9, fill: '#818cf8', fillOpacity: e.alpha }}
      />
    ));
}
