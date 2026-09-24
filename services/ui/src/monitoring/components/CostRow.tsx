import { DataRow } from '../../shared/DataRow';
import { fmt, fmtDelta } from './format';

interface CostRowProps { label: string; value: number; pctStr?: string; indent?: boolean; bold?: boolean; colorClass?: string; delta?: number }

export function CostRow({ label, value, pctStr, indent = false, bold = false, colorClass, delta }: CostRowProps) {
  return (
    <DataRow
      label={label}
      value={fmt(value)}
      secondary={pctStr}
      delta={delta !== undefined ? fmtDelta(delta, value - delta) : undefined}
      indent={indent}
      bold={bold}
      colorClass={colorClass}
    />
  );
}
