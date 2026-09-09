export type Segment = [count: number, colorClass: string, label: string];

interface Props {
  segments: Segment[];
  total: number;
  onSegmentEnter?: (label: string) => (e: React.MouseEvent) => void;
  onSegmentLeave?: () => void;
  className?: string;
  style?: React.CSSProperties;
}

export function SegmentedBar({ segments, total, onSegmentEnter, onSegmentLeave, className = 'h-3.5 w-40', style }: Props): JSX.Element {
  return (
    <div className={`flex bg-zinc-300 rounded overflow-hidden ${className}`} style={style}>
      {total === 0 ? null : segments.map(([count, colorClass, label], i) => {
        const pct = (count / total) * 100;
        return pct > 0 ? (
          <div
            key={i}
            className={`${colorClass} h-full flex-shrink-0`}
            style={{ width: `${pct}%` }}
            onMouseEnter={onSegmentEnter?.(label)}
            onMouseLeave={onSegmentLeave}
          />
        ) : null;
      })}
    </div>
  );
}
