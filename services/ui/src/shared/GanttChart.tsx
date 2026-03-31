import * as Plot from '@observablehq/plot';
import { useEffect, useRef } from 'react';

export type GanttRow = {
  label: string;
  start: Date;
  end: Date;
  category: string;
  tooltip?: string;
};

type Props = {
  rows: GanttRow[];
  colorMap: Record<string, string>;
  width: number;
};

export function GanttChart({ rows, colorMap, width }: Props): JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || rows.length === 0) return;

    const plot = Plot.plot({
      width,
      marginLeft: 160,
      marginRight: 20,
      color: { domain: Object.keys(colorMap), range: Object.values(colorMap), legend: true },
      x: { type: 'time', label: 'Time' },
      y: { label: null },
      marks: [
        Plot.barX(rows, {
          x1: 'start',
          x2: 'end',
          y: 'label',
          fill: 'category',
          title: (d: GanttRow) => d.tooltip ?? `${d.label}: ${d.category}`,
          rx: 2,
        }),
        Plot.ruleX([rows.reduce((mn, r) => (r.start < mn ? r.start : mn), rows[0].start)]),
      ],
    });

    containerRef.current.replaceChildren(plot);
    return () => plot.remove();
  }, [rows, colorMap, width]);

  return <div ref={containerRef} className="w-full overflow-x-auto" />;
}
