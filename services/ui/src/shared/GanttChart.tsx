import * as Plot from '@observablehq/plot';
import { useEffect, useRef, useState } from 'react';

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
};

const ROW_HEIGHT_PX = 36;
const CHART_MARGINS_PX = 60; // top + bottom margins + legend

export function GanttChart({ rows, colorMap }: Props): JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState<number | null>(null);

  // Only track width — reading height would cause a feedback loop where the
  // SVG grows the container, which triggers a new observation, which grows the
  // SVG further. The container's height is determined by the flex layout.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const w = entry.contentRect.width;
      if (w > 0) setWidth(w);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!containerRef.current || rows.length === 0 || !width) return;

    const yDomain = [...new Set(rows.map((r) => r.label))];
    const height = yDomain.length * ROW_HEIGHT_PX + CHART_MARGINS_PX;
    const minStart = rows.reduce((mn, r) => (r.start < mn ? r.start : mn), rows[0].start);

    const plot = Plot.plot({
      width,
      height,
      marginLeft: 160,
      marginRight: 20,
      color: { domain: Object.keys(colorMap), range: Object.values(colorMap), legend: true },
      x: { type: 'time', label: 'Time' },
      y: { label: null, domain: yDomain },
      marks: [
        Plot.barX(rows, {
          x1: 'start',
          x2: 'end',
          y: 'label',
          fill: 'category',
          rx: 2,
        }),
        Plot.tip(rows, Plot.pointer({
          x1: 'start',
          x2: 'end',
          y: 'label',
          title: (d: GanttRow) => d.tooltip ?? `${d.label}: ${d.category}`,
        })),
        Plot.ruleX([minStart]),
      ],
    });

    containerRef.current.replaceChildren(plot);
    return () => plot.remove();
  }, [rows, colorMap, width]);

  return <div ref={containerRef} className="w-full h-full overflow-y-auto" />;
}
