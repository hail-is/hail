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

const CHART_MARGINS_PX = 75; // top + bottom margins (larger to fit 2-line first tick) + legend
const MAX_BAR_HEIGHT = 200;
const MIN_BAR_HEIGHT = 20;

function formatXTick(d: Date, i: number): string {
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  const time = `${hh}:${mm}:${ss}`;
  if (i === 0) {
    const dd = String(d.getDate()).padStart(2, '0');
    const mo = String(d.getMonth() + 1).padStart(2, '0');
    const yyyy = d.getFullYear();
    return `${dd}/${mo}/${yyyy}\n${time}`;
  }
  return time;
}

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
    const nRows = yDomain.length;
    const rowHeightPx = Math.max(Math.floor(MAX_BAR_HEIGHT / nRows), MIN_BAR_HEIGHT);
    const height = nRows * rowHeightPx + CHART_MARGINS_PX;
    const plot = Plot.plot({
      width,
      height,
      marginLeft: 160,
      marginRight: 20,
      marginBottom: 50, // extra space for the two-line first tick label
      color: { domain: Object.keys(colorMap), range: Object.values(colorMap), legend: true },
      x: { type: 'time', axis: null }, // disable auto axis; we add it explicitly below
      y: { label: null, domain: yDomain },
      marks: [
        Plot.barX(rows, {
          x1: 'start',
          x2: 'end',
          y: 'label',
          fill: 'category',
          rx: 2,
        }),
        // Explicit bottom axis so it is always pinned to the bottom of the element
        Plot.axisX({
          anchor: 'bottom',
          label: 'Time',
          tickFormat: formatXTick,
        }),
        Plot.tip(rows, Plot.pointer({
          x1: 'start',
          x2: 'end',
          y: 'label',
          title: (d: GanttRow) => d.tooltip ?? `${d.label}: ${d.category}`,
        })),
      ],
    });

    containerRef.current.replaceChildren(plot);
    return () => plot.remove();
  }, [rows, colorMap, width]);

  return <div ref={containerRef} className="w-full" />;
}
