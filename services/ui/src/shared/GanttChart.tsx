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
  const [dims, setDims] = useState<{ width: number; height: number } | null>(null);

  // The container's height is pinned by the parent flex layout (items-stretch +
  // overflow-hidden resets min-height:auto), so reading it here cannot feed back
  // into the layout — the SVG content never changes the container's CSS height.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width > 0 && height > 0) setDims({ width, height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!containerRef.current || rows.length === 0 || !dims) return;

    const yDomain = [...new Set(rows.map((r) => r.label))];
    // Use the container's full height when it's taller than the content would be,
    // so the chart fills the panel rather than leaving empty space.
    const contentHeight = yDomain.length * ROW_HEIGHT_PX + CHART_MARGINS_PX;
    const height = Math.max(contentHeight, dims.height);
    const minStart = rows.reduce((mn, r) => (r.start < mn ? r.start : mn), rows[0].start);

    const plot = Plot.plot({
      width: dims.width,
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
  }, [rows, colorMap, dims]);

  return <div ref={containerRef} className="w-full h-full" />;
}
