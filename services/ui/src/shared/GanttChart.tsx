import * as Plot from '@observablehq/plot';
import { useEffect, useRef, useState } from 'react';

export interface GanttRow {
  label: string;
  start: Date;
  end: Date;
  category: string;
  tooltip?: string;
}

interface RuleMark {
  x: Date;
  label: string;
}

interface Props {
  rows: GanttRow[];
  colorMap: Map<string, string>;
  ruleXs?: RuleMark[];
  extendToNow?: boolean;
}

interface TooltipState { text: string; x: number; y: number }

const BOTTOM_MARGINS_PX = 65; // bottom axis + legend
const RULE_LABEL_MARGIN_PX = 50; // extra top margin when rule labels are shown
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

export function GanttChart({ rows, colorMap, ruleXs, extendToNow }: Props): JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState<number | null>(null);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

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
    return () => { ro.disconnect(); };
  }, []);

  useEffect(() => {
    if (!containerRef.current || rows.length === 0 || !width) return;

    const rules = ruleXs ?? [];
    const hasRules = rules.length > 0;
    const marginTop = hasRules ? RULE_LABEL_MARGIN_PX : 10;

    const now = new Date();
    const xDomain: [Date, Date] | undefined = extendToNow
      ? [new Date(Math.min(...rows.map((r) => r.start.getTime()))), now]
      : undefined;

    const yDomain = [...new Set(rows.map((r) => r.label))];
    const nRows = yDomain.length;
    const rowHeightPx = Math.max(Math.floor(MAX_BAR_HEIGHT / nRows), MIN_BAR_HEIGHT);
    const height = nRows * rowHeightPx + marginTop + BOTTOM_MARGINS_PX;
    const plot = Plot.plot({
      width,
      height,
      marginTop,
      marginLeft: 160,
      marginRight: 20,
      marginBottom: 50, // extra space for the two-line first tick label
      color: { domain: [...colorMap.keys()], range: [...colorMap.values()], legend: true },
      x: { type: 'time', axis: null, domain: xDomain }, // disable auto axis; we add it explicitly below
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
        ...(hasRules ? [
          Plot.ruleX(rules, {
            x: 'x',
            stroke: '#94a3b8',
            strokeWidth: 1,
            strokeDasharray: '4,3',
          }),
          Plot.text(rules, {
            x: 'x',
            text: 'label',
            frameAnchor: 'top',
            textAnchor: 'end',
            rotate: -45,
            fontSize: 9,
            fill: '#475569',
            dy: -4,
          }),
        ] : []),
      ],
    });

    // Attach hover listeners directly to each bar rect so the full bar area
    // triggers the tooltip, rather than relying on Plot.pointer's centroid snap.
    const barGroup = plot.querySelector('g[aria-label="bar"]');
    if (barGroup) {
      barGroup.querySelectorAll('rect').forEach((rect, i) => {
        const row = rows.at(i);
        if (!row) return;
        const text = row.tooltip ?? `${row.label}: ${row.category}`;
        rect.addEventListener('mouseenter', (e) => {
          setTooltip({ text, x: (e as MouseEvent).clientX, y: (e as MouseEvent).clientY });
        });
        rect.addEventListener('mousemove', (e) => {
          setTooltip({ text, x: (e as MouseEvent).clientX, y: (e as MouseEvent).clientY });
        });
        rect.addEventListener('mouseleave', () => setTooltip(null));
      });
    }

    containerRef.current.replaceChildren(plot);
    return () => {
      plot.remove();
      setTooltip(null);
    };
  }, [rows, colorMap, ruleXs, extendToNow, width]);

  return (
    <>
      <div ref={containerRef} className="w-full" />
      {tooltip && (
        <div
          className="fixed z-50 max-w-xs rounded bg-gray-800 px-2 py-1 text-xs text-white shadow-lg whitespace-pre pointer-events-none"
          style={{ left: tooltip.x + 14, top: tooltip.y - 8 }}
        >
          {tooltip.text}
        </div>
      )}
    </>
  );
}
