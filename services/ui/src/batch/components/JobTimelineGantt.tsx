import { useState } from 'react';
import { GanttChart, GanttRow } from '../../shared/GanttChart';
import { Job, Attempt, TERMINAL_STATES } from './jobModels';

// px.colors.qualitative.Prism from Plotly — same palette used in the classic job page
const PLOTLY_PRISM = [
  '#5F4690', '#1D6996', '#38A6A5', '#0F8554', '#73AF48',
  '#EDAD08', '#E17C05', '#CC503E', '#94346E', '#6F4070', '#994E95', '#666666',
];

// Fixed color assignments matching the classic Plotly chart (task_names order)
const STEP_COLOR_MAP: Record<string, string> = {
  'pulling':                   PLOTLY_PRISM[0],
  'setting up overlay':        PLOTLY_PRISM[1],
  'setting up network':        PLOTLY_PRISM[2],
  'running':                   PLOTLY_PRISM[3],
  'uploading_log':             PLOTLY_PRISM[4],
  'uploading_resource_usage':  PLOTLY_PRISM[5],
  'prior attempt':             '#9ca3af',
  'creating':                  '#e2e8f0',
};

interface GanttData {
  rows: GanttRow[];
  ruleXs: { x: Date; label: string }[];
}

function buildGanttRows(job: Job, attempts: Attempt[], showPriorAttempts: boolean): GanttData {
  const rows: GanttRow[] = [];
  const ruleXs: { x: Date; label: string }[] = [];
  const nowMs = Date.now();

  // If the job is still running but the last attempt already has a reason
  // (failed/preempted), treat it as a prior attempt rather than the live
  // "latest" — consistent with how the tab icons are determined.
  const lastAttempt = attempts[attempts.length - 1];
  const lastIsPrior = !TERMINAL_STATES.has(job.state) && lastAttempt?.reason != null;
  const priorAttempts = lastIsPrior ? attempts : attempts.slice(0, -1);

  // Prior attempts: one coarse bar per attempt row (all except the last)
  if (showPriorAttempts) {
    for (const a of priorAttempts) {
      const startMs = a.start_time_ms;
      if (startMs == null) continue;
      // A prior attempt with a reason is complete — don't stretch its bar to nowMs
      const isComplete = a.reason != null || a.end_time_ms != null;
      const endMs = a.end_time_ms ?? (isComplete ? startMs : nowMs);
      const durationMs = endMs - startMs;
      rows.push({
        label: `attempt ${a.attempt_id.slice(0, 8)}`,
        start: new Date(startMs),
        end: new Date(endMs),
        category: 'prior attempt',
        tooltip: [
          `Row: attempt ${a.attempt_id.slice(0, 8)}`,
          `Task: prior attempt`,
          `Start: ${new Date(startMs).toLocaleString()}`,
          `Finish: ${new Date(endMs).toLocaleString()}`,
          `Duration: ${(durationMs / 1000).toFixed(1)}s`,
          a.reason ? `Detail: ${a.reason}` : '',
        ].filter(Boolean).join('\n'),
      });
    }
  }

  // Latest attempt: all containers combined onto a single row
  const latest = lastIsPrior ? null : lastAttempt;
  if (latest?.start_time_ms != null) {
    const latestLabel = `attempt ${latest.attempt_id.slice(0, 8)}`;
    const isLatestComplete = latest.reason != null || latest.end_time_ms != null;
    const latestFallbackEndMs = isLatestComplete ? (latest.end_time_ms ?? latest.start_time_ms) : nowMs;
    const statuses = job.status?.container_statuses ?? {};

    // Find the earliest task start across all containers to bound the creating block
    let firstTaskStartMs: number | null = null;
    for (const container of ['input', 'main', 'output'] as const) {
      const cs = statuses[container];
      if (!cs) continue;
      for (const [, timingData] of Object.entries(cs.timing)) {
        if (timingData?.start_time == null) continue;
        if (firstTaskStartMs === null || timingData.start_time < firstTaskStartMs) {
          firstTaskStartMs = timingData.start_time;
        }
      }
    }

    // Creating placeholder: time from attempt start to first task start.
    // Only shown when the creating phase is at least 1 second; sub-second
    // durations are noise and would render as "0.0s" in the tooltip.
    const schedEndMs = firstTaskStartMs ?? latestFallbackEndMs;
    const creatingDurationMs = schedEndMs - latest.start_time_ms;
    if (creatingDurationMs >= 1000) {
      rows.push({
        label: latestLabel,
        start: new Date(latest.start_time_ms),
        end: new Date(schedEndMs),
        category: 'creating',
        tooltip: [
          `Row: ${latestLabel}`,
          `Task: creating`,
          `Start: ${new Date(latest.start_time_ms).toLocaleString()}`,
          `Finish: ${new Date(schedEndMs).toLocaleString()}`,
          `Duration: ${(creatingDurationMs / 1000).toFixed(1)}s`,
        ].join('\n'),
      });
      ruleXs.push({ x: new Date(latest.start_time_ms), label: 'creating' });
    }

    // Task bars — all containers merged onto the same row
    for (const container of ['input', 'main', 'output'] as const) {
      const cs = statuses[container];
      if (!cs) continue;
      let containerStartMs: number | null = null;
      for (const [stepName, timingData] of Object.entries(cs.timing)) {
        if (!timingData || timingData.start_time == null) continue;
        const startMs = timingData.start_time;
        const endMs = timingData.finish_time ?? latestFallbackEndMs;
        const durationMs = timingData.duration ?? (endMs - startMs);
        rows.push({
          label: latestLabel,
          start: new Date(startMs),
          end: new Date(endMs),
          category: stepName,
          tooltip: [
            `Row: ${latestLabel}`,
            `Task: ${stepName} (${container})`,
            `Start: ${new Date(startMs).toLocaleString()}`,
            `Finish: ${new Date(endMs).toLocaleString()}`,
            `Duration: ${(durationMs / 1000).toFixed(1)}s`,
          ].join('\n'),
        });
        if (containerStartMs === null || startMs < containerStartMs) {
          containerStartMs = startMs;
        }
      }
      // One rule per container division, at its earliest task start
      if (containerStartMs !== null) {
        ruleXs.push({ x: new Date(containerStartMs), label: container });
      }
    }
  }

  return { rows, ruleXs };
}

function buildColorMap(rows: GanttRow[]): Record<string, string> {
  const map: Record<string, string> = {};
  for (const cat of new Set(rows.map((r) => r.category))) {
    // eslint-disable-next-line security/detect-object-injection
    map[cat] = STEP_COLOR_MAP[cat] ?? '#9ca3af';
  }
  return map;
}

type Props = {
  job: Job;
  attempts: Attempt[];
  isTerminal: boolean;
};

export function JobTimelineGantt({ job, attempts, isTerminal }: Props): JSX.Element | null {
  const [showPriorAttempts, setShowPriorAttempts] = useState(true);
  const hasPriorAttempts = attempts.length > 1;
  const { rows, ruleXs } = buildGanttRows(job, attempts, showPriorAttempts);
  const colorMap = buildColorMap(rows);

  if (rows.length === 0) return null;

  return (
    <div className="w-full lg:flex-1 bg-slate-100 border rounded overflow-hidden max-h-[32rem] overflow-y-auto">
      {hasPriorAttempts && (
        <div className="flex items-center gap-2.5 px-3 pt-2 pb-1">
          <button
            type="button"
            role="switch"
            aria-checked={showPriorAttempts}
            onClick={() => setShowPriorAttempts((v) => !v)}
            className={`relative h-5 w-9 flex-shrink-0 rounded-full p-0 transition-colors duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 ${showPriorAttempts ? 'bg-sky-600' : 'bg-slate-300'}`}
          >
            <span
              className="absolute h-4 w-4 rounded-full bg-white shadow-sm"
              style={{ top: '2px', left: showPriorAttempts ? '18px' : '2px', transition: 'left 200ms ease-in-out' }}
            />
          </button>
          <button
            type="button"
            className="text-sm text-slate-600 bg-transparent border-none p-0 cursor-pointer select-none focus:outline-none focus-visible:underline"
            onClick={() => setShowPriorAttempts((v) => !v)}
          >
            Show prior attempts
          </button>
        </div>
      )}
      <GanttChart rows={rows} colorMap={colorMap} ruleXs={ruleXs} extendToNow={!isTerminal} />
    </div>
  );
}
