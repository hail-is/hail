import prettyMilliseconds from 'pretty-ms';
import { ToggleSwitch } from './ToggleSwitch';
import { formatClockTick, formatTimestamp } from './timeUtils';

// How a chart's time axis positions and labels values
export interface TimeAxisMode {
  axisTitle: string;
  scale: 'time' | 'linear';
  // Epoch-ms timestamp -> axis value
  toX: (ms: number, originMs: number) => number;
  // May contain '\n' for a multi-line label
  formatTick: (x: number, index: number) => string;
  formatTooltip: (x: number) => string;
}

export const CLOCK_TIME: TimeAxisMode = {
  axisTitle: 'Time',
  scale: 'time',
  toX: (ms) => ms,
  formatTick: formatClockTick,
  formatTooltip: formatTimestamp,
};

// Unlike formatDuration, keeps fractional seconds so nearby points on a short axis get distinct labels
function formatElapsedSeconds(s: number): string {
  // pretty-ms renders zero as "0ms"; a bare origin reads right whatever units the other ticks use
  return s === 0 ? '0' : prettyMilliseconds(s * 1000, { unitCount: 2, subSecondsAsDecimals: true });
}

export const RELATIVE_TIME: TimeAxisMode = {
  axisTitle: 'Elapsed',
  scale: 'linear',
  toX: (ms, originMs) => (ms - originMs) / 1000,
  formatTick: formatElapsedSeconds,
  formatTooltip: (s) => `+${formatElapsedSeconds(s)}`,
};

export function TimeAxisToggle({ mode, onChange }: { mode: TimeAxisMode; onChange: (_m: TimeAxisMode) => void }): JSX.Element {
  return (
    <ToggleSwitch
      checked={mode === RELATIVE_TIME}
      onChange={(relative) => { onChange(relative ? RELATIVE_TIME : CLOCK_TIME); }}
      label="Relative time"
    />
  );
}
