import { Fragment, type ReactNode } from 'react';

// Standard 16-color ANSI palette, tuned for readability on a light background
// (the log viewer renders on bg-slate-50), so pure white/yellow are avoided.
const ANSI_COLORS: Record<number, string> = {
  0: '#3f3f46', // black -> zinc-700 (pure black is too harsh on slate-50)
  1: '#dc2626', // red
  2: '#16a34a', // green
  3: '#a16207', // yellow -> amber-700 (legible on light bg)
  4: '#2563eb', // blue
  5: '#c026d3', // magenta
  6: '#0891b2', // cyan
  7: '#71717a', // white -> zinc-500
  8: '#a1a1aa', // bright black (gray)
  9: '#ef4444', // bright red
  10: '#22c55e', // bright green
  11: '#ca8a04', // bright yellow
  12: '#3b82f6', // bright blue
  13: '#d946ef', // bright magenta
  14: '#06b6d4', // bright cyan
  15: '#27272a', // bright white -> near-black for contrast
};

// xterm 256-color palette: 16 standard + 6x6x6 color cube + 24 grayscale ramp.
function ansi256ToHex(n: number): string {
  if (n < 16) return ANSI_COLORS[n];
  if (n < 232) {
    const i = n - 16;
    const levels = [0, 95, 135, 175, 215, 255];
    const r = levels[Math.floor(i / 36) % 6];
    const g = levels[Math.floor(i / 6) % 6];
    const b = levels[i % 6];
    return `rgb(${r}, ${g}, ${b})`;
  }
  const gray = 8 + (n - 232) * 10;
  return `rgb(${gray}, ${gray}, ${gray})`;
}

export type AnsiState = {
  fg?: string;
  bg?: string;
  bold?: boolean;
  dim?: boolean;
  italic?: boolean;
  underline?: boolean;
};

function styleFromState(state: AnsiState): React.CSSProperties {
  const style: React.CSSProperties = {};
  if (state.fg) style.color = state.fg;
  if (state.bg) style.backgroundColor = state.bg;
  if (state.bold) style.fontWeight = 'bold';
  if (state.dim) style.opacity = 0.65;
  if (state.italic) style.fontStyle = 'italic';
  if (state.underline) style.textDecoration = 'underline';
  return style;
}

function applyParams(state: AnsiState, params: number[]): AnsiState {
  const next = { ...state };
  for (let i = 0; i < params.length; i++) {
    const p = params[i];
    if (p === 0) {
      return {};
    } else if (p === 1) {
      next.bold = true;
    } else if (p === 2) {
      next.dim = true;
    } else if (p === 3) {
      next.italic = true;
    } else if (p === 4) {
      next.underline = true;
    } else if (p === 22) {
      next.bold = false;
      next.dim = false;
    } else if (p === 23) {
      next.italic = false;
    } else if (p === 24) {
      next.underline = false;
    } else if (p >= 30 && p <= 37) {
      next.fg = ANSI_COLORS[p - 30];
    } else if (p === 38) {
      if (params[i + 1] === 5 && params[i + 2] !== undefined) {
        next.fg = ansi256ToHex(params[i + 2]);
        i += 2;
      } else if (params[i + 1] === 2 && params[i + 4] !== undefined) {
        next.fg = `rgb(${params[i + 2]}, ${params[i + 3]}, ${params[i + 4]})`;
        i += 4;
      }
    } else if (p === 39) {
      next.fg = undefined;
    } else if (p >= 40 && p <= 47) {
      next.bg = ANSI_COLORS[p - 40];
    } else if (p === 48) {
      if (params[i + 1] === 5 && params[i + 2] !== undefined) {
        next.bg = ansi256ToHex(params[i + 2]);
        i += 2;
      } else if (params[i + 1] === 2 && params[i + 4] !== undefined) {
        next.bg = `rgb(${params[i + 2]}, ${params[i + 3]}, ${params[i + 4]})`;
        i += 4;
      }
    } else if (p === 49) {
      next.bg = undefined;
    } else if (p >= 90 && p <= 97) {
      next.fg = ANSI_COLORS[8 + (p - 90)];
    } else if (p >= 100 && p <= 107) {
      next.bg = ANSI_COLORS[8 + (p - 100)];
    }
  }
  return next;
}

// eslint-disable-next-line no-control-regex
const SGR_RE = /\x1b\[([0-9;]*)m/g;

// Parses one line of text containing ANSI SGR color/style escape codes into React nodes,
// carrying style state in from the previous line and returning the state to carry into the next.
export function ansiLineToNodes(line: string, incomingState: AnsiState): { nodes: ReactNode[]; outgoingState: AnsiState } {
  if (!line.includes('\x1b[')) {
    const style = styleFromState(incomingState);
    const nodes = Object.keys(style).length ? [<span style={style} key="0">{line}</span>] : [line];
    return { nodes, outgoingState: incomingState };
  }

  const nodes: ReactNode[] = [];
  let state = incomingState;
  let lastIndex = 0;
  let key = 0;
  SGR_RE.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = SGR_RE.exec(line)) !== null) {
    const chunk = line.slice(lastIndex, match.index);
    if (chunk) {
      const style = styleFromState(state);
      nodes.push(
        Object.keys(style).length ? <span style={style} key={key++}>{chunk}</span> : <Fragment key={key++}>{chunk}</Fragment>
      );
    }
    const params = match[1].length ? match[1].split(';').map((s) => (s === '' ? 0 : parseInt(s, 10))) : [0];
    state = applyParams(state, params);
    lastIndex = SGR_RE.lastIndex;
  }
  const rest = line.slice(lastIndex);
  if (rest) {
    const style = styleFromState(state);
    nodes.push(Object.keys(style).length ? <span style={style} key={key++}>{rest}</span> : <Fragment key={key++}>{rest}</Fragment>);
  }
  return { nodes, outgoingState: state };
}
