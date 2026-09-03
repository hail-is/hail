import Anser, { type AnserJsonEntry } from 'anser';
import { Fragment, type ReactNode } from 'react';

function styleFromEntry(entry: AnserJsonEntry): React.CSSProperties {
  const style: React.CSSProperties = {};
  if (entry.fg) style.color = `rgb(${entry.fg})`;
  if (entry.bg) style.backgroundColor = `rgb(${entry.bg})`;
  for (const decoration of entry.decorations) {
    if (decoration === 'bold') style.fontWeight = 'bold';
    else if (decoration === 'dim') style.opacity = 0.65;
    else if (decoration === 'italic') style.fontStyle = 'italic';
    else if (decoration === 'underline') style.textDecoration = 'underline';
    else if (decoration === 'strikethrough') style.textDecoration = 'line-through';
  }
  return style;
}

// Renders text containing ANSI SGR color/style escape codes into React nodes.
// Colors and styles carry across lines within `text` (e.g. a color set on one line
// and reset on a later one), matching normal terminal behavior.
export function ansiTextToNodes(text: string): ReactNode[] {
  if (!text.includes('\x1b[')) return [text];

  return Anser.ansiToJson(text, { use_classes: false, remove_empty: true }).map((entry, i) => {
    const style = styleFromEntry(entry);
    return Object.keys(style).length ? (
      <span style={style} key={i}>
        {entry.content}
      </span>
    ) : (
      <Fragment key={i}>{entry.content}</Fragment>
    );
  });
}
