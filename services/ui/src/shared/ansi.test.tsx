import { render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { ansiLineToNodes } from './ansi';

function renderLine(line: string) {
  const { nodes } = ansiLineToNodes(line, {});
  const { container } = render(<>{nodes}</>);
  return container;
}

describe('ansiLineToNodes', () => {
  it('returns plain text unchanged when there are no escape codes', () => {
    const container = renderLine('hello world');
    expect(container.textContent).toBe('hello world');
    expect(container.querySelector('span')).toBeNull();
  });

  it('colors text wrapped in an SGR color code', () => {
    const container = renderLine('\x1b[31merror\x1b[0m ok');
    expect(container.textContent).toBe('error ok');
    const span = container.querySelector('span');
    expect(span?.textContent).toBe('error');
    expect(span?.style.color).toBe('rgb(220, 38, 38)');
  });

  it('carries state across lines until reset', () => {
    const first = ansiLineToNodes('\x1b[32mstarted', {});
    expect(first.outgoingState.fg).toBe('#16a34a');
    const second = ansiLineToNodes('still green', first.outgoingState);
    const { container } = render(<>{second.nodes}</>);
    expect(container.querySelector('span')?.style.color).toBe('rgb(22, 163, 74)');
  });

  it('supports 256-color and truecolor SGR sequences', () => {
    const truecolor = renderLine('\x1b[38;2;10;20;30mrgb\x1b[0m');
    expect(truecolor.querySelector('span')?.style.color).toBe('rgb(10, 20, 30)');
  });
});
