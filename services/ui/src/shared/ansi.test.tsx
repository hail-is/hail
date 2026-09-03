import { render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { ansiTextToNodes } from './ansi';

function renderText(text: string) {
  const { container } = render(<>{ansiTextToNodes(text)}</>);
  return container;
}

describe('ansiTextToNodes', () => {
  it('returns plain text unchanged when there are no escape codes', () => {
    const container = renderText('hello world');
    expect(container.textContent).toBe('hello world');
    expect(container.querySelector('span')).toBeNull();
  });

  it('colors text wrapped in an SGR color code', () => {
    const container = renderText('\x1b[31merror\x1b[0m ok');
    expect(container.textContent).toBe('error ok');
    const span = container.querySelector('span');
    expect(span?.textContent).toBe('error');
    expect(span?.style.color).toBe('rgb(187, 0, 0)');
  });

  it('carries state across lines until reset', () => {
    const container = renderText('\x1b[32mstarted\nstill green\x1b[0m\nplain');
    const span = container.querySelector('span');
    expect(span?.textContent).toBe('started\nstill green');
    expect(span?.style.color).toBe('rgb(0, 187, 0)');
    expect(container.textContent).toBe('started\nstill green\nplain');
  });

  it('supports 256-color and truecolor SGR sequences', () => {
    const truecolor = renderText('\x1b[38;2;10;20;30mrgb\x1b[0m');
    expect(truecolor.querySelector('span')?.style.color).toBe('rgb(10, 20, 30)');

    const palette = renderText('\x1b[38;5;208morange\x1b[0m');
    expect(palette.querySelector('span')?.style.color).toBe('rgb(255, 135, 0)');
  });
});
