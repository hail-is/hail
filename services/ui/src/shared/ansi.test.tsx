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

  it('maps an SGR color code to a CSS color style', () => {
    const container = renderText('\x1b[31merror\x1b[0m ok');
    expect(container.textContent).toBe('error ok');
    const span = container.querySelector('span');
    expect(span?.textContent).toBe('error');
    expect(span?.style.color).toBe('rgb(187, 0, 0)');
  });

  it('maps SGR decorations to CSS styles', () => {
    const container = renderText('\x1b[1mbold\x1b[0m');
    expect(container.querySelector('span')?.style.fontWeight).toBe('bold');
  });
});
