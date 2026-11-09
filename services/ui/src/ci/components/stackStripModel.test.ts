import { describe, expect, it } from 'vitest';
import type { PrStack, PrStackEntry } from '../../shared/ciApi';
import { stackMetaItems, stackStrip } from './stackStripModel';

function entry(position: number, number: number, state = 'OPEN'): PrStackEntry {
  return { position, number, state, title: `PR ${number}` };
}

function stack(entries: PrStackEntry[], meta: Record<string, string> = {}): PrStack {
  return { number: 7, base_ref_name: 'main', size: entries.length, entries, name: 'A stack', doc_url: null, meta };
}

describe('stackStrip', () => {
  const s = stack([entry(1, 10, 'MERGED'), entry(2, 11, 'CLOSED'), entry(3, 12), entry(4, 13)]);

  it('lists the top of the stack first', () => {
    expect(stackStrip(s, 12, 3).entries.map((e) => e.number)).toEqual([13, 12, 11, 10]);
  });

  it('steps to adjacent entries whatever their state', () => {
    const m = stackStrip(s, 12, 3);
    expect(m.current?.number).toBe(12);
    expect(m.above?.number).toBe(13);
    expect(m.below?.number).toBe(11);
  });

  it('has nothing beyond the ends', () => {
    expect(stackStrip(s, 13, 4).above).toBeNull();
    expect(stackStrip(s, 10, 1).below).toBeNull();
  });

  it('falls back to the PR position when entries are capped', () => {
    const m = stackStrip(s, 99, 5);
    expect(m.current).toBeNull();
    expect(m.below?.number).toBe(13);
    expect(m.above).toBeNull();
  });

  it('has no neighbours without a position', () => {
    const m = stackStrip(s, 99, null);
    expect(m.above).toBeNull();
    expect(m.below).toBeNull();
  });
});

describe('stackMetaItems', () => {
  it('links values starting http', () => {
    expect(stackMetaItems(stack([], { tracking: 'https://x/1', owner: 'me' }))).toEqual([
      { key: 'tracking', value: 'https://x/1', isLink: true },
      { key: 'owner', value: 'me', isLink: false },
    ]);
  });
});
