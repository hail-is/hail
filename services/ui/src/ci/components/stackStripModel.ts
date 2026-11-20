import type { PrStack, PrStackEntry } from '../../shared/ciApi';

export interface StackStripModel {
  // Top of the stack first, so "up" in the strip is up the stack.
  entries: PrStackEntry[];
  current: PrStackEntry | null; // null when the PR isn't in `entries` (they're capped)
  above: PrStackEntry | null; // the adjacent entry, whatever its state
  below: PrStackEntry | null;
}

export function stackStrip(stack: PrStack, prNumber: number, position: number | null): StackStripModel {
  const entries = [...stack.entries].sort((a, b) => b.position - a.position);
  const current = entries.find((e) => e.number === prNumber) ?? null;
  const at = current?.position ?? position;
  const find = (p: number): PrStackEntry | null => entries.find((e) => e.position === p) ?? null;
  return {
    entries,
    current,
    above: at === null ? null : find(at + 1),
    below: at === null ? null : find(at - 1),
  };
}

export interface StackMetaItem {
  key: string;
  value: string;
  isLink: boolean;
}

// Stack Meta keys beyond name / reference doc (the API sends those separately).
export function stackMetaItems(stack: PrStack): StackMetaItem[] {
  return Object.entries(stack.meta).map(([key, value]) => ({ key, value, isLink: value.startsWith('http') }));
}
