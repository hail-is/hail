// Labels CI acts on, by what they do (label names from `ci/ci/github.py`).
export type LabelEffect = 'blocks-merge' | 'priority' | 'test-control';

const LABEL_EFFECTS = new Map<string, { effect: LabelEffect; description: string }>([
  ['WIP', { effect: 'blocks-merge', description: 'CI won\'t merge this PR' }],
  ['stacked PR', { effect: 'blocks-merge', description: 'CI won\'t merge this PR' }],
  ['prio:high', { effect: 'priority', description: 'built and merged ahead of other PRs' }],
  ['do-not-test', { effect: 'test-control', description: 'CI won\'t build or test this PR' }],
  ['rerun all tests', { effect: 'test-control', description: 'CI runs the full test suite, not just affected steps' }],
]);

export function labelEffect(label: string): { effect: LabelEffect; description: string } | undefined {
  return LABEL_EFFECTS.get(label);
}

const EFFECT_ORDER: LabelEffect[] = ['blocks-merge', 'priority', 'test-control'];

function rank(label: string): number {
  const known = labelEffect(label);
  return known === undefined ? EFFECT_ORDER.length : EFFECT_ORDER.indexOf(known.effect);
}

// Labels CI acts on first, most significant first, then the rest alphabetically.
export function sortLabels(labels: string[]): string[] {
  return [...labels].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}

// Only the labels CI acts on, most significant first.
export function visibleLabels(labels: string[]): string[] {
  return sortLabels(labels.filter((l) => labelEffect(l) !== undefined));
}
