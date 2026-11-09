import { labelEffect, type LabelEffect } from './labelModel';

const EFFECT_CLASSES = new Map<LabelEffect, string>([
  ['blocks-merge', 'bg-red-100 text-red-800 border-red-200'],
  ['priority', 'bg-amber-100 text-amber-800 border-amber-200'],
  ['test-control', 'bg-zinc-100 text-zinc-500 border-zinc-200'],
]);

const UNKNOWN_CLASSES = 'bg-zinc-100 text-zinc-600 border-zinc-200';

// A GitHub label as a pill, colored by what it makes CI do.
export function LabelChip({ label }: { label: string }): JSX.Element {
  const known = labelEffect(label);
  const classes = (known && EFFECT_CLASSES.get(known.effect)) ?? UNKNOWN_CLASSES;
  return (
    <span
      title={known ? `${label}: ${known.description}` : label}
      className={`inline-block whitespace-nowrap rounded-full border px-2 py-px text-xs font-medium ${classes}`}
    >
      {label}
    </span>
  );
}
