import { useLayoutEffect, useRef } from 'react';
import type { PrStack, PrStackEntry } from '../../shared/ciApi';
import { stackMetaItems, stackStrip } from './stackStripModel';

const STATE_ICON = new Map([
  ['OPEN', { name: 'radio_button_unchecked', className: 'text-green-600', label: 'open' }],
  ['MERGED', { name: 'merge', className: 'text-purple-600', label: 'merged' }],
  ['CLOSED', { name: 'block', className: 'text-red-600', label: 'closed' }],
]);

function StepLink({ entry, href, icon, label }: {
  entry: PrStackEntry | null;
  href: (_n: number) => string;
  icon: string;
  label: string;
}): JSX.Element {
  const className = 'flex items-center justify-center gap-1 w-full py-1 text-xs rounded border';
  if (entry === null) {
    return (
      <span className={`${className} border-zinc-100 text-zinc-300`}>
        <span aria-hidden="true" className="material-symbols-outlined text-base">{icon}</span>{label}
      </span>
    );
  }
  return (
    <a href={href(entry.number)} className={`${className} border-zinc-200 text-zinc-600 hover:border-sky-400 hover:text-sky-600`}>
      <span aria-hidden="true" className="material-symbols-outlined text-base">{icon}</span>{label} · #{entry.number}
    </a>
  );
}

function Tile({ entry, current, href }: { entry: PrStackEntry; current: boolean; href: string }): JSX.Element {
  const state = STATE_ICON.get(entry.state) ?? { name: 'help', className: 'text-zinc-400', label: entry.state.toLowerCase() };
  const muted = entry.state !== 'OPEN';
  const look = current
    ? 'border-sky-500 bg-sky-50 ring-1 ring-sky-500'
    : 'border-zinc-200 hover:border-sky-400 hover:bg-zinc-50';
  return (
    <a
      href={href}
      title={`${entry.title} (${state.label})`}
      aria-current={current ? 'page' : undefined}
      className={`block rounded border px-2 py-1.5 ${look} ${muted && !current ? 'opacity-60' : ''}`}
    >
      <div className="flex items-center gap-1 text-xs text-zinc-500">
        <span className="tabular-nums w-5">{entry.position}</span>
        <span aria-hidden="true" className={`material-symbols-outlined text-sm ${state.className}`}>{state.name}</span>
        <span className="font-medium text-zinc-700">#{entry.number}</span>
        {muted && <span>· {state.label}</span>}
      </div>
      <div className="text-xs text-zinc-700 line-clamp-2 break-words">{entry.title}</div>
    </a>
  );
}

// The PR's native stack as a scrolling column of tiles, top of the stack first, opened with the
// current PR in view.
export function StackStrip({ stack, prNumber, position, basePath, wbIndex }: {
  stack: PrStack;
  prNumber: number;
  position: number | null;
  basePath: string;
  wbIndex: string;
}): JSX.Element {
  const model = stackStrip(stack, prNumber, position);
  const href = (n: number): string => `${basePath}/watched_branches/${wbIndex}/pr/${n}`;
  const listRef = useRef<HTMLDivElement>(null);
  const currentRef = useRef<HTMLDivElement>(null);
  const merged = stack.entries.filter((e) => e.state === 'MERGED').length;

  // Scroll only the list, not the page (scrollIntoView would move the page too).
  useLayoutEffect(() => {
    const list = listRef.current;
    const tile = currentRef.current;
    if (!list || !tile) return;
    list.scrollTop = tile.offsetTop - (list.clientHeight - tile.clientHeight) / 2;
  }, [prNumber]);

  return (
    <section aria-label="Stack" className="rounded border border-amber-300 bg-white p-2 text-sm">
      <div className="mb-2">
        <div className="flex items-center gap-1 text-xs uppercase tracking-wide text-amber-700">
          <span aria-hidden="true" className="material-symbols-outlined text-sm">stacks</span>Stack
        </div>
        <div className="font-semibold text-zinc-800 break-words">{stack.name ?? 'Unnamed stack'}</div>
        <div className="text-xs text-zinc-500">
          {model.current !== null ? `${model.current.position} of ${stack.size}` : `${stack.size} PRs`}
          {merged > 0 && ` · ${merged} merged`} · on {stack.base_ref_name}
        </div>
        {stack.doc_url !== null && (
          <a href={stack.doc_url} target="_blank" rel="noopener noreferrer" className="text-xs text-sky-600 hover:underline">
            reference doc
          </a>
        )}
        {stackMetaItems(stack).map((item) => (
          <div key={item.key} className="text-xs text-zinc-500 break-words">
            {item.key}:{' '}
            {item.isLink
              ? <a href={item.value} target="_blank" rel="noopener noreferrer" className="text-sky-600 hover:underline">{item.value}</a>
              : item.value}
          </div>
        ))}
      </div>
      <StepLink entry={model.above} href={href} icon="arrow_upward" label="up" />
      <div ref={listRef} className="relative my-1 max-h-[60vh] overflow-y-auto space-y-1 py-1">
        {model.entries.map((entry) => {
          const current = entry.number === prNumber;
          return (
            <div key={entry.number} ref={current ? currentRef : undefined}>
              <Tile entry={entry} current={current} href={href(entry.number)} />
            </div>
          );
        })}
      </div>
      <StepLink entry={model.below} href={href} icon="arrow_downward" label="down" />
      <div className="mt-1 text-center text-xs text-zinc-400">{stack.base_ref_name}</div>
    </section>
  );
}
