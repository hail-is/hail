import { useState, ReactNode } from 'react';

export function CollapsibleItem({ title, summary, children, onExpand }: {
  title: string;
  summary?: ReactNode;
  children: ReactNode;
  // Fires on every closed→open transition (not on close, and not while already open — e.g. a
  // re-render). If the caller wants a lazy fetch that only ever runs once, it must track that
  // itself (e.g. skip if already fetched/in flight) — `children` itself still unmounts on
  // collapse, so any state a fetch result needs to survive across toggles must live in the
  // caller, not in a component nested inside `children`.
  onExpand?: () => void;
}): JSX.Element {
  const [open, setOpen] = useState(false);
  return (
    <li>
      <button
        type="button"
        onClick={() => {
          setOpen((o) => {
            if (!o) onExpand?.();
            return !o;
          });
        }}
        className="w-full flex justify-between items-center px-4 py-3 text-sm text-left hover:bg-slate-100"
      >
        <span className="font-medium">{title}</span>
        <div className="flex items-center gap-2 text-zinc-400 text-xs">
          {summary != null && <span>{summary}</span>}
          <span>{open ? '▴' : '▾'}</span>
        </div>
      </button>
      {open && <div className="px-4 pb-3">{children}</div>}
    </li>
  );
}
