import { useState } from 'react';
import type { ReactNode } from 'react';

export function CollapsibleItem({ title, summary, children, startOpen = false }: {
  title: string;
  summary?: ReactNode;
  children: ReactNode;
  startOpen?: boolean;
}): JSX.Element {
  const [open, setOpen] = useState(startOpen);
  return (
    <li>
      <button
        onClick={() => { setOpen((o) => !o); }}
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
