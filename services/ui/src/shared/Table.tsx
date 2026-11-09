// Table primitives shared across services: grey header bar, striped rows, row hover.

export const TH_BASE = 'whitespace-nowrap px-2.5 py-0.5 text-white bg-zinc-500 font-normal text-left border-r border-zinc-400 last:border-r-0';
export const TD_BASE = 'px-2.5 py-0.5';

export function Th({ children, className = '', colSpan }: { children?: React.ReactNode; className?: string; colSpan?: number }): JSX.Element {
  return <th colSpan={colSpan} className={`${TH_BASE} ${className}`}>{children}</th>;
}

export function Td({ children, className = '' }: { children?: React.ReactNode; className?: string }): JSX.Element {
  return <td className={`${TD_BASE} ${className}`}>{children}</td>;
}

export function DataTr({ children }: { children: React.ReactNode }): JSX.Element {
  return <tr className="even:bg-zinc-100 hover:bg-zinc-200">{children}</tr>;
}

// Placeholder for a cell with no value, so empty columns still read as columns.
export function EmptyCell(): JSX.Element {
  return <span className="text-zinc-300">&ndash;</span>;
}
