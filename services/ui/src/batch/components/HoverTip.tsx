import type { ReactNode } from 'react';

// A tooltip that appears immediately on hover, unlike a native `title` (which waits ~1s).
export function HoverTip({ tip, align = 'left', className = 'inline-block', children }: {
  tip: ReactNode;
  align?: 'left' | 'right';
  className?: string;
  children: ReactNode;
}): JSX.Element {
  return (
    <span className={`relative group/tip ${className}`}>
      {children}
      <span
        role="tooltip"
        className={`absolute ${align === 'right' ? 'right-0' : 'left-0'} top-full mt-1 hidden group-hover/tip:block bg-zinc-800 text-white text-xs font-normal rounded px-2 py-1 z-20 pointer-events-none whitespace-nowrap text-left`}
      >
        {tip}
      </span>
    </span>
  );
}
