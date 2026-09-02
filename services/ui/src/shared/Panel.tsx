import { useState } from 'react';

interface PanelProps { title: string; subtitle?: string; collapsible?: boolean; viewSelector?: React.ReactNode; children: React.ReactNode }

export function Panel({ title, subtitle, collapsible = false, viewSelector, children }: PanelProps) {
  const [collapsed, setCollapsed] = useState(false);
  return (
    <div className="bg-white rounded-lg border border-zinc-200 shadow-sm overflow-hidden">
      <div className={`px-5 py-4 bg-zinc-50 flex items-center gap-2 ${!collapsed ? 'border-b border-zinc-200' : ''}`}>
        {collapsible && (
          <button onClick={() => setCollapsed(c => !c)} className="p-1 rounded hover:bg-zinc-200 text-zinc-400 transition-colors">
            <svg className={`w-4 h-4 transition-transform ${collapsed ? '' : 'rotate-90'}`} viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M7.293 4.293a1 1 0 011.414 0l5 5a1 1 0 010 1.414l-5 5a1 1 0 01-1.414-1.414L11.586 10 7.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
        )}
        <div className="flex-1">
          <h2 className="text-base font-semibold text-zinc-800">{title}</h2>
          {subtitle && <p className="text-xs text-zinc-500 mt-0.5">{subtitle}</p>}
        </div>
        {viewSelector && <div onClick={e => e.stopPropagation()}>{viewSelector}</div>}
      </div>
      {!collapsed && <div className="px-5 py-3">{children}</div>}
    </div>
  );
}
