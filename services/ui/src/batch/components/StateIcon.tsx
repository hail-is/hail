export function stateColor(state: string): string {
  switch (state) {
    case 'Success': return 'text-green-600';
    case 'Running': case 'Creating': return 'text-sky-600';
    case 'Failed': case 'Error': return 'text-red-600';
    case 'Cancelled': return 'text-zinc-400';
    default: return 'text-zinc-600';
  }
}

function stateIcon(state: string): string {
  switch (state) {
    case 'Success': return 'check';
    case 'Failed': case 'Error': return 'close';
    case 'Cancelled': return 'close';
    default: return 'schedule';
  }
}

export function StateIcon({ state }: { state: string }): JSX.Element {
  if (state === 'Running') {
    return (
      <svg className="animate-spin h-4 w-4 text-sky-600 flex-shrink-0" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
      </svg>
    );
  }
  const icon = stateIcon(state);
  const color = icon === 'schedule' ? 'text-zinc-400' : stateColor(state);
  return <span className={`material-symbols-outlined text-base leading-none ${color}`}>{icon}</span>;
}
