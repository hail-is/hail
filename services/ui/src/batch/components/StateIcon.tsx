import { SpinnerIcon } from '../../shared/SpinnerIcon';

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
    return <SpinnerIcon className="text-sky-600" />;
  }
  const icon = stateIcon(state);
  const color = icon === 'schedule' ? 'text-zinc-400' : stateColor(state);
  return <span className={`material-symbols-outlined text-base leading-none ${color}`}>{icon}</span>;
}
