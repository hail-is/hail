import { SpinnerIcon } from '../../shared/SpinnerIcon';

export function stateColor(state: string): string {
  switch (state) {
    case 'Success': return 'text-green-600';
    case 'Running': case 'Creating': return 'text-sky-600';
    case 'Failed': case 'Error': return 'text-red-600';
    case 'Cancelled': return 'text-yellow-600';
    default: return 'text-zinc-600';
  }
}

function stateIcon(state: string): string {
  switch (state) {
    case 'Success': return 'check';
    case 'Failed': return 'close';
    case 'Error': return 'error';
    case 'Cancelled': return 'block';
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

export function BatchStateIcon(
  { state, nJobs, nCompleted }: { state: string; nJobs: number; nCompleted: number },
): JSX.Element | null {
  if (nJobs - nCompleted > 0) {
    return <SpinnerIcon className={state === 'failure' ? 'text-red-600' : 'text-sky-600'} />;
  }
  switch (state) {
    case 'success':
      return <span className="material-symbols-outlined text-base leading-none text-green-600">check</span>;
    case 'failure':
      return <span className="material-symbols-outlined text-base leading-none text-red-600">close</span>;
    case 'cancelled':
      return <span className="material-symbols-outlined text-base leading-none text-yellow-600">block</span>;
    default:
      return null;
  }
}
