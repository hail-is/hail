import { BatchStateIcon } from '../../shared/BatchStateIcon';
import { SpinnerIcon } from '../../shared/SpinnerIcon';

// A PR's CI build state or GitHub status: the batch states, plus the extra states CI reports
// when there's no finished batch to describe (`null` = no build yet) and GitHub's `pending`.
export function BuildStateIcon({ state }: { state: string | null }): JSX.Element | null {
  switch (state) {
    case null:
    case 'pending':
      return <span className="material-symbols-outlined align-middle text-lg text-zinc-400">schedule</span>;
    case 'building':
      return <SpinnerIcon className="inline-block align-middle text-sky-600" />;
    case 'unauthorized':
      return <span className="material-symbols-outlined align-middle text-lg text-amber-600">lock</span>;
    default:
      return <BatchStateIcon state={state} />;
  }
}
