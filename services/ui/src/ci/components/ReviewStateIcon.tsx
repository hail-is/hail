import type { StageStatus } from './prTableModel';

// Same symbols as the merge eligibility list on the PR details page.
export function ReviewStateIcon({ status }: { status: StageStatus }): JSX.Element | null {
  switch (status) {
    case 'pass':
      return <span className="material-symbols-outlined align-middle text-lg text-green-600">check_circle</span>;
    case 'fail':
      return <span className="material-symbols-outlined align-middle text-lg text-red-600">block</span>;
    case 'pending':
      return <span className="material-symbols-outlined align-middle text-lg text-zinc-400">pending</span>;
    default:
      return null;
  }
}
