// Same symbols as the merge eligibility list on the PR details page.
export function ReviewStateIcon({ state }: { state: string | null }): JSX.Element | null {
  switch (state) {
    case 'approved':
      return <span className="material-symbols-outlined align-middle text-lg text-green-600">check_circle</span>;
    case 'changes_requested':
      return <span className="material-symbols-outlined align-middle text-lg text-red-600">block</span>;
    case 'pending':
      return <span className="material-symbols-outlined align-middle text-lg text-zinc-400">pending</span>;
    default:
      return null;
  }
}
