// Batch/job-group states use a distinct, lowercase vocabulary ('success', 'failure',
// 'cancelled', 'running') from job states ('Success', 'Failed', ... — see StateIcon in
// batch/components), so they need their own icon mapping rather than sharing that switch.
export function BatchStateIcon({ state }: { state: string }): JSX.Element | null {
  switch (state) {
    case 'success':
      return <span className="material-symbols-outlined align-middle text-lg text-green-600">check_circle</span>;
    case 'failure':
    case 'error':
      return <span className="material-symbols-outlined align-middle text-lg text-red-600">cancel</span>;
    case 'cancelled':
      return <span className="material-symbols-outlined align-middle text-lg text-zinc-400">block</span>;
    default:
      return null;
  }
}
