import { useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { createPortal } from 'react-dom';
import type { Batch } from '../../shared/batchApi';
import { SpinnerIcon } from '../../shared/SpinnerIcon';

export type BatchAction = 'cancel' | 'delete';

// Native list markers drift out of line with the text, so bullets are drawn alongside it instead.
function Bullets({ children }: { children: ReactNode }): JSX.Element {
  return <ul className="space-y-1.5">{children}</ul>;
}

function Bullet({ children }: { children: ReactNode }): JSX.Element {
  return (
    <li className="flex gap-2 leading-5">
      <span aria-hidden="true" className="flex h-5 shrink-0 items-center">
        <span className="block h-1.5 w-1.5 rounded-full bg-zinc-400" />
      </span>
      <span>{children}</span>
    </li>
  );
}

function Impact({ action, batch }: { action: BatchAction; batch: Batch }): JSX.Element {
  const unfinished = batch.n_jobs - batch.n_completed;
  if (action === 'cancel') {
    return (
      <Bullets>
        <Bullet>
          {unfinished === 1 ? 'The 1 unfinished job' : `All ${unfinished.toLocaleString()} unfinished jobs`} will be
          stopped. Running jobs are killed and their partial work is lost; jobs that haven&apos;t started yet will
          never run.
        </Bullet>
        <Bullet>Jobs marked <code className="text-[0.9em] leading-none">always_run</code> will still run.</Bullet>
        <Bullet>Jobs that have already finished keep their results and logs.</Bullet>
        <Bullet>This can&apos;t be undone. To continue the work you&apos;ll need to submit a new batch.</Bullet>
      </Bullets>
    );
  }
  return (
    <Bullets>
      <Bullet>
        The batch, its {batch.n_jobs.toLocaleString()} {batch.n_jobs === 1 ? 'job' : 'jobs'}, and their logs and
        statuses will no longer be visible in the UI or available through the API — for you or anyone else in
        the billing project.
      </Bullet>
      <Bullet>Cost already incurred stays billed to <span className="font-medium">{batch.billing_project}</span>.</Bullet>
      <Bullet>This can&apos;t be undone.</Bullet>
    </Bullets>
  );
}

// A modal that spells out what cancelling or deleting a batch does before doing it. Portalled to
// <body> so it doesn't inherit styles (e.g. text alignment) from wherever it's opened, like a table cell.
export function BatchActionDialog({ action, batch, onConfirm, onClose }: {
  action: BatchAction;
  batch: Batch;
  onConfirm: () => Promise<void>;
  onClose: () => void;
}): JSX.Element {
  const ref = useRef<HTMLDialogElement>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    ref.current?.showModal();
  }, []);

  const verb = action === 'cancel' ? 'Cancel' : 'Delete';
  const name = batch.attributes?.name;

  return createPortal(
    <dialog
      ref={ref}
      onClose={onClose}
      onCancel={(e) => { if (busy) e.preventDefault(); }}
      onClick={(e) => { if (e.target === ref.current && !busy) ref.current.close(); }}
      className="rounded-lg p-0 shadow-xl backdrop:bg-black/40 max-w-lg w-full text-left"
    >
      <div className="p-6 space-y-4 text-sm text-zinc-700">
        <h2 className="text-lg font-medium text-zinc-900">
          {verb} batch {batch.id}{name ? <span className="font-light"> ({name})</span> : null}?
        </h2>
        <Impact action={action} batch={batch} />
        <div className="flex justify-end gap-2 pt-2">
          <button
            type="button"
            onClick={() => { ref.current?.close(); }}
            disabled={busy}
            className="px-3 py-1.5 rounded border border-zinc-300 hover:bg-zinc-100 disabled:opacity-50"
          >
            Don&apos;t {action}
          </button>
          <button
            type="button"
            onClick={() => {
              setBusy(true);
              onConfirm().finally(() => { setBusy(false); ref.current?.close(); });
            }}
            disabled={busy}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded bg-red-600 text-white hover:bg-red-700 disabled:opacity-75"
          >
            {busy && <SpinnerIcon className="h-3 w-3" />}
            {verb} batch
          </button>
        </div>
      </div>
    </dialog>,
    document.body,
  );
}
