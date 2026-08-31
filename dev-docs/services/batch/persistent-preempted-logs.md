# Persistent Logs for Preempted Jobs

Job logs are uploaded to GCS once at the end of execution. If a worker VM is preempted first, the
log is lost. This feature starts a dedicated `log-sync.sh` subprocess alongside each container that
incrementally syncs the log file to GCS on a size-tiered schedule, so the most recent snapshot
survives preemption.

The subprocess coordinates with worker.py solely through an instruction file written to
`/run/batch-worker/log-sync/` (outside job scratch space, unreachable by the user). The worker
writes `state=done` and sends `SIGUSR1` when the job finishes; the subprocess does a final upload
and exits. On preemption the worker sends `SIGUSR1` without marking done so the subprocess uploads
immediately rather than waiting out its sleep interval.

## Race Conditions

**SIGUSR1 arriving while gcloud cp is running, or in the gap between post-upload re-source and
sleep start.** Bash traps fire between commands, so a signal during `gcloud storage cp` is queued
and fires after the upload returns. A subtler race: SIGUSR1 fires after the post-upload re-source
(which saw `state=running`) but before `SLEEP_PID` is set, making `wakeup()` a no-op. Closed by a
second re-source immediately before the `sleep` call; the remaining window is microseconds.

**Bytes written to the log after the upload started but before the container exits.** `finish()`
writes `state=done` then sends SIGUSR1. The subprocess wakes, does upload N (possibly slightly
stale), re-sources, sees `state=done`, and does a **final upload N+1** capturing any trailing bytes.
The double-upload on termination is intentional.

**Transient gcloud failure.** The upload error is printed and the loop continues; the next cycle
retries. `set -euo pipefail` does not abort on upload failure.

**Very large files (xlarge tier, >50 MB, 10-min sleep) during preemption.** `wakeup()` interrupts
the sleep and starts an upload, but uploading a multi-gigabyte file within the 30s preemption
window is not guaranteed. The last completed upload survives; trailing bytes may be lost. Best-effort.

**gcloud cp still running when the 30s preemption window expires, or when `finish()` times out
(120 s).** If `finish()` times out, SIGKILL is sent to the bash process but not to `gcloud storage
cp` (a grandchild). The orphaned gcloud process finishes its upload after cleanup starts. Best-effort.

## Cost

- **Network (GCE → GCS, same region):** free.
- **Write operations:** ~$0.000005 per `gcloud storage cp`. Each sync overwrites the same GCS object
  (same path, same `attempt_id`), so storage cost scales with log size, not upload count. The
  size-tiered intervals keep Class A operation counts low at scale.
