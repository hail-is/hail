# Automatic apt Source Rewriting

On GCP, Hail Batch automatically rewrites apt sources in job containers to use the regional GCE
Ubuntu mirror instead of the default Canonical mirror (`archive.ubuntu.com`). This reduces
CloudNAT egress costs and improves `apt-get` performance.

## How It Works

At job submission time, if the job is targeting GCP and uses a shell-invoked command
(`[shell, '-c', script]`), the batch frontend prepends a `sed` one-liner to the job script that
rewrites `/etc/apt/sources.list` and all files in `/etc/apt/sources.list.d/`:

```bash
sed -i "s|archive\.ubuntu\.com|${HAIL_REGION}.gce.archive.ubuntu.com|g" \
  /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null || true
```

`HAIL_REGION` is an environment variable injected into every job container by the worker at
runtime, containing the GCP region the worker is running in (e.g. `us-central1`). Because it is
resolved at runtime, the correct regional mirror is used even if a job is retried in a different
region.

The rewrite is part of the stored job command and is visible in the **Command** section of the
Batch UI.

## Implementation

The rewrite is applied in `batch/batch/front_end/front_end.py`, in the job processing loop,
immediately after the Docker Hub image rewriting block. The logic is:

- Only applies when `cloud == 'gcp'`
- Only applies to `type == 'docker'` processes
- Only applies to shell-invoked commands (3-element list with `command[1] == '-c'`)
- Skipped if the job spec includes `apt_rewrite: false`

`HAIL_REGION` is set by the worker in `batch/batch/worker/worker.py` (`hail_extra_env`, alongside
`HAIL_BATCH_ID`, `HAIL_JOB_ID`, etc.).

## Opting Out

Jobs that need the canonical Canonical mirror can opt out via the `apt_rewrite` job parameter:

**Python batch client (`hailtop.batch`):**
```python
j = b.new_job()
j.apt_rewrite(False)
```

**Low-level batch client (`hailtop.batch_client`):**
```python
batch.create_job(image, command, apt_rewrite=False)
```

**Raw job spec:**
```json
{ "apt_rewrite": false, ... }
```

## Why Not Worker-Side?

An earlier implementation applied this rewrite in the worker process after receiving the job spec.
Moving it to the frontend means:

- The rewrite is visible in the Batch UI (Command section), so users can see exactly what runs
- It follows the same pattern as Docker Hub image rewriting
- The worker no longer silently mutates the command it was given
