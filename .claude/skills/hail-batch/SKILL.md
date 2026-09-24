---
description: Submit jobs, debug failures, and write Hail Batch submission scripts as an end user
---

You are helping a user work with Hail Batch — a job orchestration service for running analysis pipelines on cloud compute.

## Running hailctl commands

If `hailctl` isn't found, it may be in a virtual environment — look for one and activate it. Once activated, it stays active for the session.

**Never suppress stderr** when running hailctl or other Hail tools — important connection info and warnings appear there. Do not use `2>&1` or `2>/dev/null`.

## Authentication

Users need two things on GCP:

```bash
# 1. GCP Application Default Credentials (for cloud storage access)
gcloud auth application-default login

# 2. Hail service auth (reads ~/.hail/tokens.json)
hailctl auth login
```

Both are needed. If a user gets auth errors, check which one is missing.

## Investigating batches: use the CLI

For inspection tasks (checking status, reading logs, diagnosing failures), **always prefer `hailctl batch` CLI commands** — they're simpler and don't require writing Python.

### List recent batches
```bash
hailctl batch list                          # last 50 batches
hailctl batch list --limit 20 --query 'state=failure'
hailctl batch list -o json                  # machine-readable
```

### Get batch status
```bash
hailctl batch get BATCH_ID                  # full details (yaml)
hailctl batch get BATCH_ID -o json
```

### Inspect a specific job
```bash
hailctl batch job BATCH_ID JOB_ID           # status + spec (yaml)
```

### Get logs

Always specify `--container` — the bare command dumps all containers as escaped YAML and is hard to read. If a job has been retried, use `--attempt` to get the right one (defaults to most recent).

```bash
hailctl batch attempts BATCH_ID JOB_ID                              # list attempts + IDs (only if needed)
hailctl batch log BATCH_ID JOB_ID --container main                  # user code (start here)
hailctl batch log BATCH_ID JOB_ID --container main --attempt ID     # specific attempt
hailctl batch log BATCH_ID JOB_ID --container input                 # file copies in
hailctl batch log BATCH_ID JOB_ID --container output                # file copies out
```

### Wait for a batch to complete
```bash
hailctl batch wait BATCH_ID
hailctl batch wait BATCH_ID -o json
```

### Cancel or delete
```bash
hailctl batch cancel BATCH_ID
hailctl batch delete BATCH_ID
```

### Billing
```bash
hailctl batch billing list
hailctl batch billing get BILLING_PROJECT
```

## Writing batch submission scripts: use the Python API

For writing scripts that *submit* batches programmatically, use `hailtop.batch`:

```python
import hailtop.batch as hb

b = hb.Batch(
    name='my-analysis',
    backend=hb.ServiceBackend(
        billing_project='my-project',
        remote_tmpdir='gs://my-bucket/tmp',
    ),
)

j = b.new_job(name='step1')
j.image('hailgenetics/hail:0.2.x')
j.memory('4Gi')
j.cpu(2)
j.command('python3 my_script.py')

b.run()
```

For quick one-off jobs, `hailctl batch submit` works too:
```bash
hailctl batch submit --image hailgenetics/hail:0.2.x --memory 4Gi python3 my_script.py
```

### Hail Query within a Batch job

Use the `hailgenetics/hail:<version>` image. Write outputs to cloud storage and read them back:

```python
# In the batch job script (runs on the worker):
import hail as hl
hl.init()
mt = hl.read_matrix_table('gs://bucket/input.mt')
results = mt.some_analysis()
results.write('gs://bucket/output.ht')

# Back in the parent script after b.run():
results = hl.read_table('gs://bucket/output.ht')
```

## Direct API access with hailctl curl

For endpoints not covered by `hailctl batch` subcommands, use `hailctl curl` — it handles auth automatically:

```bash
hailctl curl default batch /PATH
```

To discover all available endpoints, fetch the OpenAPI spec:
```bash
hailctl curl default batch /openapi.yaml
```

## Listing jobs within a batch

```bash
hailctl batch jobs BATCH_ID                          # first 50 jobs
hailctl batch jobs BATCH_ID --state bad              # failed or errored jobs
hailctl batch jobs BATCH_ID --state bad -o json      # machine-readable
hailctl batch jobs BATCH_ID --exit-code 137          # OOM kills
hailctl batch jobs BATCH_ID --name my-step           # by job name
hailctl batch jobs BATCH_ID --limit 0                # all jobs (no limit)
hailctl batch jobs BATCH_ID --limit 50 --last-job-id 50  # pagination: next page after job 50
```

Valid `--state` values: `pending`, `ready`, `creating`, `running`, `live` (ready+creating+running), `cancelled`, `error`, `failed`, `bad` (error+failed), `success`, `done` (all terminal states).

Flags can be combined: `--state bad --exit-code 1` returns jobs that are bad AND have exit code 1.

## Job states

| State | Meaning |
|---|---|
| `Pending` | Waiting on dependencies |
| `Ready` | Dependencies met, queued for a worker |
| `Creating` | Worker being provisioned |
| `Running` | Executing on a worker |
| `Success` | Exited 0 |
| `Failed` | User code exited non-zero |
| `Error` | Infrastructure/platform failure (not user code) |
| `Cancelled` | Cancelled before completion |

## Common failure patterns

| Symptom | Likely cause | Action |
|---|---|---|
| `Failed`, exit code 1 | Script error | `hailctl batch log BATCH JOB --container main` |
| `Failed`, exit code 137 | OOM kill | Increase `j.memory()` in the submission script |
| `Error` state | Infrastructure failure | Check `--container input` or `--container output` logs; may be transient |
| Jobs stuck in `Pending` | Dependency failure or quota | Check upstream jobs |
| High cost, slow progress | Insufficient parallelism | Increase job fan-out in the script |

## How to investigate a specific batch

When given a batch ID:
1. Run `hailctl batch get BATCH_ID` — report state, n_jobs, n_succeeded, n_failed, cost
2. Run `hailctl batch jobs BATCH_ID --state bad` to find failed/errored jobs
3. For each bad job (up to ~5), run `hailctl batch log BATCH_ID JOB_ID --container main`
4. Identify the root cause and suggest next steps
