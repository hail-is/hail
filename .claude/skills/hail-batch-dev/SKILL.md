---
description: Work on the Hail system itself — CI pipeline, flaky tests, Batch service internals, dev namespace deployments, deployed infrastructure management
---

You are helping a Hail service engineer investigate Hail Batch infrastructure or develop improvements to it.

**For end-user batch investigation (logs, job status, writing scripts), also invoke `/hail-batch`.**

## Vetted runbooks

Read the matching runbook before starting one of these tasks, and follow it. These have been checked step by step against production; other docs in `dev-docs/` may be out of date.

When you finish a runbook, report your results, then suggest a few ways to improve the next run: insights or hints that would have changed how this run went, rather than nits. When an improvement is a small, concrete change with a big payoff (e.g. one extra pattern in a grep), give the exact change.

| Task | Runbook | Last verified |
|---|---|---|
| Finding flaky CI tests and why they fail | `dev-docs/services/flaky_tests.md` | 2026-10-14 |

## Running hailctl commands

`hailctl` is likely in a virtual environment — if it isn't found, look for one and activate it. Once activated, it stays active for the session.

**Never suppress stderr** when running hailctl or other Hail tools — important connection info and warnings appear there. Do not use `2>&1` or `2>/dev/null`.

## What this adds over /hail-batch

- Namespace/deploy config targeting (non-default namespaces)
- Service-level debugging (batch-driver, workers, CloudSQL)
- ServiceBackend internals (how Hail Query drives Batch)
- CI batch investigation
- Developing and testing changes to the batch service itself

## Authentication

Same as end users, plus namespace-specific tokens:

```bash
gcloud auth application-default login       # GCP ADC for cloud storage
hailctl auth login                          # Hail service auth
```

To check which namespaces you have tokens for and confirm your current identity:
```bash
hailctl auth list    # lists namespaces (* = current)
hailctl auth user    # shows current user identity
```

To switch namespaces persistently:
```bash
hailctl dev config set default_namespace my-namespace
hailctl dev config list                             # show current settings
```

Or per-command with an env var:
```bash
HAIL_DEFAULT_NAMESPACE=my-namespace hailctl batch list
HAIL_DEFAULT_NAMESPACE=my-namespace hailctl batch get BATCH_ID
```

For engineers working across multiple namespaces regularly, config profiles are useful:
```bash
hailctl config profile create my-namespace
hailctl config set batch/billing_project my-project
hailctl config profile load my-namespace
hailctl config profile list                         # * marks active profile
hailctl config profile load default                 # switch back
```

Or in Python:
```python
from hailtop.config import DeployConfig
from hailtop.batch_client import client as bc

deploy_config = DeployConfig(
    location='external',
    domain='hail.is',
    base_path='',
    default_namespace='my-namespace',
)
with bc.BatchClient(billing_project='my-project', deploy_config=deploy_config) as client:
    batch = client.get_batch(BATCH_ID)
```

## Troubleshooting connection errors

If you get "no upstream", "connection refused", or "batch not found" errors, the most likely causes are:
- The target namespace is not running (dev namespaces are ephemeral and may have expired)
- `hailctl dev config` is pointing at the wrong namespace — run `hailctl dev config list` to check
- The `HAIL_DEFAULT_NAMESPACE` env var is overriding your config unexpectedly

Check with `hailctl dev config list` first before digging deeper.

If the namespace has expired or doesn't exist, it needs to be started. There are two options:

**Start from scratch** (deploys services via CI into a fresh namespace — takes a while, returns a batch ID to track progress):
```bash
HAIL_DEFAULT_NAMESPACE=default hailctl dev deploy \
  -b <github_user>/hail:<branch> \
  -s deploy_batch,add_developers   # adjust steps as needed
```
This prints a batch ID. Give the user the exact command to monitor it in a separate terminal:
```bash
hailctl batch wait <id>
```
Then ask whether they'd like you to watch it too and verify the deployment looks good when it comes up (useful if they want to iterate on failures). Waiting can take 20-30 minutes, so only do it if the user explicitly asks.

**Update a running namespace live** (faster, redeploys a single service):
```bash
make -C <service> deploy NAMESPACE=<my-namespace>   # e.g. make -C batch deploy NAMESPACE=chrisl
```

**Important**: both commands affect live infrastructure. Do not run them without explicit confirmation from the user — either ask the user to run the command themselves, or show the exact command and wait for approval before executing.

## Direct API access with hailctl curl

Prefer a native `hailctl` command when one exists. When it doesn't, `hailctl curl` calls any service endpoint and handles auth for you:

```bash
# hailctl curl NAMESPACE SERVICE PATH [curl args]
hailctl curl default ci /openapi.yaml -s -o ci-openapi.yaml   # extra curl flags go after the path
```

### Discovering endpoints

Each service publishes an OpenAPI spec, which is the authoritative list of its endpoints (prefer it over any list in documentation):

```bash
hailctl curl default batch /openapi.yaml
hailctl curl default auth /openapi.yaml
hailctl curl default ci /openapi.yaml
hailctl curl default batch-driver /openapi.yaml
hailctl curl default monitoring /openapi.yaml
```

Specs list mutating operations (`POST`, `DELETE`, ...) right next to read-only ones. `hailctl curl` sends a GET by default; make any other call only with the user's explicit go-ahead.

If you find yourself making the same `hailctl curl` call repeatedly, it may be worth adding a native command (see *Enhancing the hailctl CLI* below).

**When adding a new endpoint**, update the service's `openapi.yaml` in the same PR. The specs are Jinja templates next to each service's other templates (e.g. `batch/batch/front_end/templates/openapi.yaml`, `ci/ci/templates/openapi.yaml`).

**When an endpoint doesn't match the spec** (wrong path, missing params, incorrect response shape), fix the spec too — a drift between implementation and spec is a bug.

## Investigating with the CLI (prefer this for inspection)

The `hailctl batch` CLI works against any namespace via `HAIL_DEFAULT_NAMESPACE`. Prefer it for inspection — see `/hail-batch` for the full command reference.

To list jobs within a batch, use `hailctl batch jobs` — it pages through results for you:

```bash
hailctl batch jobs BATCH_ID --state bad              # failed or errored jobs
hailctl batch jobs BATCH_ID --state live --limit 0   # everything still pending/running
hailctl batch jobs BATCH_ID --name test_batch_db -o json | jq '.[] | {job_id, state, exit_code}'
```

If you need the raw API, use `/api/v2alpha/.../jobs` (what the CLI uses). Its `q` filter takes newline-separated terms like `state=bad` or `exit_code=1`. Don't send that syntax to the v1alpha endpoints: v1 reads `key=value` as a job *attribute* match, so `state=failed` silently returns no jobs (v1 wants a bare word like `failed`).

For anything the CLI can't reach (job attempts, raw DB state, worker-level details), drop into the Python client:

```python
from hailtop.batch_client import client as bc

with bc.BatchClient(billing_project='my-project') as client:
    job = client.get_job(BATCH_ID, JOB_ID)
    attempts = job.attempts()           # retry history
    usage = job.resource_usage()        # per-container resource metrics
    status = job.status()               # full status dict including worker name
```

### OOM diagnosis
```python
status = job.status()
# Worker-level OOM (job killed by cgroup, not user code):
oom = status.get('status', {}).get('main', {}).get('out_of_memory')
# Or use the helper:
from hailtop.batch_client.aioclient import Job
oom = Job._get_out_of_memory(status, 'main')
```

### Worker identification
```python
worker = status.get('status', {}).get('main', {}).get('worker')
# Use this to find worker VM logs in Cloud Logging
```

## Worker-level logs (GCP Cloud Logging)

Worker logs aren't in the batch API — go to Cloud Logging:
```bash
gcloud logging read \
  'resource.type="gce_instance" resource.labels.instance_name="<worker-name>"' \
  --project=<gcp-project> --limit=100
```

## Batch service architecture

- **`batch`**: API server — handles client requests, validates, persists to MySQL
- **`batch-driver`**: Schedules jobs, manages the state machine, provisions/deprovisions workers
- **workers**: GCE VMs that pull and execute jobs

State machine:
```
Pending → Ready → Creating → Running → {Success, Failed, Error, Cancelled}
```

`Error` = infrastructure failure (batch-driver or worker), not user code.

Key source directories:
- `batch/batch/` — API server and driver Python code
- `batch/jvm-entryway/` — Scala JVM worker entrypoint (for Query jobs)
- `batch/worker/` — Worker Python code

## ServiceBackend internals

When users do `hl.init(backend='batch')`, Hail Query uses `ServiceBackend` to:
1. Serialize the query IR to cloud storage
2. Submit a Batch job via `hailtop.batch_client.aioclient.BatchClient`
3. The JVM worker (`jvm-entryway`) reads the IR, executes, writes results back
4. Python side reads results from cloud storage

Key source: `hail/python/hail/backend/service_backend.py`

To find the batch submitted by a Query operation, look for batches with names/attributes matching the IR token logged by the ServiceBackend.

## Database

Batch state lives in MySQL (Google CloudSQL). Schema: `batch` database.

Key tables: `batches`, `jobs`, `job_groups`, `attempts`, `resources`, `billing_projects`

To connect (requires port-forward or `make local-mysql`):
```bash
kubectl -n <namespace> port-forward svc/db 3306:3306
mysql -h 127.0.0.1 -P 3306 -u batch -pbatch batch
```

## kubectl

```bash
# Pod logs
kubectl -n <namespace> logs -l app=batch --tail=200
kubectl -n <namespace> logs -l app=batch-driver --tail=200

# Pod status
kubectl -n <namespace> get pods -l app=batch
kubectl -n <namespace> get pods -l app=batch-driver

# Port-forward batch API
kubectl -n <namespace> port-forward svc/batch 5000:5000
```

## Developing changes to batch

### Local dev loop
```bash
# Sync Python changes to running pods without rebuilding images
python3 devbin/sync.py \
  --namespace <ns> \
  --app batch --app batch-driver \
  --path batch/batch /opt/venv/lib/python3.11/site-packages/

# Deploy a full image rebuild to a dev namespace
HAIL_DEFAULT_NAMESPACE=default hailctl dev deploy \
  -b <github_user>/hail:<branch> \
  -s deploy_batch
```

### Running batch tests
```bash
# Integration tests against a deployed namespace
make -C hail pytest-qob NAMESPACE=default

# Batch service tests
make -C hail jvm-test   # if touching JVM/Scala side
```

### DB migrations

New migrations go in `batch/sql/`. They're applied by the `*_database` CI steps. For dev namespaces, apply manually:
```bash
mysql -h ... batch < batch/sql/your-migration.sql
```

## CI batch investigation

CI submits batches to the `ci` billing project. To investigate a CI build failure:
1. Find the batch ID from the CI UI or `ci/ci/ci.py` logs
2. Use `HAIL_DEFAULT_NAMESPACE=default hailctl batch get BATCH_ID`
3. CI batches are large — filter to failed jobs and check their logs
4. Job names correspond to step names in `build.yaml`

## Enhancing the hailctl CLI

If you hit a gap in the CLI during an investigation (something you can only reach through the Python client or `hailctl curl`), it may be worth adding the missing command. The CLI lives in `hail/python/hailtop/hailctl/` and is straightforward to extend.

That said, CLI additions should be a **separate PR** from whatever you're currently working on — note it as a follow-up rather than doing it now.

## Common dev-level failure patterns

| Symptom | Likely cause | Action |
|---|---|---|
| Many `Error` jobs | batch-driver bug or worker crash | Check batch-driver pod logs |
| Jobs stuck in `Creating` | Worker pool exhausted or GCE quota | Check batch-driver scheduling logs |
| Worker VM crash (OOM) | Worker itself OOM (not the job) | Cloud Logging for the worker VM |
| Auth failures in jobs | SA key missing/expired | Check `create_test_gsa_keys` CI step |
| ServiceBackend hangs | IR too large or batch not completing | Find the submitted batch; check driver logs |
| `make local-mysql` fails | Port conflict or prior container | `docker ps` and kill the old container |
