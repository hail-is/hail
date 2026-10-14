# Flaky Tests

Runbook for finding flaky CI tests and working out why they fail.

When someone clicks "retry" on a PR, CI records every `Failed` or `Error` job in that PR's batch to the
`retried_tests` table. A test that keeps turning up there across many different PRs is probably flaky.

## Quick look: the dashboard

[ci.hail.is/flaky_tests](https://ci.hail.is/flaky_tests) (also under "Flaky Tests" in the CI menu) shows the same
data as a leaderboard of retried tests with daily charts, over a selectable time window (14 days by default). Use it to
get a feel for what is currently noisy. For a deeper investigation, use the API below, which makes it easy to slice the
data and follow each run through to its logs.

## Investigating with the API

You need `hailctl` (from the repo's virtualenv), `jq`, and the `READ_CI` permission.

### 1. Pull the last 30 days

`GET /api/v1alpha/retried_tests` defaults to the last 14 days, so pass `after`. It returns at most 500 rows per page;
follow `cursor` while `has_more` is true. The full parameter list is in CI's spec
(`hailctl curl default ci /openapi.yaml`). The commands below write and read files in the current directory, so
start in a scratch directory (e.g. `mkdir -p /tmp/flaky && cd /tmp/flaky`).

```bash
AFTER=$(date -u -v-30d +%Y-%m-%dT%H:%M:%SZ)   # GNU date: date -u -d '30 days ago' +%Y-%m-%dT%H:%M:%SZ
OUT=retried_tests.jsonl; : > "$OUT"; CURSOR=
while :; do
  RESP=$(hailctl curl default ci "/api/v1alpha/retried_tests?limit=500&after=$AFTER${CURSOR:+&cursor=$CURSOR}" -s)
  echo "$RESP" | jq -c '.rows[]' >> "$OUT"
  [ "$(echo "$RESP" | jq -r .has_more)" = true ] || break
  CURSOR=$(echo "$RESP" | jq -r .cursor)
done
```

Each row has `id`, `batch_id`, `job_id`, `job_name`, `state`, `exit_code`, `pr_number`, `target_branch`,
`source_branch`, `source_sha`, `retried_by` and `retried_at`.

**Done when** the loop exits with `has_more` false and `retried_tests.jsonl` holds one row per line.

### 2. Rank: consistently high, or recently spiking

```bash
python3 - retried_tests.jsonl <<'EOF'
import json, sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
rows = [json.loads(l) for l in open(sys.argv[1])]
week_ago = datetime.now(timezone.utc) - timedelta(days=7)
stats = defaultdict(lambda: {'n': 0, 'last7': 0, 'prs': set(), 'errors': 0})
for r in rows:
    s = stats[r['job_name']]
    s['n'] += 1
    s['prs'].add(r['pr_number'])
    s['errors'] += r['state'] == 'Error'
    s['last7'] += datetime.fromisoformat(r['retried_at']) >= week_ago
print(f"{'job_name':55} {'30d':>4} {'7d':>4} {'PRs':>4} {'Error':>5}")
for name, s in sorted(stats.items(), key=lambda kv: -kv[1]['n'])[:30]:
    # Spike: at least 3 in the last week, at a rate at least 1.5x that of the 23 days before
    spike = '  <- spike' if s['last7'] >= 3 and s['last7'] / 7 >= 1.5 * (s['n'] - s['last7']) / 23 else ''
    print(f"{name:55} {s['n']:4} {s['last7']:4} {len(s['prs']):4} {s['errors']:5}{spike}")
EOF
```

Reading the table:
- **Many distinct PRs** means the test is probably flaky. Many rows from **one PR** might mean that PR genuinely
  broke the test; treat with caution.
- **A spike** (at least 3 in the last week, at 1.5x or more the rate of the 23 days before) points at a recent change in code, infrastructure or a
  dependency. Check what merged around when it started.
- **`Error` rather than `Failed`** means the job didn't exit on its own: either the infrastructure failed (worker,
  preemption, image pull) or the job hit its timeout. Timeouts are a common cause of flakiness, so check the job's
  error for `timed out after` before blaming infrastructure.
- **Setup and cleanup steps** (`*_database`, `create_*`, `deploy_*`, `delete_*`, `cancel_*`) failing often usually
  indicate the test namespace itself was unhealthy, which then shows up as failures in many tests. To see which
  failures happened together, group the rows by batch:

  ```bash
  jq -r '"\(.batch_id) PR#\(.pr_number) \(.job_name)"' retried_tests.jsonl \
    | awk '{k=$1" "$2; a[k]=a[k]" "$3; n[k]++} END {for (k in a) if (n[k]>1) print n[k], k ":" a[k]}' \
    | sort -rn | head -20
  ```

  Cleanup steps (`cancel_all_running_test_batches`, `delete_test_billing_projects`) often top the ranking because
  they run even after something upstream broke the namespace. When the grouping shows them failing alongside setup
  or service failures, investigate that upstream cause first. They can still be flaky in their own right, so check
  any runs where they failed on their own.
- **Failures that were alone in their batch** can't be fallout from something else breaking in the same build. Which
  view is most revealing varies from one investigation to the next, so this is worth trying alongside the ranking
  and the batch grouping rather than instead of them:

  ```bash
  jq -s -r 'group_by(.batch_id)[] | select(length==1) | .[0] | "\(.job_name) \(.state) batch=\(.batch_id) job=\(.job_id) PR#\(.pr_number)"' retried_tests.jsonl | sort
  ```

**Done when** you have a shortlist of candidates: tests that are high or spiking, and spread across several PRs.

### 3. Dig into individual runs

Start by printing a one-line **signature** for every run of a candidate. Seeing all the runs side by side shows
patterns much more clearly than checking them one at a time. This loop is written for bash:

```bash
NAME=test_batch_invariants   # the candidate's job_name
ERROR_SIG='.[0] | "worker=\(.status.worker) " + ((([.status.container_statuses // {} | to_entries[] | .value.error // empty] + [.status.error // empty]) | .[0] // "-") | split("\n") | map(select(length>0)) | last)'
jq -r --arg n "$NAME" 'select(.job_name == $n) | "\(.batch_id) \(.job_id) \(.state)"' retried_tests.jsonl |
while read -r B J STATE; do
  if [ "$STATE" = Error ]; then
    SIG=$(hailctl batch job "$B" "$J" -o json | jq -r "$ERROR_SIG")
  else
    SIG=$(hailctl batch log "$B" "$J" --container main | sed 's/\x1b\[[0-9;]*m//g' \
      | grep -m1 -E '^E +[^ ]|\] Test is\.hail\..* failed:|^(FAILED|ERROR) ')
  fi
  echo "$B/$J $STATE: ${SIG:-(no summary line found; read the log)}"
done
```

What the signature shows:
- **`Failed` runs:** the first failure line of the test's log, with ANSI colour codes stripped. For pytest that is
  the first `E   ` line, which holds the actual assertion (the `FAILED` summary line truncates it). For Scala/JVM
  tests it is the `Test is.hail... failed:` line.
- **`Error` runs:** the worker, plus the last line of the error, e.g. `ContainerTimeoutError: timed out after 300s`.
  Printing the worker makes a single bad worker (e.g. a full disk) easy to spot.
- **`Error` runs with `worker=null`:** the job never got a container. The batch-driver failed to schedule it, and
  the error comes from the driver, e.g. failing to read a Kubernetes secret for the test namespace. Treat these as
  infrastructure problems, not as the test itself being flaky.

Then drill into individual runs:

```bash
# What else failed in the same batch
hailctl batch jobs BATCH_ID --state bad -o json | jq -c '.[] | {job_id, name, state, exit_code}'

# The full log, with colour codes stripped
hailctl batch log BATCH_ID JOB_ID --container main | sed 's/\x1b\[[0-9;]*m//g' | less

# For pytest, the full assertion and traceback
hailctl batch log BATCH_ID JOB_ID --container main | sed 's/\x1b\[[0-9;]*m//g' | grep -E '^E '
```

`hailctl batch job BATCH_ID JOB_ID` prints the job's full status and spec as YAML, including cost breakdowns, which
is long but has everything (timing, worker, exit code, errors).

Look for patterns across the runs:
- **The same error or assertion every time:** compare the end of the `main` log, or the pytest short summary.
- **Timeouts:** `Timeout (>Ns)` from pytest-timeout, or the step's `timeout:` in `build.yaml`.
- **Shared infrastructure:** the same worker or zone, or runs bunched in time (one bad hour hitting several PRs).
- **Failing together:** several tests failing in the same batch suggests one upstream cause, such as a
  test-namespace service or database going down mid-build. Kubernetes events for a PR namespace are in Cloud Logging
  under `resource.labels.namespace_name="pr-<N>-default-<suffix>"` (the namespace is the batch's `namespace`
  attribute).
- **Failed prerequisites:** a test whose setup step failed is not itself flaky.

**Done when** each candidate has either a named pattern backed by specific batch/job IDs, or a note of which runs you
checked and that no pattern emerged.

### 4. Propose next steps

Split the findings into:
- **Fixes the evidence supports directly:** a timeout that is too tight, a missing retry on a known-transient error,
  an ordering race in `build.yaml` (`dependsOn` / `after`), a resource request that is too small. Cite the batch/job
  IDs that show the problem.
- **Further investigations:** patterns that are suggestive but unconfirmed, and what data would confirm them (more
  runs, logs, Cloud Logging queries, the batch database).

Prefer fixing the root cause (infrastructure or test design) over adding retries.
