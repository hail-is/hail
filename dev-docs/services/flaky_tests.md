# Flaky Tests

Runbook for finding flaky CI tests and working out why they fail.

When someone clicks "retry" on a PR, CI records every `Failed` or `Error` job in that PR's batch to the
`retried_tests` table. A test that keeps turning up there across many different PRs is probably flaky.

The goal of a sweep is a **ranked list of findings**, each written up as a work item with its evidence. Working out
the root cause of each finding and fixing it is downstream work, done per item. Keep the sweep itself broad and
shallow: enough digging to tell findings apart and weigh them, no more.

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
start in a scratch directory.

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

### 2. Triage by job: what is noisy, and what fails together

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
  any runs where they failed on their own. To see how much of each job's failure count is likely fallout, count its
  rows in batches where a setup step also failed:

  ```bash
  jq -s -r '
    (map(select(.job_name | test("_database$|^create_|^deploy_"))) | map(.batch_id) | unique) as $broken
    | group_by(.job_name)[]
    | "\(.[0].job_name)  \(map(select(.batch_id as $b | $broken | index($b))) | length)/\(length) alongside a setup failure"
  ' retried_tests.jsonl
  ```
- **Failures that were alone in their batch** can't be fallout from something else breaking in the same build. Which
  view is most revealing varies from one investigation to the next, so this is worth trying alongside the ranking
  and the batch grouping rather than instead of them:

  ```bash
  jq -s -r 'group_by(.batch_id)[] | select(length==1) | .[0] | "\(.job_name) \(.state) batch=\(.batch_id) job=\(.job_id) PR#\(.pr_number)"' retried_tests.jsonl | sort
  ```

Job names are not tests: one job can be a shard running many tests, and one failure can cascade into several jobs.
This step only tells you where to look.

**Done when** you have a shortlist of jobs worth a closer look (high, spiking, or spread across several PRs) and know
which failures are likely fallout from something upstream.

### 3. Turn runs into findings

A **signature** is a one-line summary of why a run failed: which test, and the cause (the failing assertion, the
error, the timeout). Pulling a signature for every run of the shortlisted jobs, and grouping runs that share one,
turns "this job fails a lot" into **findings**: a specific test or step, failing for a specific reason, in these runs.
Seeing the runs side by side is what makes patterns visible; checking runs one at a time rarely does.

Where to find a signature:
- **`Failed` runs:** the test's log (`hailctl batch log BATCH_ID JOB_ID --container main`). Strip ANSI colour codes
  first (`sed 's/\x1b\[[0-9;]*m//g'`). For pytest, the test ID is on the `FAILED path::test` summary line and the
  cause on the first `E   ` line (the summary line truncates it). For Scala/JVM tests, look for
  `Test is.hail... failed:` lines. For other jobs (setup steps and scripts), the cause is usually the last line of
  the log, often the final line of a Python traceback.
- **`Error` runs:** the job's status (`hailctl batch job BATCH_ID JOB_ID -o json`). The cause is usually the last line
  of a container's `error`, e.g. `timed out after 300s`; note the `worker` too, since a single bad worker shows up
  as a cluster. If `worker` is null the job never got a container, and the batch-driver's reason is in the top-level
  `status.error`. Treat these as infrastructure, not the test.

For example, this bash loop prints a signature per run for one job (change the `select` to cover more):

```bash
ERROR_SIG='.[0] | "worker=\(.status.worker) " + ((([.status.container_statuses // {} | to_entries[] | .value.error // empty] + [.status.error // empty]) | .[0] // "-") | split("\n") | map(select(length>0)) | last)'
jq -r 'select(.job_name == "test_batch_1") | "\(.batch_id) \(.job_id) \(.job_name) \(.state)"' retried_tests.jsonl |
while read -r B J NAME STATE; do
  if [ "$STATE" = Error ]; then
    SIG=$(hailctl batch job "$B" "$J" -o json | jq -r "$ERROR_SIG")
  else
    LOG=$(hailctl batch log "$B" "$J" --container main | sed 's/\x1b\[[0-9;]*m//g')
    SIG=$(echo "$LOG" | grep -E '^E +[^ ]|\] Test is\.hail\..* failed:|^(FAILED|ERROR) ' | head -3 | tr '\n' ' ')
    [ -n "$SIG" ] || SIG=$(echo "$LOG" | grep -v '^[[:space:]]*$' | tail -1)
  fi
  echo "$B/$J $NAME $STATE: ${SIG:-(empty log)}"
done
```

When a run needs a closer look, `hailctl batch jobs BATCH_ID --state bad` shows what else failed in the same batch,
and `hailctl batch job BATCH_ID JOB_ID` shows everything about one run (timing, worker, exit code, errors).

Patterns that separate findings:
- **The same test and cause every time:** a flaky test, or a timeout that is too tight.
- **Shared infrastructure:** the same worker or zone, or runs bunched in time (one bad hour hitting several PRs).
- **Failing together:** several jobs failing in the same batch suggests one upstream cause, such as a
  test-namespace service or database going down mid-build. Kubernetes events for a PR namespace are in Cloud Logging
  under `resource.labels.namespace_name="pr-<N>-default-<suffix>"` (the namespace is the batch's `namespace`
  attribute).
- **Failed prerequisites:** a test whose setup step failed is not itself flaky.

**Done when** every shortlisted run has a signature, and the runs are grouped into findings: each one a test or step,
a cause, and the batch/job IDs that show it. Stop digging here; root-causing a finding is part of its work item.

### 4. Rank the findings

Weigh each finding by its cost to the team, using what you already have:
- **Retries caused:** how many retries it accounts for, and across how many PRs. This is the main cost: each retry
  means a new CI batch (a whole build, or a tactical rebuild of the failed steps), and the PR waits for it. A flake
  that fails within a minute still costs that whole rebuild, so judge it by its retries, not its own runtime.
- **Time to failure:** a secondary cost. A job that runs for an hour before failing (`duration` in
  `hailctl batch job`) delays the retry by that hour.
- **Trend:** steady, spiking, or fading out (perhaps already fixed).
- **Already tracked?** Search issues and PRs, both open and closed, for the test or step name (e.g.
  `gh search issues --repo hail-is/hail <name>` and `gh search prs --repo hail-is/hail <name>`). A fix merged last
  week won't have cleared the earlier failures from the 30-day history, and an open issue or PR may already be
  tracking the problem. Link them from the work item instead of duplicating effort.

**Done when** you have a list of findings ordered by impact, each with its evidence and impact noted.

### 5. Hand off: present the findings and ask what to do next

Present the ranked findings, each drafted as a work item, then stop and ask whoever requested the sweep what to do
next: which findings to file (e.g. as GitHub issues), which to pick up now, and which to drop. Leave filing issues and
starting fixes until they have chosen. Logs can contain credentials (e.g. test database passwords), so redact them
from anything that will be posted publicly.

Draft each work item with:
- **What:** the test or step, and its signature.
- **Evidence:** the batch/job IDs, the PRs affected, and when it started.
- **Impact:** retries caused, PRs affected, and any extra delay from slow failures.
- **Already tracked:** related issues and PRs (open or closed, e.g. a recent fix), or a note that you searched and
  found none.
- **Hypothesis:** your best guess at the kind of problem (flaky test, timeout too tight, infrastructure, fallout from
  an upstream failure), and how confident you are.
- **Starting points** for whoever picks it up: the test's source (`git grep <test name>`), its step in `build.yaml`
  (`timeout:`, `alwaysRun`, `dependsOn`), `git log` on the test, step or code around when the problem started, and
  any related PRs or issues.

If a fix is obvious from the evidence (e.g. a timeout clearly too tight), suggest it in the work item, but leave
confirming it to whoever picks it up. Prefer fixing the root cause (infrastructure or test design) over adding retries.

Also list the classes of failure you chose not to shortlist, with a rough count and the reason for each: for example
failures confined to a single PR, one-off infrastructure blips, fallout already explained by another finding, or
problems already being fixed in an open PR. This lets the requester check that nothing important was set aside.

### 6. Suggest improvements for the next sweep

In the same report, suggest a few ways to make the next sweep faster or more accurate: changes to this runbook, or to
the tooling and data behind it (e.g. a missing API field, CLI command or dashboard view). Write for a capable runner:
offer the insight or hint that would have changed how this sweep went, and leave the mechanics to them. When the
improvement is a small, concrete change with a big payoff (e.g. one extra pattern in a grep that would have caught a
whole class of failures), suggest the exact change.
