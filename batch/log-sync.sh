#!/bin/bash
# Incrementally syncs a job log file to GCS.
# Usage: log-sync.sh <instruction-file>
# The instruction file is sourced each loop iteration so that state changes
# written by the worker (e.g. state=done) are picked up without any IPC.
# Responds to SIGUSR1 to skip the current sleep and proceed immediately,
# which the worker sends when the job finishes.
set -euo pipefail

INSTRUCTION_FILE=$1
SLEEP_PID=""

wakeup() {
    [[ -n "${SLEEP_PID:-}" ]] && kill "$SLEEP_PID" 2>/dev/null || true
}
trap wakeup SIGUSR1

validate() {
    [[ "${log:-}"                 =~ ^/              ]] || { echo "bad log: ${log:-unset}";                            exit 1; }
    [[ "${remote:-}"              =~ ^gs://          ]] || { echo "bad remote: ${remote:-unset}";                      exit 1; }
    [[ "${state:-}"               =~ ^(running|done)$ ]] || { echo "bad state: ${state:-unset}";                       exit 1; }
    [[ "${interval:-}"            =~ ^[0-9]+$        ]] || { echo "bad interval: ${interval:-unset}";                  exit 1; }
    [[ "${large_file_limit:-}"    =~ ^[0-9]+$        ]] || { echo "bad large_file_limit: ${large_file_limit:-unset}";  exit 1; }
    [[ "${large_file_interval:-}" =~ ^[0-9]+$        ]] || { echo "bad large_file_interval: ${large_file_interval:-unset}"; exit 1; }
}

# Write own PID so the worker can detect if we've died
sed -i "s|^pid=.*|pid=$$|" "$INSTRUCTION_FILE"

while true; do
    # shellcheck source=/dev/null
    source "$INSTRUCTION_FILE"
    validate

    if [[ -s "$log" ]]; then
        gcloud storage cp "$log" "$remote"
    fi

    [[ "$state" == "done" ]] && break

    file_size=$(stat --printf="%s" "$log" 2>/dev/null || echo 0)
    sleep_time=$(( file_size > large_file_limit ? large_file_interval : interval ))

    sleep "$sleep_time" &
    SLEEP_PID=$!
    wait "$SLEEP_PID" || true
    SLEEP_PID=""
done
