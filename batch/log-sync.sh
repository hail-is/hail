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

_base=$(basename "$INSTRUCTION_FILE" .conf)
_batch="${_base%%_*}"; _rest="${_base#*_}"; _job="${_rest%%_*}"; _attempt="${_rest#*_}"
PREFIX="[log-sync ${_batch}/${_job}/${_attempt}]"

wakeup() {
    [[ -n "${SLEEP_PID:-}" ]] && kill "$SLEEP_PID" 2>/dev/null || true
}
trap wakeup SIGUSR1

validate() {
    [[ "${log:-}"              =~ ^/              ]] || { echo "$PREFIX bad log: ${log:-unset}";                        exit 1; }
    [[ "${remote:-}"           =~ ^gs://          ]] || { echo "$PREFIX bad remote: ${remote:-unset}";                  exit 1; }
    [[ "${state:-}"            =~ ^(running|done)$ ]] || { echo "$PREFIX bad state: ${state:-unset}";                   exit 1; }
    [[ "${interval:-}"         =~ ^[0-9]+$        ]] || { echo "$PREFIX bad interval: ${interval:-unset}";              exit 1; }
    [[ "${small_limit:-}"      =~ ^[0-9]+$        ]] || { echo "$PREFIX bad small_limit: ${small_limit:-unset}";        exit 1; }
    [[ "${small_interval:-}"   =~ ^[0-9]+$        ]] || { echo "$PREFIX bad small_interval: ${small_interval:-unset}";  exit 1; }
    [[ "${medium_limit:-}"     =~ ^[0-9]+$        ]] || { echo "$PREFIX bad medium_limit: ${medium_limit:-unset}";      exit 1; }
    [[ "${medium_interval:-}"  =~ ^[0-9]+$        ]] || { echo "$PREFIX bad medium_interval: ${medium_interval:-unset}"; exit 1; }
    [[ "${large_limit:-}"      =~ ^[0-9]+$        ]] || { echo "$PREFIX bad large_limit: ${large_limit:-unset}";        exit 1; }
    [[ "${large_interval:-}"   =~ ^[0-9]+$        ]] || { echo "$PREFIX bad large_interval: ${large_interval:-unset}";  exit 1; }
    [[ "${xlarge_limit:-}"     =~ ^[0-9]+$        ]] || { echo "$PREFIX bad xlarge_limit: ${xlarge_limit:-unset}";      exit 1; }
    [[ "${xlarge_interval:-}"  =~ ^[0-9]+$        ]] || { echo "$PREFIX bad xlarge_interval: ${xlarge_interval:-unset}"; exit 1; }
}

# Write own PID so the worker can detect if we've died
sed -i "s|^pid=.*|pid=$$|" "$INSTRUCTION_FILE"

while true; do
    # shellcheck source=/dev/null
    source "$INSTRUCTION_FILE"
    validate

    file_size=$(stat --printf="%s" "$log" 2>/dev/null || echo 0)
    if [[ -s "$log" ]]; then
        echo "$PREFIX uploading ${file_size}B to $remote"
        gcloud storage cp "$log" "$remote" || echo "$PREFIX upload failed, will retry next cycle"
    else
        echo "$PREFIX skipping upload, log is empty"
    fi

    # Re-source after the upload in case state=done was written while gcloud was running
    # (the SIGUSR1 would have been a no-op during the upload, so we must catch it here).
    # If done, do one final upload to capture bytes written after the previous cp started.
    # shellcheck source=/dev/null
    source "$INSTRUCTION_FILE"
    if [[ "$state" == "done" ]]; then
        file_size=$(stat --printf="%s" "$log" 2>/dev/null || echo 0)
        if [[ -s "$log" ]]; then
            echo "$PREFIX final upload ${file_size}B to $remote"
            gcloud storage cp "$log" "$remote" || echo "$PREFIX final upload failed"
        fi
        break
    fi

    if (( file_size > xlarge_limit )); then
        tier=xlarge; sleep_time=$xlarge_interval
    elif (( file_size > large_limit )); then
        tier=large;  sleep_time=$large_interval
    elif (( file_size > medium_limit )); then
        tier=medium; sleep_time=$medium_interval
    elif (( file_size > small_limit )); then
        tier=small;  sleep_time=$small_interval
    else
        tier=tiny;   sleep_time=$interval
    fi
    echo "$PREFIX ${file_size}B -> tier=${tier}, sleeping ${sleep_time}s"

    # Re-source just before sleeping: SIGUSR1 may have fired between the post-upload re-source
    # and now (when SLEEP_PID was still empty and the signal was lost). Catching it here keeps
    # the window to microseconds rather than a full tier interval.
    # shellcheck source=/dev/null
    source "$INSTRUCTION_FILE"
    if [[ "$state" == "done" ]]; then
        file_size=$(stat --printf="%s" "$log" 2>/dev/null || echo 0)
        if [[ -s "$log" ]]; then
            echo "$PREFIX final upload ${file_size}B to $remote"
            gcloud storage cp "$log" "$remote" || echo "$PREFIX final upload failed"
        fi
        break
    fi

    sleep "$sleep_time" &
    SLEEP_PID=$!
    wait "$SLEEP_PID" || true
    SLEEP_PID=""
done
