#!/bin/bash
# Incrementally syncs a job log file to GCS.
# Usage: log-sync.sh <instruction-file>
# The instruction file is re-read each loop iteration so that state changes
# written by the worker (e.g. state=done) are picked up without any IPC.
# Responds to SIGUSR1 to skip the current sleep and proceed immediately,
# which the worker sends when the job finishes.
set -euo pipefail

INSTRUCTION_FILE=$1
SLEEP_PID=""
last_uploaded_size=-1
wakeup_pending=0

_base=$(basename "$INSTRUCTION_FILE" .conf)
_batch="${_base%%_*}"; _rest="${_base#*_}"; _job="${_rest%%_*}"; _attempt="${_rest#*_}"
PREFIX="[log-sync ${_batch}/${_job}/${_attempt}]"

load_instruction_file() {
    while IFS='=' read -r key value; do
        declare -g "$key=$value"
    done < "$INSTRUCTION_FILE"
}

wakeup() {
    wakeup_pending=1
    [[ -n "${SLEEP_PID:-}" ]] && kill "$SLEEP_PID" 2>/dev/null || true
}
trap wakeup SIGUSR1
sed -i "s|^trap_installed=.*|trap_installed=1|" "$INSTRUCTION_FILE"

validate() {
    [[ "${log:-}"                =~ ^/              ]] || { echo "$PREFIX bad log: ${log:-unset}";                              exit 1; }
    [[ "${remote:-}"             =~ ^gs://          ]] || { echo "$PREFIX bad remote: ${remote:-unset}";                        exit 1; }
    [[ "${state:-}"              =~ ^(running|done)$ ]] || { echo "$PREFIX bad state: ${state:-unset}";                         exit 1; }
    [[ "${target_bytes_per_s:-}" =~ ^[0-9]+$        ]] || { echo "$PREFIX bad target_bytes_per_s: ${target_bytes_per_s:-unset}"; exit 1; }
    [[ "${min_interval:-}"       =~ ^[0-9]+$        ]] || { echo "$PREFIX bad min_interval: ${min_interval:-unset}";            exit 1; }
    [[ "${max_interval:-}"       =~ ^[0-9]+$        ]] || { echo "$PREFIX bad max_interval: ${max_interval:-unset}";            exit 1; }
}

while true; do
    load_instruction_file
    validate

    file_size=$(stat --printf="%s" "$log" 2>/dev/null || echo 0)
    if [[ -e "$log" ]] && (( file_size != last_uploaded_size )); then
        echo "$PREFIX uploading ${file_size}B to $remote"
        # Transient gcloud failure: skip this cycle and retry next iteration rather than aborting.
        if gcloud storage cp "$log" "$remote"; then
            last_uploaded_size=$file_size
        else
            echo "$PREFIX upload failed, will retry next cycle"
        fi
    else
        echo "$PREFIX skipping upload, no new bytes"
    fi

    # Re-read after the upload: if SIGUSR1 fired while gcloud was running, the bash trap queued
    # it and wakeup() was a no-op (SLEEP_PID was empty). Re-reading here catches state=done set
    # during the upload so we don't then sleep a full interval before noticing.
    # If done, do a final upload (upload N+1) to capture bytes written after upload N started.
    load_instruction_file
    if [[ "$state" == "done" ]]; then
        file_size=$(stat --printf="%s" "$log" 2>/dev/null || echo 0)
        if [[ -e "$log" ]] && (( file_size != last_uploaded_size )); then
            echo "$PREFIX final upload ${file_size}B to $remote"
            gcloud storage cp "$log" "$remote" || echo "$PREFIX final upload failed"
        fi
        break
    fi

    sleep_time=$(( target_bytes_per_s > 0 ? file_size / target_bytes_per_s : min_interval ))
    (( sleep_time < min_interval )) && sleep_time=$min_interval
    (( sleep_time > max_interval )) && sleep_time=$max_interval
    echo "$PREFIX ${file_size}B, sleeping ${sleep_time}s"

    # Re-read just before sleeping: SIGUSR1 may have fired after the post-upload re-read
    # (which saw state=running) but before SLEEP_PID was set, so wakeup() would have been a no-op.
    # wakeup_pending catches that case; consuming it here skips the sleep so the next iteration
    # uploads immediately. With the size-change guard this is safe: if nothing new was written
    # the upload is skipped and we sleep normally on the following iteration.
    load_instruction_file
    if [[ "$state" == "done" ]]; then
        file_size=$(stat --printf="%s" "$log" 2>/dev/null || echo 0)
        if [[ -e "$log" ]] && (( file_size != last_uploaded_size )); then
            echo "$PREFIX final upload ${file_size}B to $remote"
            gcloud storage cp "$log" "$remote" || echo "$PREFIX final upload failed"
        fi
        break
    fi
    if (( wakeup_pending )); then
        wakeup_pending=0
        continue
    fi

    sleep "$sleep_time" &
    SLEEP_PID=$!
    wait "$SLEEP_PID" || true
    SLEEP_PID=""
done
