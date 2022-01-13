#!/bin/bash

set -ex

COMMAND=("$@")

restart() {
    kill $PID
    sleep 1
    if kill -0 $PID >/dev/null 2>&1
    then
        sleep 5
        kill -9 $PID || true
    fi

    "${COMMAND[@]}" &
    PID=$!
    restarted=yes
}
trap restart SIGINT

term() {
    kill -TERM $PID
}
trap term SIGTERM


"${COMMAND[@]}" &
PID=$!
restarted=yes

while [ $restarted == "yes" ]
do
    restarted=no
    wait $PID || true
done

trap - SIGTERM SIGINT
wait $PID
