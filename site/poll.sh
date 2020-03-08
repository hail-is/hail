#!/bin/bash
set -x

while true; do
    /bin/bash poll-0.2.sh
    /bin/bash poll-0.1.sh
    sleep 180
done
