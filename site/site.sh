#!/bin/bash
set -ex

/bin/bash /poll.sh &

exec nginx -g "daemon off;"
