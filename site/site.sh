#!/bin/bash
set -ex

/bin/bash /poll.sh &

nginx -g "daemon off;"
