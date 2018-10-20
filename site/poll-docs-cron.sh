#!/bin/bash
set -ex

# FIXME rotate logs
bash /poll-0.1.sh >>/var/log/cron.log 2>&1
bash /poll-0.2.sh >>/var/log/cron.log 2>&1
