#!/bin/bash

bash /poll-devel.sh
bash /poll-0.1.sh

# otherwise cron jobs won't run
# see: https://stackoverflow.com/questions/34962020/cron-and-crontab-files-not-executed-in-docker
touch /etc/crontab /etc/cron.*/*

service rsyslog start
service cron start

nginx -g "daemon off;"
