#!/bin/bash
set -ex

# otherwise cron jobs won't run
# see: https://stackoverflow.com/questions/34962020/cron-and-crontab-files-not-executed-in-docker
touch /etc/crontab /etc/cron.*/*

service rsyslog start
service cron start

# link nginx logs to docker log collectors
ln -sf /dev/stdout /var/log/nginx/access.log
ln -sf /dev/stderr /var/log/nginx/error.log

nginx -g "daemon off;"
