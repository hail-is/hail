#!/bin/bash

bash /poll-devel.sh
bash /poll-0.1.sh

service rsyslog start
service cron start

nginx -g "daemon off;"
