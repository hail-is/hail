#!/bin/bash
set -ex

sed -e "s,@domain@,$DOMAIN,g" < router.nginx.conf.in > /etc/nginx/conf.d/router.conf

nginx -g "daemon off;"
