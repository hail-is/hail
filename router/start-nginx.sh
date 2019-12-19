#!/bin/bash
set -ex

case "$HAIL_DEFAULT_NAMESPACE" in
    default)
        DOMAIN=$HAIL_DOMAIN
        NOTEBOOK_BASE_PATH=""
        WORKSHOP_BASE_PATH=""
        ;;
    *)
        DOMAIN=""
        NOTEBOOK_BASE_PATH="/$HAIL_DEFAULT_NAMESPACE/notebook"
        WORKSHOP_BASE_PATH="/$HAIL_DEFAULT_NAMESPACE/workshop"
        ;;
esac

sed -e "s,@domain@,$DOMAIN,g" \
    -e "s,@notebook_base_path@,$NOTEBOOK_BASE_PATH,g" \
    -e "s,@workshop_base_path@,$WORKSHOP_BASE_PATH,g" \
    -e "s,@namespace@,$HAIL_DEFAULT_NAMESPACE,g" \
    < /router.nginx.conf.in > /etc/nginx/conf.d/router.conf

nginx -g "daemon off;"
