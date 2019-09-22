#!/bin/bash
set -ex

case "$HAIL_DEFAULT_NAMESPACE" in
    default)
        DOMAIN=$HAIL_DOMAIN
        NOTEBOOK_BASE_PATH=""
        ;;
    *)
        DOMAIN=""
        NOTEBOOK_BASE_PATH="/$HAIL_DEFAULT_NAMESPACE/notebook"
        ;;
esac

sed -e "s,@domain@,$DOMAIN,g" \
    -e "s,@notebook_base_path@,$NOTEBOOK_BASE_PATH,g" \
    < /router.nginx.conf.in > /etc/nginx/conf.d/router.conf

nginx -g "daemon off;"
