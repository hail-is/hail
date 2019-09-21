#!/bin/bash
set -ex

case "$HAIL_DEFAULT_NAMESPACE" in
    default)
        DOMAIN=$HAIL_DOMAIN
        NOTEBOOK2_BASE_PATH=""
        ;;
    *)
        DOMAIN=""
        NOTEBOOK2_BASE_PATH="/$HAIL_DEFAULT_NAMESPACE/notebook2"
        ;;
esac

sed -e "s,@domain@,$DOMAIN,g" \
    -e "s,@notebook2_base_path@,$NOTEBOOK2_BASE_PATH,g" \
    < /router.nginx.conf.in > /etc/nginx/conf.d/router.conf

nginx -g "daemon off;"
