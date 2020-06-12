#!/bin/bash
set -ex

/bin/bash /poll.sh &

rewrite_conf_location=/etc/nginx/conf.d/rewrite-links-for-namespace.conf

if [ $HAIL_DEFAULT_NAMESPACE == "default" ]
then
    prefix=""
    cat >$rewrite_conf_location <<EOF
# no rewriting necessary because we are in the default namespace
EOF
else
    prefix="/$HAIL_DEFAULT_NAMESPACE/site"
    cat >$rewrite_conf_location <<EOF
subs_filter href=(['"])/([^/]) href=\$1/$HAIL_DEFAULT_NAMESPACE/site/\$2 gr;
subs_filter src=(['"])/([^/]) src=\$1/$HAIL_DEFAULT_NAMESPACE/site/\$2 gr;
subs_filter (\ *import\ [^\ ]*\ from\ ['"])/([^/]) \$1/$HAIL_DEFAULT_NAMESPACE/site/\$2 gr;
EOF
fi

sed -e "s,@prefix@,$prefix,g" \
    < /hail.nginx.conf.in > /etc/nginx/conf.d/hail.conf

exec nginx -g "daemon off;"
