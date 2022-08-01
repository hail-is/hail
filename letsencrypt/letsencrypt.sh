#!/bin/bash
set -ex

if [ -z "${DRY_RUN+x}" ]
then
    # if DRY_RUN is not set
    KUBECTL_APPLY='kubectl apply -f -'
else
    CERTBOT_FLAGS=--test-cert
    KUBECTL_APPLY=cat
fi

certbot certonly --standalone $CERTBOT_FLAGS --cert-name $DOMAIN -n --agree-tos -m cseed@broadinstitute.org -d $DOMAINS

# https://github.com/certbot/certbot/blob/master/certbot-nginx/certbot_nginx/_internal/tls_configs/options-ssl-nginx.conf
cat >/options-ssl-nginx.conf <<EOF
ssl_session_cache shared:le_nginx_SSL:10m;
ssl_session_timeout 1440m;
ssl_session_tickets off;
ssl_protocols TLSv1.2 TLSv1.3;
ssl_prefer_server_ciphers on;
ssl_ciphers "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384";
EOF

set +x # do not leak the secrets into the stdout logs

cat | $KUBECTL_APPLY <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: letsencrypt-config
  namespace: default
type: Opaque
data:
  fullchain.pem: $(cat /etc/letsencrypt/live/$DOMAIN/fullchain.pem | base64 | tr -d \\n)
  privkey.pem: $(cat /etc/letsencrypt/live/$DOMAIN/privkey.pem | base64 | tr -d \\n)
  options-ssl-nginx.conf: $(cat /options-ssl-nginx.conf | base64 | tr -d \\n)
  ssl-dhparams.pem: $(cat /opt/certbot/src/certbot/certbot/ssl-dhparams.pem | base64 | tr -d \\n)
EOF

set -x

echo finished updating cert.
