#!/bin/bash
set -ex

for cert_id in $CERT_IDS_TO_REVOKE
do
    echo revoking $cert_id

    wget -O $cert_id.crt https://crt.sh/?d=$cert_id

    subject_name=$(openssl x509 -subject -noout -in $cert_id.crt | sed 's/subject=CN = //')
    subject_alternate_names=$(openssl x509 -ext subjectAltName -noout -in $cert_id.crt | tail -n +2 | sed 's/ *DNS://g')

    # https://letsencrypt.org/docs/revoking/
    # Verify we control this domain. Include a non-existing subdomain to ensure no certificate is created.
    ! certbot certonly \
            --standalone $CERTBOT_FLAGS \
            --cert-name $subject_name \
            -d bogus-for-revoke.$subject_name \
            -n \
            --agree-tos \
            -m cseed@broadinstitute.org \
            -d $subject_name \
            -d $subject_alternate_names

    certbot revoke --non-interactive --no-delete-after-revoke --cert-path $cert_id.crt

    echo finished revoking $cert_id
done

echo finished revoking certs.
