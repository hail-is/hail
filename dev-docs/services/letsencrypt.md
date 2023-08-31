# Let's Encrypt

Hail uses Let's Encrypt certificates for the gateway pod.

## Refreshing Certificates

Update the `letsencrypt-config` secret:

```
make -C letsencrypt run
```

Restart your gateway pods without downtime. When the restart they load the new certificate:

```
kubectl rollout restart deployment gateway-deployment
```

## Revoking Certificates

First, gather a list of the crt.sh IDs for the certificates you want to revoke from
https://crt.sh/?q=YOUR_DOMAIN_HERE . You will notice there is always a precertificate and a leaf
certificate. Both have the same serial number so revoking one revokes the other. In the next step,
the command will fail if you specify an already revoked certificate, so you should only specify one
of each precertificate and leaf certificate pair.

To get list of IDs run the following:

```
$ CERT_IDS_TO_REVOKE=$(curl https://crt.sh/?q=YOUR_DOMAIN_HERE | pup 'td.outer a json{}' | jq '.[].text' | egrep -o '[0-9]{10}')
$ CERT_IDS_TO_REVOKE=$(echo \'$CERT_IDS_TO_REVOKE\')
$ echo $CERT_IDS_TO_REVOKE
'6503198927 6503193970 6128196502'
```

```
make -C letsencrypt revoke CERT_IDS_TO_REVOKE='6503198927 6503193970 6128196502'
```
