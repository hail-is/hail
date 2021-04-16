# TLS Cookbook

## Create a Self-Signed x509 Certificate in PEM Format

Produce an x509 certificate. The key is a 4096-bit RSA key. The cert is valid
from today until 365 days from today. The server must be accessed via the domain
name `localhost`. A client that verifies hostnames will reject this certificate
if the server is accessed via an IP (like `127.0.0.1`) or other names (like
`wm06b-953`). The certificate is not password protected due to `-nodes`.

```
openssl req -x509 \
        -newkey rsa:4096 \
        -keyout server-key.pem \
        -out server-cert.pem \
        -days 365 \
        -subj '/CN=localhost' \
        -nodes \
        -sha256
```

## Bundle a Key and Certificate into a PKCS12 File

Create a PKCS12 file. PKCS12 files are primarily useful for creating instances
of a Java `KeyStore`. It is not possible to [elide a
password](https://stackoverflow.com/questions/27497723/export-a-pkcs12-file-without-an-export-password).
Using the empty string as a password is not recommended because many tools do
not properly support it.

```
openssl pkcs12 -export \
        -out server-keystore.p12 \
        -inkey server-key.pem \
        -in server-cert.pem \
        -passout pass:foobar
```

## Inspect a Certificate

Print the start and end dates for a given certificate. Note that a certificate
whose start date is in the future is called "expired" by many tools.

```
openssl x509 -startdate -enddate -noout -in cert.pem
```

Print a complete textual representation of the certificate.

```
openssl x509 -text -noout -in cert.pem
```

## Determine the Cause of Certificate Expiration

Check the start and end dates.

```
openssl x509 -startdate -enddate -noout -in cert.pem
```

Is the start date in the future? Is the end date in the past? What is the time
on the machine on which the certificate failure occurred? Remember to always
compare times in UTC!

All Hail certificates are signed by a root certificate named `hail-root`. If
this certificate is expired, the signed certificate will also be called
"expired" even though it is not itself expired. Download the Hail root
certificate to a local file and inspect the start and end dates.

```
kubectl get secrets ssl-config-hail-root -o json \
    | jq -r '.data["hail-root-cert.pem"]' \
    | base64 --decode \
    > hail-root-cert.pem
```

This certificate should be bit-for-bit identical to the `SERVICE-incoming.pem`
trust file, but you should verify that. For example, download the
`batch` incoming trust:

```
kubectl get secrets ssl-config-batch -o json \
    | jq -r '.data["batch-incoming.pem"]' \
    | base64 --decode \
    > hail-root-cert.pem
```

## Regenerate All the Certificates

If something has gone wrong, a relatively straightforward way to get back to
working is to regenerate all the certificates. This procedure will cause
downtime: services using the old certs will not trust servers using the new
certs and vice-versa. Once all services have restarted, there should be no
downtime.

1. Regenerate the root certificate (from your laptop):

```
openssl req -new -x509 \
        -subj /CN=hail-root \
        -nodes \
        -newkey rsa:4096 \
        -keyout hail-root-key.pem \
        -out hail-root-cert.pem \
        -days 365 \
        -sha256
```

2. Update kubernetes with the new root certificate:

```
kubectl create secret generic \
        -n default ssl-config-hail-root \
        --from-file=hail-root-key.pem \
        --from-file=hail-root-cert.pem \
        --save-config \
        --dry-run=client \
        -o yaml \
    | kubectl apply -f -
```

3. Update all the service certificates:

```
make -C $HAIL/hail python/hailtop/hail_version

PYTHONPATH=$HAIL/hail/python \
        python3 $HAIL/tls/create_certs.py \
        default \
        $HAIL/tls/config.yaml \
        hail-root-key.pem \
        hail-root-cert.pem
```

4. Get a list of all the services for that need to be restarted (some of these are
   not actually services, but including them in the next step is OK).

```
SERVICES_TO_RESTART=$(python3 -c 'import os
import yaml
hail_dir = os.getenv("HAIL")
x = yaml.safe_load(open(f"{hail_dir}/tls/config.yaml"))["principals"]
print(",".join(x["name"] for x in x))')
```

5. Restart all the services by deleting the pods (but, critically, not the
   deployments):

```
kubectl delete pods -l "app in ($SERVICES_TO_RESTART)"
```
