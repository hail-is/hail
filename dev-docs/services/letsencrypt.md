# Let's Encrypt

Hail uses Let's Encrypt certificates for the gateway pod.


## Context: Description of the Let's Encrypt Job

All Hail services under the hail.is domain share a single SSL certificate with multiple Subject Alternative Names (SANs). 
Certificates are obtained from Let's Encrypt. Every ninety (90) days, the following process is triggered when you execute make -C letsencrypt as described in the next section:

- A Docker container based on certbot/certbot is built and deployed as a Kubernetes pod [ref](https://github.com/hail-is/hail/blob/main/letsencrypt/Dockerfile)
- The process generates a single certificate request covering:
  - The main domain (hail.is)
  - An internal subdomain (internal.hail.is)
  - Service-specific subdomains (batch.hail.is, batch-driver.hail.is, ci.hail.is, etc.)
  - As defined in the subdomains configuration [ref](https://github.com/hail-is/hail/blob/main/letsencrypt/subdomains.txt) and processed by the Makefile [ref](https://github.com/hail-is/hail/blob/main/letsencrypt/Makefile)
- Certbot runs in standalone mode to prove domain control and acquire a Let's Encrypt SSL certificate [ref](https://github.com/hail-is/hail/blob/main/letsencrypt/letsencrypt.sh)
- The certificate (fullchain.pem), private key (privkey.pem), and NGINX SSL configuration files are stored in a shared Kubernetes Secret named letsencrypt-config [ref](https://github.com/hail-is/hail/blob/main/letsencrypt/letsencrypt.sh#L14-L21)
- All Hail services reference this same letsencrypt-config Secret to access the certificate and configuration
- The process is repeated every 90 days via the make script, automatically rotating the shared certificate for all services
- The certificates are provisioned with Let's Encrypt, which generates certificates chaining to ISRG Root X1 or ISRG Root X2 that are widely trusted. 
- The system enforces modern TLS security through NGINX configuration [ref](https://github.com/hail-is/hail/blob/main/letsencrypt/letsencrypt.sh#L31-L41) that requires:
  - TLS 1.2 and 1.3 protocols only
  - Strong cipher suites including:
    - ECDHE-ECDSA with AES-GCM
    - ECDHE-RSA with AES-GCM
    - ECDHE with ChaCha20-Poly1305
    - DHE-RSA with AES-GCM
- Reference Let's Encrypt Certificate Compatibility for further detail.


## Refreshing Certificates

Certificates must be updated once per cluster.

### Setting the `kubectl` Context

Why? 

1. We will be using the letsencrypt image to create a job instance in kubernetes.  
2. We will be restarting the gateway pods.

See [Setting the `kubectl` Context](setting_the_kubectl_context.md).

### Connect or fetch credentials to the container registry

See [Connecting Docker to Container Registry Credentials](connecting_docker_to_container_registry_creds.md).

### Rotating One Cluster's Certificates

After setting the `kubectl` context to the appropriate cluster, clean the
`letsencrypt-image` and `pushed-private-letsencrypt-image` files left by make.

```
make clean-image-targets
```

Update the `letsencrypt-config` secret:

```
NAMESPACE=default make -C letsencrypt run
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
