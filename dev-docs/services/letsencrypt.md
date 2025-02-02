# Let's Encrypt

Hail uses Let's Encrypt certificates for the gateway pod.

## Refreshing Certificates

Certificates must be updated once per cluster.

### Setting the `kubectl` Context

The hail project itself maintains two kubernetes clusters, one in GCP and one in
Azure.

If you have authenticated with `kubectl` to the appropriate cluster and `docker`
for the corresponding container registry, then you should only need to set the
current `kubectl` context by running:

```
kubectl config use-context <CONTEXT NAME>
```

The contexts can be listed with:
```
kubectl config get-contexts
```

If you are not authenticated, then you can run the following functions from
[`devbin/functions.sh`](/devbin/functions.sh):

```
# for GCP
gcpsetcluster <PROJECT>
# for Azure
azsetcluster <RESOURCE_GROUP>
```

### Connect or fetch credentials to the container registry

In GCP, something like:
```
gcloud auth login
gcloud auth configure-docker us-central1-docker.pkg.dev # Depending on where GAR is hosted
```


In Azure, something like:
```
az login
az acr login --name haildev # the container registry name is findable in the azure console
```

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
