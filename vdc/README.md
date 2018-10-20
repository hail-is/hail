Playbook for building the Hail GCP project.

### Setup

Deployment assumes the following things have been set up by hand
beforehand:

 - In the GCP console, go to APIs & Services > Library and enable the
   following APIs:

   - Enable Identity and Access Management (IAM) API
   - Enable Cloud SQL Admin API

 - In the GCP console, go to VPC network > External IP addresses >
   RESERVE STATIC IP ADDRESS and reserve a static IP address called
   `site`. Also see [address.yaml](address.yaml).

 - Update the domain's DNS to point to `site`'s external IP address.

 - In the GCP console, go to APIs & Services > Credentials > OAuth
   consent screen, and configure the consent screen.

 - In the GCP console, go to APIs & Services > Credentials >
   Credentials and create an OAuth client ID.

   - Choose application type "Web application".
   - Authorized redirect URIs will be
     `https://upload.hail.is/oauth2callback` adjusted for domain.

### Deploy

 - Put the OAuth client ID JSON in `./client_secret.json`.  It can be
   downloaded from APIs & Services > Credentials > Credentials.

 - Put secrets.yaml in `./secrets.yaml`.

 - Run, for example:

```
make build-out PROJECT=hail-vdc IP=35.188.91.25 DOMAIN=staging.hail.is
```

   Warning: modifies gcloud project configuration setting

### FIXME

 - Doesn't deploy ci, which can't have multiple running instances.
 - Batch likely doesn't work, needs k8s service account to create pods.
 - Describe secrets.yaml.
 - List of APIs to enable is not exhaustive.  Run through in a fresh
   project.
