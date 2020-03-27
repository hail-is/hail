Playbook for building the Hail GCP project.

### Setup

Deployment assumes the following things have been set up by hand
beforehand:

 - In the GCP console, go to APIs & Services > Library and enable the
   following APIs:

   - Enable Identity and Access Management (IAM) API
   - Enable Cloud SQL Admin API
   - Google Cloud Deployment Manager V2 API

 - Reserve a static IP address `site` by running `make create-address`.

 - Update the domain's DNS to point to `site`'s external IP address.
   You can print the IP address by running `make echo-ip`.

 - Create a service account
   `deploy@<project-id>.iam.gserviceaccount.com` with the project
   owner role.

 - Activate the deploy service account in `gcloud` by running `make
   activate-deploy`.

### Deploy

 - Put secrets.yaml in `./secrets.yaml`.

 - Run, for example:

```
make PROJECT=hail-vdc IP=35.188.91.25 DOMAIN=hail.is build-out
```

   Warning: modifies gcloud, kubectl configuration setting

 - Add `vdc-sa@<project-id>.iam.gserviceaccount.com` service account
   to broad-ctsa/artifacts.broad-ctsa.appspot.com to Storage Object
   Viewer role.

### Finish

 - destroy the privileged deploy service account with `make destroy-deploy`

### FIXME

 - Doesn't deploy ci, which can't have multiple running instances.
 - Describe secrets.yaml.
