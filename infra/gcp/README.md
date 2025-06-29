This is a work in progress to use Terraform to manage our cloud
infrastructure.

# Instructions:

## Project Setup

- You will need a GCP project.  Configure `gcloud` to point at your project:

   ```
   gcloud config set project <gcp-project-id>
   gcloud config set compute/zone <gcp-zone>
   ```

- Enable the GCP services needed by Hail:

   ```
   gcloud services enable \
       container.googleapis.com \
       compute.googleapis.com \
       cloudkms.googleapis.com \
       cloudresourcemanager.googleapis.com \
       servicenetworking.googleapis.com \
       sqladmin.googleapis.com \
       serviceusage.googleapis.com \
       dns.googleapis.com \
       logging.googleapis.com \
       cloudprofiler.googleapis.com \
       monitoring.googleapis.com \
       iam.googleapis.com \
       artifactregistry.googleapis.com \
       cloudbilling.googleapis.com
   ```

- Delete the default network if it exists. Enabling the networking
  API creates it.

- Determine a domain name for the deployment. We will use it now and register it with a DNS provider later.

- Go to the Google Cloud console, API & Services.
  - Configure the consent screen.
    - You can probably leave most fields on the first page empty. Give it a sensible name and management email.
    - Add the scope: `../auth/userinfo.email`.
  - Back in Credentials, create an OAuth client ID of type `Web application`. Authorize the redirect URIs:
    - `https://auth.<domain>/oauth2callback`
    - `http://127.0.0.1/oauth2callback`
  - Download the client secret as `/tmp/auth_oauth2_client_secret.json`.
  - Create another OAuth client ID of type `Desktop app`
    - Download it as `/tmp/hailctl_client_secret.json`.

## Set up Terraform configuration

- Create some useful environment variables:
  - where `GITHUB_ORGANIZATION` corresponds to the GitHub organization used for your Hail Batch deployment (e.g. [`hail-is`](https://github.com/hail-is/hail)). This avoids collisions between configuration files from different Hail deployments.
  - If multiple hail instances are using subdomains hosted from the same github organization/repository, you can use something like `hail-is/sandbox` to differentiate between them.

```
export HAIL=<hail checkout directory>
export GITHUB_ORGANIZATION=<path to your working directory within $HAIL/infra/gcp>
export GCP_PROJECT=<gcp project name>
```

- Create `infra/gcp/$GITHUB_ORGANIZATION/global.tfvars` based on the template below.


   ```
   # organization_domain is a string that is the domain of the organization
   # E.g. "hail.is"
   organization_domain = "<domain>"

   # The GitHub organization hosting your Hail Batch repository, e.g. "hail-is".
   # Matching the location of your project files within the infra/gcp directory.
   # eg hail.is/sandbox
   github_organization = "<github-organization>"

   # batch_gcp_regions is a JSON array of string, the names of the gcp
   # regions to schedule over in Batch. E.g. "[\"us-central1\"]"
   batch_gcp_regions = "<batch-gcp-regions>"

   gcp_project = "<gcp-project-id>"

   # This is the bucket location that spans the regions you're going to
   # schedule across in Batch.  If you are running on one region, it can
   # just be that region. E.g. "US"
   batch_logs_bucket_location = "<bucket-location>"

   # The storage class for the batch logs bucket.  It should span the
   # batch regions and be compatible with the bucket location.
   batch_logs_bucket_storage_class = "STANDARD"

   # Similarly, bucket locations and storage classes are specified
   # for other services:
   hail_query_bucket_location = "<bucket-location>"
   hail_query_bucket_storage_class = "STANDARD"
   hail_test_gcs_bucket_location = "<bucket-location>"
   hail_test_gcs_bucket_storage_class = "STANDARD"

   gcp_region = "<gcp-region>"

   gcp_zone = "<gcp-zone>"

   gcp_location = "<gcp-region>"

   domain = "<domain>"

   # If set to true, pull the base ubuntu image from Artifact Registry.
   # Otherwise, assumes GCR.
   use_artifact_registry = true
   ```

- You can optionally create a `/tmp/ci_config.json` file to enable CI triggered by GitHub
  events. Note that `github_oauth_token` is not necessarily an OAuth2 access token. In fact, it
  should be a fine-grained personal access token. The currently public documentation on fine-grained
  access tokens is not very good. Check this [page in
  `github/docs`](https://github.com/github/docs/blob/main/content/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens.md)
  for information on how to create a personal access token that is privileged to access the
  `hail-is` organization. Note in particular that personal access tokens have a "resource owner"
  field which is fixed at creation time. The token can only read or write to repositories owned by
  the "resource owner".

  ```json
  {
      "bucket_location": "<gcp-zone>",
      "bucket_storage_class": "STANDARD",
      "deploy_steps": [
          "deploy_batch",
          "test_batch_0",
          "deploy_ci"
      ],
      "github_context": "ci-gcp",
      "github_oauth_token": "<TOKEN>",
      "github_user1_oauth_token": "<TOKEN>",
      "watched_branches": [
          [
              "hail-is/hail:main",
              true,
              false
          ]
      ]
  }
  ```

## Check Service Accounts into Repository

- Install [sops](https://github.com/mozilla/sops).

- Set up a key for sops to use:

  ```sh
  gcloud auth application-default login

  gcloud kms keyrings create sops --location global

  gcloud kms keys create sops-key --location global --keyring sops --purpose encryption

  gcloud kms keys list --location global --keyring sops
  ```

  You should see:

  ```sh
  NAME                                                                         PURPOSE          PRIMARY_STATE
  projects/<gcp-project-id>/locations/global/keyRings/sops/cryptoKeys/sops-key ENCRYPT_DECRYPT  ENABLED
  ```

  This key can be shared with other developers in your team, controlling access through IAM.  It needs to be created outside of Terraform to avoid a cyclic dependency: the Terraform configuration needs to decrypt `sops` files.

- Create a service account for Terraform with Owner role.  We use
  service account name `terraform`.  Create a JSON service account key
  and place it in `/tmp/terraform_sa_key.json`.

  ```
  gcloud iam service-accounts create terraform --display-name="Terraform Account"

  gcloud projects add-iam-policy-binding <project-id> --member='serviceAccount:terraform@<project-id>.iam.gserviceaccount.com' --role='roles/owner'

  gcloud iam service-accounts keys create /tmp/terraform_sa_key.json  --iam-account=terraform@<project-id>.iam.gserviceaccount.com
  ```


- Encrypt the above files and add them to the repository.

  ```sh
  sops --encrypt --gcp-kms projects/$GCP_PROJECT/locations/global/keyRings/sops/cryptoKeys/sops-key /tmp/auth_oauth2_client_secret.json > $HAIL/infra/gcp/$GITHUB_ORGANIZATION/auth_oauth2_client_secret.enc.json
  sops --encrypt --gcp-kms projects/$GCP_PROJECT/locations/global/keyRings/sops/cryptoKeys/sops-key /tmp/hailctl_client_secret.json > $HAIL/infra/gcp/$GITHUB_ORGANIZATION/hailctl_client_secret.enc.json

  # Optional
  sops --encrypt --gcp-kms projects/$GCP_PROJECT/locations/global/keyRings/sops/cryptoKeys/sops-key /tmp/ci_config.json > $HAIL/infra/gcp/$GITHUB_ORGANIZATION/ci_config.enc.json

  sops --encrypt --gcp-kms projects/$GCP_PROJECT/locations/global/keyRings/sops/cryptoKeys/sops-key /tmp/terraform_sa_key.json > $HAIL/infra/gcp/$GITHUB_ORGANIZATION/terraform_sa_key.enc.json

  git add $HAIL/infra/gcp/$GITHUB_ORGANIZATION/*

  # git commit and push as desired.
  ```

- If you want Zulip integration for alerts from CI and Grafana, create a zuliprc file:

  ```sh
  cat /tmp/zuliprc <<EOF
  [api]
  key=SECRET_KEY_HERE
  email=YOUR_BOT_EMAIL_HERE
  site=YOUR_SITE_HERE
  EOF
  ```

- Encrypt the zuliprc with SOPS:

  ```sh
  sops --encrypt --gcp-kms projects/<gcp-project-id>/locations/global/keyRings/sops/cryptoKeys/sops-key /tmp/zuliprc \
       >$HAIL/infra/gcp/$GITHUB_ORGANIZATION/zuliprc.enc
  ```

## Terraforming the Project

- Preparation
  - Install terraform.
  - Switch directory to `$HAIL/infra/gcp`.
  - Clear any existing terraform state:

```
rm -rf .terraform terraform.lock.hcl terraform.tfstate terraform.tfstate.backup
```

- Run `terraform init`.

- Run `terraform apply -var-file=$GITHUB_ORGANIZATION/global.tfvars`.  At the
  time of writing, this takes ~15m.

## Register the domain

   Register the predetermined `domain` with a DNS registry.

   The IP address to use will be available in GCP cloud console under `Network Services -> Load balancing`.
   Click through to the external load balancer and find its IP address.

   Add two records with the same IP address:
    - `<domain>`
    - `*.<domain>`

## Deploy Hail to Kubernetes

We can now deploy Hail to the kubernetes cluster that terraform created.

### Set up kubectl

- Terraform created a GKE cluster named `vdc`.  Configure `kubectl`
   to point at the vdc cluster:

   ```
   gcloud container clusters get-credentials --zone <gcp-zone> vdc
   ```

### Deploy with a cloud VM

#### Creating a suitable cloud VM

Using the Google cloud console, create a VM in Compute Engine:

  - Use an easy to recognize name (eg `cjl-temp-hail-deployer`)
  - Region / zone: use the same as the GKE cluster
  - n1-standard-8
  - OS and storage
    - Ubuntu 24.04 LTS
    - 100GB PD-SSD
  - On the internal network
  - Security
    - Allow full access to all Cloud APIs
    - Run as the Terraform service account

Note: 10GB will run out of space.

An example command to create a VM via the gcloud CLI is given below, but be careful to make sure the settings
are correct for your deployment.

```
gcloud compute instances create bootstrap-vm \
    --project=<PROJECT> \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --provisioning-model=STANDARD \
    --service-account=<TERRAFORM-SERVICE-ACCOUNT> \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --create-disk=auto-delete=yes,boot=yes,device-name=instance-20240716-184710,image=projects/ubuntu-os-cloud/global/images/ubuntu-2404-noble-amd64-v20250606,mode=rw,size=200,type=projects/hail-vdc-dgoldste/zones/us-central1-a/diskTypes/pd-balanced
```

#### Cloud VM commands

We assume the rest of the commands are run on the VM. You will need to connect to this instance with ssh.
You can copy a `gcloud compute ssh` command to do this directly from the VM details page in the cloud console,
or construct it manually via the gcloud CLI. It will look something like:

```
gcloud compute ssh --zone "us-central1-a" "<VM-NAME>" --project "<PROJECT>"
```

##### Prerequisites

- If necessary, install `gke-gcloud-auth-plugin`:

  ```
  # Check for necessity:
  gke-gcloud-auth-plugin --version
  ```

  - Follow the instructions [here](https://cloud.google.com/sdk/docs/install#deb) to install the Google Cloud SDK
    package source.
  - Follow the `apt-get` instructions [here](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_plugin)
    to install the `gke-gcloud-auth-plugin`.

- Install the `docker-buildx-plugin`:
  - Follow the instructions [here](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) to set
    up the repository (ie step 1).
  - Install the `docker-buildx-plugin`:
    ```
    sudo apt-get install docker-buildx-plugin
    ```


- Clone the Hail Github repository:

  ```
  git clone https://github.com/hail-is/hail.git
  ```

- If you are working from a branch, check out that branch now too.

- In the $HAIL/infra directory, run

  ```
  ./install_bootstrap_dependencies.sh
  ```

- At this point, log out and ssh back in (so that changes to group settings
  for Docker can be applied).

---

- The following steps should be completed from
  the $HAIL/infra/gcp directory, unless otherwise stated.

- Run the following to authenticate docker and kubectl with the new artifact
  registry and kubernetes cluster, respectively. The `GKE_ZONE` is the zone of
  the GKE cluster and the `GAR_REGION` is the region of the artifact registry.

  ```
  ./bootstrap.sh configure_gcloud <GKE_ZONE> <GAR_REGION>
  ```

- Edit `$HAIL/letsencrypt/subdomains.txt` to include just the services you plan
  to use in this deployment, e.g. `auth`, `batch` and `batch-driver`.

- Deploy unmanaged resources by running

> [!WARNING]
> If using Google Artifact Registry, the kubernetes system user (called something like `<ID>-compute@developer.gserviceaccount.com`)
> will need to be granted read permission on the registry:
> `gcloud artifacts repositories add-iam-policy-binding hail --location=us-central1 --member=serviceAccount:<ID>-compute@developer.gserviceaccount.com --role="roles/artifactregistry.reader"`

  ```
  ./bootstrap.sh deploy_unmanaged
  ```

- Create the batch worker VM image. Run:

  ```
  NAMESPACE=default $HAIL/batch/gcp-create-worker-image.sh
  ```

- Download the global-config to be used by `bootstrap.py`.

  ```
  sudo mkdir /global-config
  source $HAIL/devbin/functions.sh
  download-secret global-config
  sudo cp contents/* /global-config/
  cd -
  sudo chmod +r /global-config/*
  ```

- Bootstrap the cluster.

  ```
  ./bootstrap.sh bootstrap $GITHUB_ORGANIZATION/hail:<BRANCH> deploy_batch
  ```

- Deploy the gateway: run `make -C $HAIL/gateway envoy-xds-config deploy NAMESPACE=default`.

- Create the initial (developer) user.

  ```
  ./bootstrap.sh bootstrap $GITHUB_ORGANIZATION/hail:<BRANCH> create_initial_user <USERNAME> <EMAIL>
  ```

  Additional users can be added by the initial user by going to auth.<domain>/users.

> [!NOTE]
> Troubleshooting this step:
> When I ran this step (perhaps because I had to log in and out of my cloud VM a couple of times), the
> hailctl command was not properly authenticating and the create_initial_user step failed. To make it work, I had to:
>   - Edit the $HAIL/build.yaml file
>     - Under the 'create_initial_user' step, in the script section:
>       - Add an additional environment line: `export HAIL_IDENTITY_PROVIDER_JSON='{"idp": "Google"}'`
>       - Add some commands to create an additional domain-setting config file, right under the environment exports:
>         - `mkdir ~/.hail`
>         - `echo '{"location":"external","default_namespace":"default","domain":"<DOMAIN>"}'>~/.hail/deploy-config.json`

## Remove the cloud VM

- Once the deployment is complete, you can remove the cloud VM in the Google cloud console.
