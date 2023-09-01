This is a work in progress to use Terraform to manage our cloud
infrastructure.

Instructions:

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

- Go to the Google Cloud console, API & Services, Credentials.
  Configure the consent screen.  Add the scope:
  https://www.googleapis.com/auth/userinfo.email.  Back in Credentials, create an OAuth
  client ID.  Authorize the redirect URIs:

   - https://auth.<domain>/oauth2callback
   - http://127.0.0.1/oauth2callback

  Download the client secret as `/tmp/auth_oauth2_client_secret.json`.

- Create `infra/gcp/$GITHUB_ORGANIZATION/global.tfvars` based on the template below, where `$GITHUB_ORGANIZATION` corresponds to the GitHub organization used for your Hail Batch deployment (e.g. [`hail-is`](https://github.com/hail-is/hail)). This avoids collisions between configuration files from different Hail deployments.


   ```
   # organization_domain is a string that is the domain of the organization
   # E.g. "hail.is"
   organization_domain = "<domain>"

   # The GitHub organization hosting your Hail Batch repository, e.g. "hail-is".
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
   batch_logs_bucket_storage_class = "MULTI_REGIONAL"

   # Similarly, bucket locations and storage classes are specified
   # for other services:
   hail_query_bucket_location = "<bucket-location>"
   hail_query_bucket_storage_class = "MULTI_REGIONAL"
   hail_test_gcs_bucket_location = "<bucket-location>"
   hail_test_gcs_bucket_storage_class = "MULTI_REGIONAL"

   gcp_region = "<gcp-region>"

   gcp_zone = "<gcp-zone>"

   gcp_location = "<gcp-region>"

   domain = "<domain>"

   # If set to true, pull the base ubuntu image from Artifact Registry.
   # Otherwise, assumes GCR.
   use_artifact_registry = false
   ```

- You can optionally create a `/tmp/ci_config.json` file to enable CI triggered by GitHub events:

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
  sops --encrypt --gcp-kms projects/<gcp-project-id>/locations/global/keyRings/sops/cryptoKeys/sops-key /tmp/auth_oauth2_client_secret.json > $HAIL/infra/gcp/$GITHUB_ORGANIZATION/auth_oauth2_client_secret.enc.json

  # Optional
  sops --encrypt --gcp-kms projects/<gcp-project-id>/locations/global/keyRings/sops/cryptoKeys/sops-key /tmp/ci_config.json > $HAIL/infra/gcp/$GITHUB_ORGANIZATION/ci_config.enc.json

  sops --encrypt --gcp-kms projects/<gcp-project-id>/locations/global/keyRings/sops/cryptoKeys/sops-key /tmp/terraform_sa_key.json > $HAIL/infra/gcp/$GITHUB_ORGANIZATION/terraform_sa_key.enc.json

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

- Install terraform.

- Run `terraform init`.

- Run `terraform apply -var-file=$GITHUB_ORGANIZATION/global.tfvars`.  At the
  time of writing, this takes ~15m.

- Terraform created a GKE cluster named `vdc`.  Configure `kubectl`
   to point at the vdc cluster:

   ```
   gcloud container clusters get-credentials --zone <gcp-zone> vdc
   ```

   Register `domain` with a DNS registry with the `ip` field in the
   Kubernetes global-config. This should point to the kubernetes
   external load balancer.


You can now install Hail:

- Create a VM on the internal network, standard-8, 100GB PD-SSD,
  Ubuntu 20.04 TLS, allow full access to all Cloud APIs, use the
  Terraform service account.  10GB will run out of space.  We assume
  the rest of the commands are run on the VM.  You will need to
  connect to this instance with ssh.  You may want to add a suiteable
  ssh forwarding rule to the default network.

- Clone the Hail Github repository:

  ```
  git clone https://github.com/hail-is/hail.git
  ```

- In the $HAIL/infra directory, run

  ```
  ./install_bootstrap_dependencies.sh
  ```

  At this point, log out and ssh back in (so that changes to group settings
  for Docker can be applied). The following steps should be completed from
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

  ```
  ./bootstrap.sh deploy_unmanaged
  ```

- Create the batch worker VM image. Run:

  ```
  NAMESPACE=default $HAIL/batch/gcp-create-worker-image.sh
  ```

- Download the global-config to be used by `bootstrap.py`.

  ```
  mkdir /global-config
  kubectl -n default get secret global-config -o json | jq -r '.data | map_values(@base64d) | to_entries|map("echo -n \(.value) > /global-config/\(.key)") | .[]' | bash
  ```

- Bootstrap the cluster.

  ```
  ./bootstrap.sh bootstrap $GITHUB_ORGANIZATION/hail:<BRANCH> deploy_batch
  ```

- Deploy the gateway: run `make -C $HAIL/gateway envoy-xds-config deploy`.

- Create the initial (developer) user.

  ```
  ./bootstrap.sh bootstrap $GITHUB_ORGANIZATION/hail:<BRANCH> create_initial_user <USERNAME> <EMAIL>
  ```

  Additional users can be added by the initial user by going to auth.<domain>/users.
