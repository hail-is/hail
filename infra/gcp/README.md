This is a work in progress to use Terraform to manage our cloud
infrastructure.

Instructions:

- You will need a GCP project.  Configure `gcloud` to point at your project:

   ```
   gcloud config set project <gcp-project-id>
   gcloud config set compute/zone <gcp-zone>
   ```

- Create a service account for Terraform with Owner role.  We use
  service account name `terraform`.  Create a JSON service account key
  and place it in `$HOME/.hail/terraform_sa_key.json`.

  ```
  gcloud iam service-accounts create terraform --display-name="Terraform Account"
  gcloud projects add-iam-policy-binding <project-id> --member='serviceAccount:terraform@<project-id>.iam.gserviceaccount.com' --role='roles/owner'
  gcloud iam service-accounts keys create $HOME/.hail/terraform_sa_key.json  --iam-account=terraform@<project-id>.iam.gserviceaccount.com
  ```

- Enable the GCP services needed by Hail:

   ```
   gcloud services enable \
       container.googleapis.com \
       compute.googleapis.com \
       cloudresourcemanager.googleapis.com \
       servicenetworking.googleapis.com \
       sqladmin.googleapis.com \
       serviceusage.googleapis.com \
       dns.googleapis.com \
       logging.googleapis.com \
       cloudprofiler.googleapis.com \
       monitoring.googleapis.com \
       iam.googleapis.com \
       artifactregistry.googleapis.com
   ```

- Delete the default network if it exists. Enabling the networking
  API creates it.

- Go to the Google Cloud console, API & Services, Credentials.
  Configure the consent screen.  Add the scope:
  https://www.googleapis.com/auth/userinfo.email.  Back in Credentials, create an OAuth
  client ID.  Authorize the redirect URIs:

   - https://auth.<domain>/oauth2callback
   - http://127.0.0.1/oauth2callback

  Download the client secret as `~/.hail/auth_oauth2_client_secret.json`.

- Install terraform.

- Create `$HOME/.hail/global.tfvars` that looks like:

   ```
   # organization_domain is a string that is the domain of the organization
   # E.g. "broadinstitute.org"
   organization_domain = "<organization-domain>"

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
   
   # Similarly, bucket locations and storage classess are specified 
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

- Run `terraform init`.

- Run `terraform apply -var-file="$HOME/.hail/global.tfvars"`.  At the
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

- Run the following to authenticate docker and kubectl with the new
  container registry and kubernetes cluster, respectively.

  ```
  ./bootstrap.sh configure_gcloud <ZONE>
  ```

- Deploy unmanaged resources by running

  ```
  ./bootstrap.sh deploy_unmanaged
  ```

- Create the batch worker VM image. Run:

  ```
  make -C $HAIL/batch gcp-create-build-worker-image-instance
  ```

  Wait for the `build-batch-worker-image` instance to be stopped. Then run:

  ```
  make -C $HAIL/batch gcp-create-worker-image
  ```

- Download the global-config to be used by `bootstrap.py`.

  ```
  mkdir /global-config
  kubectl -n default get secret global-config -o json | jq -r '.data | map_values(@base64d) | to_entries|map("echo -n \(.value) > /global-config/\(.key)") | .[]' | bash
  ```

- Bootstrap the cluster.

  ```
  ./bootstrap.sh bootstrap <REPO>/hail:<BRANCH> deploy_batch
  ```

- Deploy the gateway. First, edit `$HAIL/letsencrypt/subdomains.txt` to include
  just the deployed services. Then run `make -C $HAIL/gateway deploy`.

- Create the initial (developer) user.

  ```
  ./bootstrap.sh bootstrap <REPO>/hail:<BRANCH> create_initial_user <USERNAME> <EMAIL>
  ```

  Additional users can be added by the initial user by going to auth.<domain>/users.
