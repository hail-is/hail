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

- Check to make sure that there does not exist a default network with
  `gcloud compute networks list`. If the default network does exist, delete it.

- Go to the Google Cloud console, API & Services, Credentials.
  Configure the consent screen.  Add the scope:
  https://www.googleapis.com/auth/userinfo.email.  Back in Credentials, create an OAuth
  client ID.  Authorize the redirect URIs:

   - https://auth.<domain>/oauth2callback
   - http://127.0.0.1/oauth2callback

  Download the client secret to $HOME/.hail/client_secret.json.

- Install terraform.

- Create `$HOME/.hail/global.tfvars` that looks like:

   ```
   gsuite_organization = "<gsuite-organization>"

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

   hail_test_gcs_bucket_location = "<bucket-location>"
   hail_test_gcs_bucket_storage_class = "MULTI_REGIONAL"

   gcp_region = "<gcp-region>"

   gcp_zone = "<gcp-zone>"

   gcp_location = "<gcp-region>"

   domain = "<domain>"

   # If set to true, pull the base ubuntu image from Artifact Registry.
   # Otherwise, assumes GCR.
   use_artifact_registry = false

   # If the third element in the inner array is set to false,
   # CI will not merge PRs in the source repo, but will still test
   # and post statuses.
   # If the fourth element is set to false, CI will not notify of failures
   # on Zulip
   ci_watched_branches = "[[\"<repo_org>/hail:main\",true,true,true]]"

   ci_github_oauth_token = "<ci_github_oauth_token>"
   ```

- Run the terraform:

  ```
  ./bootstrap.sh run_gcp_terraform
  ./bootstrap.sh run_k8s_terraform
  ```

 - Terraform created a GKE cluster named `vdc`.  Configure `kubectl`
   to point at the vdc cluster:

   ```
   gcloud container clusters get-credentials --zone <gcp-zone> vdc
   ```

You can now install Hail:

- Create a VM by running `./create_bootstrap_vm.sh`

- You will need to connect to this instance with ssh.
  You may want to add a suiteable ssh forwarding rule to the default network.

- Clone the Hail repository:

  ```
  git clone https://github.com/<repo_org>/hail.git
  ```

- To install some dependencies on the VM, In the $HAIL/infra directory, run

  ```
  ./install_bootstrap_dependencies.sh
  ```

  You will have to log out/in for the usermod to take effect.

  Then configure `gcloud` by running (in the $HAIL/infra/gcp directory):

  ```
  ./bootstrap.sh setup_gcloud <ZONE>
  ```

- Deploy unmanaged resources by running

  ```
  ./bootstrap.sh deploy_unmanaged
  ```

- Create the batch worker VM image. Run:

  ```
  make -C $HAIL/batch gcp-create-build-worker-image-instance PROJECT=<PROJECT> ZONE=<ZONE> DOCKER_ROOT_IMAGE=<DOCKER_ROOT_IMAGE>
  ```

  Wait for the `build-batch-worker-image` instance to be stopped. Then run:

  ```
  make -C $HAIL/batch gcp-create-worker-image PROJECT=<PROJECT> ZONE=<ZONE>
  ```

- Create the worker Docker image. Run:

  ```
  make -C $HAIL/batch build-worker
  ```

- Bootstrap the cluster:

  ```
  source $HOME/hail/devbin/functions.sh
  download-secret zulip-config && cp -r contents /zulip-config && cd -
  download-secret global-config && cp -r contents /global-config && cd -
  ./bootstrap.sh bootstrap <REPO>/hail:<BRANCH> deploy_batch
  ```

- Deploy the gateway. Be sure that `$HAIL/letsencrypt/subdomains.txt` includes
  only those services that you have deployed. Then run `make -C $HAIL/gateway deploy`.

- Create the initial (developer) user. Run:

  ```
  ./bootstrap.sh bootstrap <REPO>/hail:<BRANCH> create_initial_user <USERNAME> <EMAIL>
  ```

  Additional users can be added by the initial user by going to auth.<domain>/users.
