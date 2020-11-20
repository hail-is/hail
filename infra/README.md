This is a work in progress to use Terraform to manage our cloud
infrastructure.

Instructions:

- You will need a GCP project.  We assume `gcloud` is configured to
  point at your project.

- Create a service account for Terraform with Owner role, create a
  service account key and place it in
  `$HOME/.hail/terraform_sa_key.json`.

- Enable the the GCP services needed by Hail:

   ```
   gcloud services enable \
       compute.googleapis.com \
       cloudresourcemanager.googleapis.com \
       servicenetworking.googleapis.com \
       sqladmin.googleapis.com \
       container.googleapis.com \
       serviceusage.googleapis.com
   ```

- Install terraform.

- Create `$HOME/.hail/global.tfvars` that looks like:

   ```
   gcp_project = "<gcp-project>"
   gcp_region = "<gcp-region>"
   gcp_zone = "<gcp-zone>"
   domain = "<domain>"
   ```

- Run `terraform init`.

- Run `terraform apply -var-file="$HOME/.hail/global.tfvars"`.
  Terraform has created a GKE cluster named `vdc`.  We assume
  `kubectl` is configured to point at this cluster.

- Go to the Google Cloud console, VPC networks > internal > Private
  service connection > Private connections to services, and enable
  Export custom routes to both connections.

You can now install Hail:

- Run `$HAIL/docker/third-party/copy_images.sh`.  This copies some
  base images from Dockerhub (which now has rate limits) to GCR.

- Generate TLS certificates.  See ../dev-docs/tls-cookbook.md.

- Update $HAIL/config.mk with your infrastructure settings.  You can
  get settings from the default/global-config secret.

- Run `kubectl -n default apply -f $HAIL/ci/bootstrap.yaml`.

- Build the CI utils image.  Run `make push-ci-utils` in $HAIL/ci.

- Deploy the bootstrap gateway.  Run `make deploy` in
  $HAIL/bootstrap-gateway.

- Create Let's Encrypt certs. Run `make run` in $HAIL/letsencrypt.

- Deploy the gateway.  Run `make deploy` in $HAIL/gateway.

- Deploy the internal-gateway.  Run `make deploy` in $HAIL/internal-gateway.

- Go to the Google Cloud console, API & Services, Credentials.
  Configure the consent screen.  Add the scope:
  https://www.googleapis.com/auth/userinfo.email.  Create an OAuth
  client ID.  Authorize the redirect URI:
  https://auth.<domain>/oauth2callback.  Download the client secret
  as client_secret.json.  Create the auth-oauth2-client-secret secret
  with:

  ```
  kubectl -n default create secret generic auth-oauth2-client-secret --from-file=./client_secret.json
  ```

- Create a VM on the internal network, 100GB, Ubuntu 20.04 TLS, allow
  full access to all Cloud APIs.  10GB will run out of space.

- Install some dependencies:

  ```
  sudo apt update
  sudo apt install -y docker.io python3-pip mysql-client-core-8.0
  sudo usermod -a -G docker $USER
  gcloud -q auth configure-docker
  gcloud container clusters get-credentials --zone us-central1-a vdc
  git clone https://github.com/cseed/hail.git
  python3 -m pip install -r $HOME/hail/docker/requirements.txt
  ```

  You will have to log out/in for usermod to take effect.

- Bootstrap the cluster by running:

  ```
  HAIL_SHA=$(git rev-parse HEAD) \
  HAIL_CI_UTILS_IMAGE=gcr.io/<gcp-project>/ci-utils:latest \
  HAIL_CI_BUCKET_NAME=dummy \
  KUBERNETES_SERVER_URL='<k8s-server-url>' \
  HAIL_DEFAULT_NAMESPACE='default' \
  HAIL_DOMAIN=<domain> \
  HAIL_GCP_ZONE=<gcp-zone> \
  GCP_PROJECT=<gcp-project> \
  PYTHONPATH=$HOME/hail/ci:$HOME/hail/batch:$HOME/hail/hail/python \
  python3 $HAIL/ci/bootstrap.py
  ```

- You may want to add a suitable ssh forward rule to the internal network.
