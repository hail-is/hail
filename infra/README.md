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

You can now install Hail:

- Run `$HAIL/docker/third-party/copy_images.sh`.  This copies some
  base images from Dockerhub (which now has rate limits) to GCR.

- Generate TLS certificates.  See ../dev-docs/tls-cookbook.md.

- Update $HAIL/config.mk with your infrastructure settings.  You can
  get settings from the default/global-config secret.

- Build the CI utils image.  Run `make push-ci-utils` in $HAIL/ci.

- Run `kubectl -n default apply -f bootstrap.yaml`.

