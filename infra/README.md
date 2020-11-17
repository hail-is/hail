This is a work in progress to use Terraform to manage our cloud
infrastructure.

Instructions:

- You will need a GCP project.  Create a service account for Terraform
  with Editor role, and create a service account key and place it in
  `$HOME/.hail/terraform_sa_key.json`.

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

You can now install Hail.  Everything beyond this point assumes gcloud
and kubectl point to your GCP project and the cluster created by
Terraform.

- Run `$HAIL/docker/third-party/copy_images.sh`.  This copies some
  base images from Dockerhub (which now has rate limits) to GCR.

