If you're a third-party trying to deploy Hail, look at `../gcp`.

Hail team, this directory is an underestimate of our infrastructure. We are iteratively adding more
infrastructure. Infrastructure may be missing because importing into terraform would require a
destroy/create.

---

Changes from `../gcp`:

Create a bucket in which to store terraform state. Use the same region in which you plan to create
the k8s cluster.

```
PROJECT=YOUR GCP PROJECT HERE
LOCATION=us-central1
TERRAFORM_STATE_BUCKET=gs://terraform-state-$(cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 5)
gsutil mb -l us-central1 $TERRAFORM_STATE_BUCKET
gsutil -m uniformbucketlevelaccess set on $TERRAFORM_STATE_BUCKET
```

Create a key to encrypt terraform state.

```
gcloud kms keyrings create terraform-state-us-central1 \
       --location $LOCATION
gcloud kms keys create terraform-state-us-central1-key \
       --location $LOCATION \
	   --keyring terraform-state-us-central1 \
	   --purpose encryption
gcloud projects add-iam-policy-binding \
       <project-id> \
       --member='user:YOUR_EMAIL' \
	   --role='roles/owner'
gcloud kms keys list \
       --location $LOCATION \
	   --keyring terraform-state-us-central1
```
Store the Terraform key name in a variable for future use:
```
TERRAFORM_KEY_NAME=...
```
Finish the KMS setup:
```
gsutil kms authorize -p $PROJECT \
       -k $TERRAFORM_KEY_NAME
gcloud storage service-agent \
       --project=$PROJECT \
	   --authorize-cmek=$TERRAFORM_KEY_NAME
gcloud storage buckets update \
       $TERRAFORM_STATE_BUCKET \
	   --default-encryption-key=$TERRAFORM_KEY_NAME
```

I found that I had to explicitly grant read permissions to my account even though it was an Owner:

```
YOUR_USER_EMAIL=...
gcloud storage buckets add-iam-policy-binding \
       --member user:$YOUR_USER_EMAIL \
	   --role roles/storage.objectViewer \
       $TERRAFORM_STATE_BUCKET
gcloud storage buckets add-iam-policy-binding \
       --member user:$YOUR_USER_EMAIL \
	   --role roles/storage.objectCreator \
       $TERRAFORM_STATE_BUCKET
```

Create `backend.hcl`:

```
cat >infra/gcp-broad/$GITHUB_ORGANIZATION/backend.hcl <<EOF
bucket = $TERRAFORM_STATE_BUCKET
kms_encryption_key = $TERRAFORM_KEY_NAME
EOF
```

Initialize Terraform:

```
terraform init -backend-config=hail-is/backend.hcl -var-file=hail-is/global.tfvars
```

Then inspect the Terraform plan:

```
terraform plan -var-file=hail-is/global.tfvars -out=tfplan
```
