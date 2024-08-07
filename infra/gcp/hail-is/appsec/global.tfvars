# organization_domain is a string that is the domain of the organization
# E.g. "hail.is"
organization_domain = "appsec.hail.is"

# The GitHub organization hosting your Hail Batch repository, e.g. "hail-is".
github_organization = "hail-is/appsec"

# batch_gcp_regions is a JSON array of string, the names of the gcp
# regions to schedule over in Batch. E.g. "[\"us-central1\"]"
batch_gcp_regions = "[\"us-central1\"]"

gcp_project = "dsp-appsec-hail"

# This is the bucket location that spans the regions you're going to
# schedule across in Batch.  If you are running on one region, it can
# just be that region. E.g. "US"
batch_logs_bucket_location = "us-central1"

# The storage class for the batch logs bucket.  It should span the
# batch regions and be compatible with the bucket location.
batch_logs_bucket_storage_class = "STANDARD"

# Similarly, bucket locations and storage classes are specified
# for other services:
hail_query_bucket_location = "us-central1"
hail_query_bucket_storage_class = "STANDARD"
hail_test_gcs_bucket_location = "us-central1"
hail_test_gcs_bucket_storage_class = "STANDARD"

gcp_region = "us-central1"

gcp_zone = "us-central1-a"

gcp_location = "us-central1"

domain = "appsec.hail.is"

# If set to true, pull the base ubuntu image from Artifact Registry.
# Otherwise, assumes GCR.
use_artifact_registry = true
artifact_registry_location = "us"

