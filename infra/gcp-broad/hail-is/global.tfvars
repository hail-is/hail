# organization_domain is a string that is the domain of the organization
# E.g. "hail.is"
organization_domain = "hail.is"

# The GitHub organization hosting your Hail Batch repository, e.g. "hail-is".
github_organization = "hail-is"

# batch_gcp_regions is a JSON array of string, the names of the gcp
# regions to schedule over in Batch. E.g. "[\"us-central1\"]"
batch_gcp_regions = "[\"us-central1\"]"

gcp_project = "hail-vdc"

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

# FIXME: what is this for
gcp_region = "us-central1"

# FIXME: what is this for
gcp_zone = "us-central1-a"

# FIXME: what is this for
gcp_location = "us-central1"

artifact_registry_location = "us"

# FIXME: what is this for
domain = "hail.is"

# If set to true, pull the base ubuntu image from Artifact Registry.
# Otherwise, assumes GCR.
use_artifact_registry = true

k8s_nonpreemptible_node_pool_name = "non-preemptible-pool-11"
k8s_preemptible_node_pool_name = "preemptible-pool-8"

default_subnet_ip_cidr_range = "10.128.0.0/20"
