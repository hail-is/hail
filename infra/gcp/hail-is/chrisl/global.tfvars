# organization_domain is a string that is the domain of the organization
# E.g. "hail.is"
organization_domain = "broadinstitute.org"

# The GitHub organization hosting your Hail Batch repository, e.g. "hail-is".
# Matching the location of your project files within the infra/gcp directory.
# eg hail.is/sandbox
github_organization = "hail-is/chrisl"

# batch_gcp_regions is a JSON array of string, the names of the gcp
# regions to schedule over in Batch. E.g. "[\"us-central1\"]"
batch_gcp_regions = "[\"us-central1\"]"

gcp_project = "hail-vdc-chrisl"

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

domain = "chrisl.hail.is"

# Optional: Support email address to display in error pages and user-facing messages
# If not set, error pages will display "email support" without a link
# If set, error pages will display a clickable mailto link
support_email = "hail-team@broadinstitute.org"
