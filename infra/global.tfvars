organization_domain = "populationgenomics.org.au"

# batch_gcp_regions is a JSON array of string, the names of the gcp
# regions to schedule over in Batch.
batch_gcp_regions = "[\"australia-southeast1\"]"

gcp_project = "hail-295901"
gcp_location = "australia-southeast1"
gcp_region = "australia-southeast1"
gcp_zone = "australia-southeast1-b"
domain = "hail.populationgenomics.org.au"
use_artifact_registry = true

# This is the bucket location that spans the regions you're going to
# schedule across in Batch. If you are running on one region, it can
# just be that region. E.g. "US"
batch_logs_bucket_location = "australia-southeast1"

# The storage class for the batch logs bucket. It should span the
# batch regions and be compatible with the bucket location.
batch_logs_bucket_storage_class = "STANDARD"

# Similarly, bucket locations and storage classess are specified 
# for other services:
hail_query_bucket_location = "australia-southeast1"
hail_query_bucket_storage_class = "STANDARD"
hail_test_gcs_bucket_location = "australia-southeast1"
hail_test_gcs_bucket_storage_class = "REGIONAL"
