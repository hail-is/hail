# Secrets
The volume tests need access to a bucket. I created a service account called
batch-volume-test and gave it access to
gs://hail-ci-0-1-batch-volume-test-bucket. I placed its key into the test
namespace.

```
gsutil mb gs://hail-ci-0-1-batch-volume-test-bucket
gsutil iam ch \
       serviceAccount:batch-volume-tester@hail-vdc.iam.gserviceaccount.com:objectAdmin \
       gs://hail-ci-0-1-batch-volume-test-bucket
```
