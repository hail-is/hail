set +ex

gcloud auth activate-service-account \
  --key-file=/secrets/ci-deploy-0-1--hail-is-ci-test.json

DEPLOY_SHA=$(git rev-parse HEAD)
gsutil ls gs://hail-ci-test/${DEPLOY_SHA} || (
    git show HEAD > foo
    gsutil cp foo gs://hail-ci-test/$(git rev-parse HEAD)
)
