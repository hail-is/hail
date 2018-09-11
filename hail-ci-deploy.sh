set -ex

SHA=$(git rev-parse --short=12 HEAD)

# get sha label of batch deployment
DEPLOYED_SHA=$(kubectl get --selector=app=batch deployments -o jsonpath="{.items[*].metadata.labels['hail\.is/sha']}")

if [[ "$SHA" == "$DEPLOYED_SHA" ]]; then
    exit 0
fi

# requires docker
make push-hail-ci-build-image

sed -e "s,@sha@,$SHA," \
    -e "s,@image@,$(cat hail-ci-build-image)," \
    < deployment.yaml.in > deployment.yaml

kubectl apply -f deployment.yaml
