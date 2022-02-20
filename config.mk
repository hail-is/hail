CLOUD := gcp
DOCKER_PREFIX := australia-southeast1-docker.pkg.dev/hail-295901/hail
DOCKER_ROOT_IMAGE := australia-southeast1-docker.pkg.dev/hail-295901/hail/ubuntu:20.04
DOMAIN := hail.populationgenomics.org.au
PROJECT := hail-295901
REGION := australia-southeast1
ZONE := australia-southeast1-b
HAIL_TEST_GCS_BUCKET := hail-test-0d3f214ff5
INTERNAL_IP := 10.152.0.10
IP := 35.201.29.236
KUBERNETES_SERVER_URL := https://34.87.199.41

ifeq ($(NAMESPACE),default)
SCOPE = deploy
DEPLOY = true
else
SCOPE = dev
DEPLOY = false
endif
