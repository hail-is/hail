ifeq ($(CLOUD),azure)

SUBSCRIPTION_ID := $(shell az account list | jq -rj '.[0].id')
REGION := eastus
RESOURCE_GROUP := hail-dev
CONTAINER_REGISTRY_NAME := $(RESOURCE_GROUP)
SHARED_GALLERY_NAME := batch
DOCKER_PREFIX := $(CONTAINER_REGISTRY_NAME).azurecr.io

else

PROJECT := hail-vdc
DOCKER_PREFIX := gcr.io/$(PROJECT)
HAIL_TEST_GCS_BUCKET := hail-test-dmk9z
DOMAIN := hail.is
INTERNAL_IP := 10.128.0.57
IP := 35.188.91.25
KUBERNETES_SERVER_URL := https://104.198.230.143
REGION := us-central1
ZONE := us-central1-a

endif

DOCKER_ROOT_IMAGE := $(DOCKER_PREFIX)/ubuntu:18.04

ifeq ($(NAMESPACE),default)
SCOPE = deploy
DEPLOY = true
else
SCOPE = dev
DEPLOY = false
endif
