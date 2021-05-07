PROJECT := ukbb-exome-pharma
DOCKER_PREFIX := gcr.io/$(PROJECT)
DOCKER_ROOT_IMAGE := $(DOCKER_PREFIX)/ubuntu:18.04
DOMAIN := deadbeef.hail.is
INTERNAL_IP := 10.128.0.65
IP := 35.224.8.209
KUBERNETES_SERVER_URL := https://35.202.2.149
REGION := us-central1
ZONE := us-central1-a

ifeq ($(NAMESPACE),default)
SCOPE = deploy
DEPLOY = true
else
SCOPE = dev
DEPLOY = false
endif
