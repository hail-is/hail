PROJECT := hail-vdc-staging
DOCKER_ROOT_IMAGE := gcr.io/$(PROJECT)/ubuntu:18.04
DOMAIN := staging.hail.is
INTERNAL_IP := 10.128.0.2
IP := 34.120.221.136
KUBERNETES_SERVER_URL := https://34.71.246.49
REGION := us-central1
ZONE := us-central1-a
ifeq ($(NAMESPACE),default)
SCOPE = deploy
DEPLOY = true
else
SCOPE = dev
DEPLOY = false
endif
