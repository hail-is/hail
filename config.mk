DOCKER_PREFIX := gcr.io/hail-vdc
DOCKER_ROOT_IMAGE := $(DOCKER_PREFIX)/ubuntu:20.04
DOMAIN := hail.is
INTERNAL_IP := 10.128.0.57
IP := 35.188.91.25
KUBERNETES_SERVER_URL := https://104.198.230.143

ifeq ($(NAMESPACE),default)
SCOPE = deploy
DEPLOY = true
else
SCOPE = dev
DEPLOY = false
endif
