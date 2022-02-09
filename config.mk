DOCKER_PREFIX := gcr.io/hail-vdc
INTERNAL_IP := 10.128.0.57
IP := 35.188.91.25
DOMAIN := hail.is
CLOUD := gcp

ifeq ($(NAMESPACE),default)
SCOPE = deploy
DEPLOY = true
else
SCOPE = dev
DEPLOY = false
endif
