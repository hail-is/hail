DOCKER_PREFIX := haildev.azurecr.io
INTERNAL_IP := 10.128.255.254
IP := 20.62.247.135
DOMAIN := azure.hail.is

ifeq ($(NAMESPACE),default)
SCOPE = deploy
DEPLOY = true
else
SCOPE = dev
DEPLOY = false
endif
