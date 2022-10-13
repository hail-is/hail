DOCKER_PREFIX := hailazureterradev.azurecr.io
DOMAIN := hail.fake

ifeq ($(NAMESPACE),default)
SCOPE = deploy
DEPLOY = true
else
SCOPE = dev
DEPLOY = false
endif
