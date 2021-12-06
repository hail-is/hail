DOCKER_PREFIX := dgoldste.azurecr.io
INTERNAL_IP := 10.128.255.254
IP := 20.72.173.53
DOMAIN := daniel-azure.hail.is

ifeq ($(NAMESPACE),default)
SCOPE = deploy
DEPLOY = true
else
SCOPE = dev
DEPLOY = false
endif
