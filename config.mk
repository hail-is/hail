DOCKER_PREFIX := $(shell kubectl get secret global-config --template={{.data.docker_prefix}} | base64 --decode)
DOMAIN := $(shell kubectl get secret global-config --template={{.data.domain}} | base64 --decode)

ifeq ($(NAMESPACE),default)
SCOPE = deploy
DEPLOY = true
else
SCOPE = dev
DEPLOY = false
endif
