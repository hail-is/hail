DOCKER_PREFIX := $(shell kubectl get secret global-config --template={{.data.docker_prefix}} | base64 --decode)
DOMAIN := $(shell kubectl get secret global-config --template={{.data.domain}} | base64 --decode)

ifeq ($(NAMESPACE),default)
SCOPE = deploy
DEPLOY = true
else
SCOPE = dev
DEPLOY = false
endif

TOKEN = $(SCOPE)-$(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)
