# These values should be lazy so you don't need kubectl for targets that don't
# require it but also only evaluated at most once every make invocation
# https://make.mad-scientist.net/deferred-simple-variable-expansion/

ifdef NAMESPACE
DOCKER_PREFIX = $(eval DOCKER_PREFIX := $$(shell kubectl -n $(NAMESPACE) get secret global-config --template={{.data.docker_prefix}} | base64 --decode))$(DOCKER_PREFIX)
else
DOCKER_PREFIX = docker.io
endif

DOMAIN = $(eval DOMAIN := $$(shell kubectl -n $(NAMESPACE) get secret global-config --template={{.data.domain}} | base64 --decode))$(DOMAIN)
CLOUD = $(eval CLOUD := $$(shell kubectl -n $(NAMESPACE) get secret global-config --template={{.data.cloud}} | base64 --decode))$(CLOUD)
AZURE_SUBSCRIPTION_ID = $(eval AZURE_SUBSCRIPTION_ID := $$(shell kubectl -n $(NAMESPACE) get secret global-config --template={{.data.azure_subscription_id}} | base64 --decode))$(AZURE_SUBSCRIPTION_ID)

ifeq ($(NAMESPACE),default)
SCOPE = deploy
DEPLOY = true
else
SCOPE = dev
DEPLOY = false
endif

TOKEN := $(SCOPE)-$(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)
