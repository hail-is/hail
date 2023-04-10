# These values should be lazy so you don't need kubectl for targets that don't
# require it but also only evaluated at most once every make invocation
# https://make.mad-scientist.net/deferred-simple-variable-expansion/
DOCKER_PREFIX = $(eval DOCKER_PREFIX := $$(shell kubectl get secret global-config --template={{.data.docker_prefix}} | base64 --decode))$(DOCKER_PREFIX)
DOMAIN = $(eval DOMAIN := $$(shell kubectl get secret global-config --template={{.data.domain}} | base64 --decode))$(DOMAIN)
CLOUD = $(eval CLOUD := $$(shell kubectl get secret global-config --template={{.data.cloud}} | base64 --decode))$(CLOUD)

ifeq ($(NAMESPACE),default)
SCOPE = deploy
DEPLOY = true
else
SCOPE = dev
DEPLOY = false
endif

TOKEN = $(SCOPE)-$(shell cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 12)
