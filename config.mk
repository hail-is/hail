DOCKER_PREFIX := australia-southeast1-docker.pkg.dev/hail-295901/hail
INTERNAL_IP := 10.152.0.10
IP := 35.201.29.236
DOMAIN := hail.populationgenomics.org.au
CLOUD := gcp
PROJECT := hail-295901
REGION := australia-southeast1
ZONE := australia-southeast1-b

ifeq ($(NAMESPACE),default)
SCOPE = deploy
DEPLOY = true
else
SCOPE = dev
DEPLOY = false
endif
