.PHONY: run restart-proxy restart-batch-proxy restart-all-proxies
.PHONY: setup-conda-env build-hail-ci push-hail-ci test-locally
.PHONY: run-service deploy

PROJECT := ${shell gcloud config get-value project}

# Grabs details of first instance matching HAIL_CI_INSTANCE
DETAILS := ${shell gcloud compute instances list --format='table(NAME,STATUS,tags.list(),EXTERNAL_IP,ZONE)' | tail -n +2 | grep "${HAIL_CI_INSTANCE}" --color=never | grep RUNNING | head -n 1 | tr -s ' ' '\n' }
INSTANCE := ${word 1, ${DETAILS}}
EXTERNAL_IP := ${word 4, ${DETAILS}}
ZONE := ${word 5, ${DETAILS}}

HAIL_CI_LOCAL_BATCH_PORT ?= 8888

# Must not be indented for conditions to work
check-vars:
ifeq (${strip ${HAIL_CI_INSTANCE}},)
		${error HAIL_CI_INSTANCE variable not set or empty (HAIL_CI_INSTANCE='tag_name or instance_name')}
endif

ifeq (${strip ${INSTANCE}},)
		${error Couldn't find an instance name matching ${HAIL_CI_INSTANCE}}
endif

ifeq (${strip ${EXTERNAL_IP}},)
		${error Couldn't find an external ip for ${HAIL_CI_INSTANCE}}
endif

ifeq (${strip ${ZONE}},)
		${error Couldn't find a zone for ${HAIL_CI_INSTANCE}}
endif
	@echo "\n\nUsing gcloud instance name: ${INSTANCE}, external ip: ${EXTERNAL_IP}, zone: ${ZONE}\n\n"

setup-conda-env:
	conda env create --name hail-ci -f environment.yml

update-conda-env:
	conda env update --name hail-ci -f environment.yml

build-hail-ci:
	cd ../ && docker build . -t hail-ci -f ci/Dockerfile

push-hail-ci: IMAGE = gcr.io/${PROJECT}/hail-ci:${shell docker images -q --no-trunc hail-ci | head -n 1 | sed -e 's,[^:]*:,,'}
push-hail-ci: build-hail-ci
	docker tag hail-ci ${IMAGE}
	docker push ${IMAGE}
	echo ${IMAGE} > hail-ci-image
	echo built ${IMAGE}

restart-all-proxies: restart-proxy restart-batch-proxy

restart-proxy: check-vars
restart-proxy:
	-kill $(shell cat proxy.pid)
	-kill -9 $(shell cat proxy.pid)
	-rm -rf proxy.pid

	${shell gcloud compute ssh --project ${PROJECT} --zone ${ZONE} ${INSTANCE} \
	  --ssh-flag="-R 0.0.0.0:${HAIL_CI_REMOTE_PORT}:127.0.0.1:5000" \
	  --ssh-flag='-N' \
	  --ssh-flag='-T' \
	  --ssh-flag='-v' \
	  --dry-run} > proxy.log 2>proxy.err & echo $$! > proxy.pid

restart-batch-proxy: check-vars
restart-batch-proxy:
	-kill $(shell cat batch-proxy.pid)
	-kill -9 $(shell cat batch-proxy.pid)
	-rm -rf batch-proxy.pid
	${eval BATCH_POD := ${shell kubectl get pods \
                           -l app=batch \
                           --field-selector=status.phase==Running \
                           -o name \
                         | sed 's:pods/::' \
                         | head -n 1}}
	kubectl port-forward ${BATCH_POD} 8888:5000 > batch-proxy.log 2>batch-proxy.err & echo $$! > batch-proxy.pid

run-local: HAIL_CI_REMOTE_PORT = 3000
# SELF_HOSTNAME and BATCH_SERVER_URL assignment run in shell context
# and therefore cannot have a space (else it's a 2 operand command)
run-local: restart-all-proxies
	SELF_HOSTNAME=http://${EXTERNAL_IP}:${HAIL_CI_REMOTE_PORT} \
	BATCH_SERVER_URL=http://127.0.0.1:${HAIL_CI_LOCAL_BATCH_PORT} \
	source activate hail-ci && python run_ci.py

run-local-for-tests: HAIL_CI_REMOTE_PORT = 3001
run-local-for-tests: restart-all-proxies
	SELF_HOSTNAME=http://${EXTERNAL_IP}:${HAIL_CI_REMOTE_PORT} \
	BATCH_SERVER_URL=http://127.0.0.1:${HAIL_CI_LOCAL_BATCH_PORT} \
	WATCHED_TARGETS='[["hail-is/ci-test:master", true]]' \
	source activate hail-ci && pip install ../batch && python run_ci.py

test-locally: HAIL_CI_REMOTE_PORT = 3001
test-locally: restart-all-proxies
	SELF_HOSTNAME=http://${EXTERNAL_IP}:${HAIL_CI_REMOTE_PORT} \
	BATCH_SERVER_URL=http://127.0.0.1:${HAIL_CI_LOCAL_BATCH_PORT} \
	./test-locally.sh

run-service:
	kubectl apply -f k8s/service.yaml

deploy: push-hail-ci
	sed -e "s,@sha@,$(shell git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$(shell cat hail-ci-image)," \
	  < deployment.yaml.in > k8s/deployment.yaml
	kubectl -n default apply -f k8s
