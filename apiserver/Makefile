.PHONY: install-hail-locally build push run run-hail rm deploy build-hail-shadow-jar

PROJECT := $(shell gcloud config get-value project)
GSA_KEY_FILE = /gsa-key/privateKeyData

ifeq ($(IN_HAIL_CI),1)
define DOCKER_IMAGE
$(3) = gcr.io/$(PROJECT)/$(1):$$(shell docker images -q --no-trunc $(1) | sed -e 's,[^:]*:,,')
build-$(1): $(2)
	echo $$(3)
	-docker pull gcr.io/$(PROJECT)/$(1):latest
	-docker pull $$(shell grep FROM Dockerfile.$(1) | head -n 1 | sed 's/^HEAD //')
	docker build -t $(1) -f Dockerfile.$(1) . --cache-from $(1),gcr.io/$(PROJECT)/$(1):latest,$$(shell grep FROM Dockerfile.$(1) | head -n 1 | sed 's/^FROM //')

push-$(1): build-$(1)
	docker tag $(1) gcr.io/$(PROJECT)/$(1):latest
	docker push gcr.io/$(PROJECT)/$(1):latest
	docker tag $(1) gcr.io/$(PROJECT)/$(1):$$(shell docker images -q --no-trunc $(1) | sed -e 's,[^:]*:,,')
	docker push gcr.io/$(PROJECT)/$(1):$$(shell docker images -q --no-trunc $(1) | sed -e 's,[^:]*:,,')
endef
else
define DOCKER_IMAGE
$(3) = gcr.io/$(PROJECT)/$(1):$$(shell docker images -q --no-trunc $(1) | sed -e 's,[^:]*:,,')
build-$(1): $(2)
	docker build -t $(1) -f Dockerfile.$(1) .
push-$(1): build-$(1)
	docker tag $(1) gcr.io/$(PROJECT)/$(1):$$(shell docker images -q --no-trunc $(1) | sed -e 's,[^:]*:,,')
	docker push gcr.io/$(PROJECT)/$(1):$$(shell docker images -q --no-trunc $(1) | sed -e 's,[^:]*:,,')
endef
endif

build-hail-shadow-jar:
	cd ../hail && ./gradlew shadowJar

install-hail-locally:
	rm -rf build
	(cd ../hail && GRADLE_OPTS=-Xmx2048m ./gradlew shadowJar --gradle-user-home /gradle-cache)
	mkdir -p build/hail/jars
	mkdir -p build/hail/python
	cp -a ../hail/build/libs/hail-all-spark.jar build/hail/jars
	cp -a ../hail/python/hail build/hail/python

jupyter_notebook_config.py:
	sed -e "s,@project@,$(PROJECT),g" \
	    -e "s,@keyfile@,$(GSA_KEY_FILE),g" \
            < jupyter_notebook_config.py.in > $@

$(eval $(call DOCKER_IMAGE,spark-base,install-hail-locally jupyter_notebook_config.py,SPARK_BASE_IMAGE))
$(eval $(call DOCKER_IMAGE,hail-base,install-hail-locally jupyter_notebook_config.py,HAIL_BASE_IMAGE))
$(eval $(call DOCKER_IMAGE,spark-master,install-hail-locally jupyter_notebook_config.py,SPARK_MASTER_IMAGE))
$(eval $(call DOCKER_IMAGE,spark-worker,install-hail-locally jupyter_notebook_config.py,SPARK_WORKER_IMAGE))
$(eval $(call DOCKER_IMAGE,apiserver,install-hail-locally jupyter_notebook_config.py,APISERVER_IMAGE))
$(eval $(call DOCKER_IMAGE,hail-jupyter,install-hail-locally jupyter_notebook_config.py,HAIL_JUPYTER_IMAGE))

build: build-spark-base build-hail-base build-spark-master build-spark-worker build-apiserver build-hail-jupyter

push: push-spark-base push-hail-base push-spark-master push-spark-worker push-apiserver push-hail-jupyter

# to create spark network, run:
# docker network create spark
run:
	docker run --rm -d -p 8080:8080 -p 7077:7077 --network spark --name spark-master --hostname spark-master spark-master
	docker run --rm -d -p 8081:8081 --cpus 2 -m 4g --network spark --name spark-w-1 spark-worker

run-hail:
	docker run --rm -it -p 4040:4040 --network spark spark-hail /bin/bash

rm:
	docker rm -f spark-master spark-w-1

# doesn't push
run-hail-jupyter-pod: HAIL_JUPYTER_IMAGE=$(shell kubectl get deployment apiserver -o jsonpath='{.metadata.annotations.hail-jupyter-image}')
run-hail-jupyter-pod:
	sed -e "s,@hail_jupyter_image@,$(HAIL_JUPYTER_IMAGE),g" \
	  < hail-jupyter-pod.yaml.in > hail-jupyter-pod.yaml
	kubectl create -f hail-jupyter-pod.yaml

deploy: push
	sed -e "s,@spark_master_image@,$(SPARK_MASTER_IMAGE),g" \
	  -e "s,@spark_worker_image@,$(SPARK_WORKER_IMAGE),g" \
	  -e "s,@apiserver_image@,$(APISERVER_IMAGE),g" \
	  -e "s,@hail_jupyter_image@,$(HAIL_JUPYTER_IMAGE),g" \
	  < deployment.yaml.in > deployment.yaml
	kubectl -n default apply -f deployment.yaml

test: build-hail-shadow-jar
	./test-apiserver.sh

clean:
	rm jupyter_notebook_config.py
