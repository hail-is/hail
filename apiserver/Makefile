.PHONY: install-hail-locally build push run run-hail rm deploy build-hail-shadow-jar

PROJECT := $(shell gcloud config get-value project)
GSA_KEY_FILE = /gsa-key/privateKeyData

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

build-common: install-hail-locally jupyter_notebook_config.py

ifeq ($(IN_HAIL_CI),1)
build-spark-base: build-common
	docker pull debian:9.5
	-docker pull gcr.io/$(PROJECT)/spark-base:latest
	docker build -t spark-base -f Dockerfile.spark-base . --cache-from spark-base,gcr.io/$(PROJECT)/spark-base:latest,debian:9.5

build-hail-base: build-common
	-docker pull gcr.io/$(PROJECT)/spark-base:latest
	-docker pull gcr.io/$(PROJECT)/hail-base:latest
	docker build -t hail-base -f Dockerfile.hail-base . --cache-from hail-base,gcr.io/$(PROJECT)/hail-base:latest,spark-base

build-spark-master: build-common
	-docker pull gcr.io/$(PROJECT)/hail-base:latest
	-docker pull gcr.io/$(PROJECT)/spark-master:latest
	docker build -t spark-master -f Dockerfile.spark-master . --cache-from spark-master,gcr.io/$(PROJECT)/spark-master:latest,hail-base

build-spark-worker: build-common
	-docker pull gcr.io/$(PROJECT)/hail-base:latest
	-docker pull gcr.io/$(PROJECT)/spark-worker:latest
	docker build -t spark-worker -f Dockerfile.spark-worker . --cache-from spark-worker,gcr.io/$(PROJECT)/spark-worker:latest,hail-base

build-apiserver: build-common
	-docker pull gcr.io/$(PROJECT)/hail-base:latest
	-docker pull gcr.io/$(PROJECT)/apiserver:latest
	docker build -t apiserver -f Dockerfile.apiserver . --cache-from apiserver,gcr.io/$(PROJECT)/apiserver:latest,hail-base

build-hail-jupyter: build-common
	-docker pull gcr.io/$(PROJECT)/hail-base:latest
	-docker pull gcr.io/$(PROJECT)/hail-jupyter:latest
	docker build -t hail-jupyter -f Dockerfile.hail-jupyter . --cache-from hail-jupyter,gcr.io/$(PROJECT)/hail-jupyter:latest,hail-base

build: build-spark-base build-hail-base build-spark-master build-spark-worker build-apiserver build-hail-jupyter

SPARK_MASTER_IMAGE="gcr.io/$(PROJECT)/spark-master:$(shell docker images -q --no-trunc spark-master | sed -e 's,[^:]*:,,')"
SPARK_WORKER_IMAGE="gcr.io/$(PROJECT)/spark-worker:$(shell docker images -q --no-trunc spark-worker | sed -e 's,[^:]*:,,')"
APISERVER_IMAGE="gcr.io/$(PROJECT)/apiserver:$(shell docker images -q --no-trunc apiserver | sed -e 's,[^:]*:,,')"
HAIL_JUPYTER_IMAGE="gcr.io/$(PROJECT)/hail-jupyter:$(shell docker images -q --no-trunc hail-jupyter | sed -e 's,[^:]*:,,')"

push: build
	docker tag spark-master $(SPARK_MASTER_IMAGE)
	docker push $(SPARK_MASTER_IMAGE)
	docker tag spark-worker $(SPARK_WORKER_IMAGE)
	docker push $(SPARK_WORKER_IMAGE)
	docker tag apiserver $(APISERVER_IMAGE)
	docker push $(APISERVER_IMAGE)
	docker tag hail-jupyter $(HAIL_JUPYTER_IMAGE)
	docker push $(HAIL_JUPYTER_IMAGE)

deploy: push
	sed -e "s,@spark_master_image@,$(SPARK_MASTER_IMAGE),g" \
	  -e "s,@spark_worker_image@,$(SPARK_WORKER_IMAGE),g" \
	  -e "s,@apiserver_image@,$(APISERVER_IMAGE),g" \
	  -e "s,@hail_jupyter_image@,$(HAIL_JUPYTER_IMAGE),g" \
	  < deployment.yaml.in > deployment.yaml
	kubectl -n default apply -f deployment.yaml
else
build: build-common
	docker build -t spark-base -f Dockerfile.spark-base .
	docker build -t hail-base -f Dockerfile.hail-base .
	docker build -t spark-master -f Dockerfile.spark-master .
	docker build -t spark-worker -f Dockerfile.spark-worker .
	docker build -t apiserver -f Dockerfile.apiserver .
	docker build -t hail-jupyter -f Dockerfile.hail-jupyter .

err:="\nAh ah ah! You're not CI and you didn't say the magic word\n"
push:
	@echo $(err)
deploy:
	@echo $(err)
endif

build-hail-shadow-jar:
	cd ../hail && ./gradlew shadowJar

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

test: build-hail-shadow-jar
	./test-apiserver.sh

clean:
	rm jupyter_notebook_config.py
