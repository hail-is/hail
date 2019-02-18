.PHONY: install-hail-locally build push run run-hail rm deploy src-test-resources

PROJECT = $(shell gcloud config get-value project)

install-hail-locally:
	rm -rf build
	(cd ../hail && ./gradlew shadowJar)
	mkdir -p build/hail/jars
	mkdir -p build/hail/python
	cp -a ../hail/build/libs/hail-all-spark.jar build/hail/jars
	cp -a ../hail/python/hail build/hail/python

build: install-hail-locally
	docker build -t spark-base -f Dockerfile.spark-base .
	docker build -t hail-base -f Dockerfile.hail-base .
	docker build -t spark-master -f Dockerfile.master .
	docker build -t spark-worker -f Dockerfile.worker .
	docker build -t apiserver -f Dockerfile.apiserver .
	docker build -t hail-jupyter -f Dockerfile.hail-jupyter .

push: SPARK_MASTER_IMAGE="gcr.io/$(PROJECT)/spark-master:$(shell docker images -q --no-trunc spark-master | sed -e 's,[^:]*:,,')"
push: SPARK_WORKER_IMAGE="gcr.io/$(PROJECT)/spark-worker:$(shell docker images -q --no-trunc spark-worker | sed -e 's,[^:]*:,,')"
push: APISERVER_IMAGE="gcr.io/$(PROJECT)/apiserver:$(shell docker images -q --no-trunc apiserver | sed -e 's,[^:]*:,,')"
push: HAIL_JUPYTER_IMAGE="gcr.io/$(PROJECT)/hail-jupyter:$(shell docker images -q --no-trunc hail-jupyter | sed -e 's,[^:]*:,,')"
push: build
	docker tag spark-master $(SPARK_MASTER_IMAGE)
	docker push $(SPARK_MASTER_IMAGE)
	docker tag spark-worker $(SPARK_WORKER_IMAGE)
	docker push $(SPARK_WORKER_IMAGE)
	docker tag apiserver $(APISERVER_IMAGE)
	docker push $(APISERVER_IMAGE)
	docker tag hail-jupyter $(HAIL_JUPYTER_IMAGE)
	docker push $(HAIL_JUPYTER_IMAGE)

# to create spark network, run:
# docker network create spark
run:
	docker run --rm -d -p 8080:8080 -p 7077:7077 --network spark --name spark-master --hostname spark-master spark-master
	docker run --rm -d -p 8081:8081 --cpus 2 -m 4g --network spark --name spark-w-1 spark-worker

run-hail:
	docker run --rm -it -p 4040:4040 --network spark spark-hail /bin/bash

rm:
	docker rm -f spark-master spark-w-1

src-test-resources:
	rm -rf src
	mkdir -p src/test
	cp -a ../hail/src/test/resources src/test/

# SPARK_PATH must be set
run-local-apiserver:
	(cd ../hail && ./gradlew shadowJar)
	(cd ../hail && \
  PYTHONPATH=$$(ls $$SPARK_HOME/python/lib/py4j-*-src.zip):$$SPARK_HOME/python:./python \
  JAR=./build/libs/hail-all-spark.jar \
  PYSPARK_SUBMIT_ARGS="--conf spark.driver.extraClassPath=./build/libs/hail-all-spark.jar --conf spark.executor.extraClassPath=./build/libs/hail-all-spark.jar pyspark-shell" \
  python ../apiserver/apiserver/apiserver.py)

# doesn't push
run-hail-jupyter-pod: HAIL_JUPYTER_IMAGE=$(shell kubectl get deployment apiserver -o jsonpath='{.metadata.annotations.hail-jupyter-image}')
run-hail-jupyter-pod:
	sed -e "s,@hail_jupyter_image@,$(HAIL_JUPYTER_IMAGE),g" \
	  < hail-jupyter-pod.yaml.in > hail-jupyter-pod.yaml
	kubectl create -f hail-jupyter-pod.yaml

deploy: SPARK_MASTER_IMAGE="gcr.io/$(PROJECT)/spark-master:$(shell docker images -q --no-trunc spark-master | sed -e 's,[^:]*:,,')"
deploy: SPARK_WORKER_IMAGE="gcr.io/$(PROJECT)/spark-worker:$(shell docker images -q --no-trunc spark-worker | sed -e 's,[^:]*:,,')"
deploy: APISERVER_IMAGE="gcr.io/$(PROJECT)/apiserver:$(shell docker images -q --no-trunc apiserver | sed -e 's,[^:]*:,,')"
deploy: HAIL_JUPYTER_IMAGE="gcr.io/$(PROJECT)/hail-jupyter:$(shell docker images -q --no-trunc hail-jupyter | sed -e 's,[^:]*:,,')"
deploy: push
	sed -e "s,@spark_master_image@,$(SPARK_MASTER_IMAGE),g" \
	  -e "s,@spark_worker_image@,$(SPARK_WORKER_IMAGE),g" \
	  -e "s,@apiserver_image@,$(APISERVER_IMAGE),g" \
	  -e "s,@hail_jupyter_image@,$(HAIL_JUPYTER_IMAGE),g" \
	  < deployment.yaml.in > deployment.yaml
	kubectl apply -f deployment.yaml
