.PHONY: build push run run-hail rm deploy

build:
	docker build -t spark-base .
	docker build -t spark-master -f Dockerfile.master .
	docker build -t spark-worker -f Dockerfile.worker .
	docker build -t spark-hail -f Dockerfile.hail .

push: SPARK_MASTER_IMAGE="gcr.io/broad-ctsa/spark-master:$(shell docker images -q --no-trunc spark-master | sed -e 's,[^:]*:,,')"
push: SPARK_WORKER_IMAGE="gcr.io/broad-ctsa/spark-worker:$(shell docker images -q --no-trunc spark-worker | sed -e 's,[^:]*:,,')"
push: SPARK_HAIL_IMAGE="gcr.io/broad-ctsa/spark-hail:$(shell docker images -q --no-trunc spark-hail | sed -e 's,[^:]*:,,')"
push: build
	docker tag spark-master $(SPARK_MASTER_IMAGE)
	docker push $(SPARK_MASTER_IMAGE)
	docker tag spark-worker $(SPARK_WORKER_IMAGE)
	docker push $(SPARK_WORKER_IMAGE)
	docker tag spark-hail $(SPARK_HAIL_IMAGE)
	docker push $(SPARK_HAIL_IMAGE)

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
run-hail-pod:
	sed -e "s,@spark_hail_image@,$(SPARK_HAIL_IMAGE),g" \
	  < spark-hail-pod.yaml.in > spark-hail-pod.yaml
	kubectl create -f spark-hail-pod.yaml

deploy: SPARK_MASTER_IMAGE="gcr.io/broad-ctsa/spark-master:$(shell docker images -q --no-trunc spark-master | sed -e 's,[^:]*:,,')"
deploy: SPARK_WORKER_IMAGE="gcr.io/broad-ctsa/spark-worker:$(shell docker images -q --no-trunc spark-worker | sed -e 's,[^:]*:,,')"
deploy: push
	sed -e "s,@spark_master_image@,$(SPARK_MASTER_IMAGE),g" \
	  -e "s,@spark_worker_image@,$(SPARK_WORKER_IMAGE),g" \
	  < deployment.yaml.in > deployment.yaml
	kubectl apply -f deployment.yaml
