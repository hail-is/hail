build:
	docker build -t spark-base .
	docker build -t spark-master -f Dockerfile.master .
	docker build -t spark-worker -f Dockerfile.worker .
	docker build -t spark-hail -f Dockerfile.hail .

push:
	docker tag spark-master gcr.io/broad-ctsa/spark-master
	docker push gcr.io/broad-ctsa/spark-master
	docker tag spark-worker gcr.io/broad-ctsa/spark-worker
	docker push gcr.io/broad-ctsa/spark-worker
	docker tag spark-hail gcr.io/broad-ctsa/spark-hail
	docker push gcr.io/broad-ctsa/spark-hail

run:
	docker run --rm -d -p 8080:8080 -p 7077:7077 --network spark --name spark-master --hostname spark-master spark-master
# workers
	docker run --rm -d -p 8081:8081 --cpus 2 -m 4g --network spark --name spark-w-1 spark-worker

run-hail:
	docker run --rm -it -p 4040:4040 --network spark spark-hail /bin/bash

rm:
	docker rm -f spark-master spark-w-1 spark-w-2
