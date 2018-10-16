build:
	docker build -t spark-base .
	docker build -t spark-master -f Dockerfile.master .
	docker build -t spark-worker -f Dockerfile.worker .
	docker build -t spark-hail -f Dockerfile.hail .

run:
	docker run --rm -d -p 8080:8080 -p 7077:7077 --network spark --name spark-master --hostname spark-master spark-master
# workers
	docker run --rm -d --cpus 2 -m 4g --network spark --name spark-w-1 spark-worker

rm:
	docker rm -f spark-master spark-w-1 spark-w-2
