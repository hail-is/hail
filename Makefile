
build:
	docker build -t gcr.io/broad-ctsa/batch -t batch .

push:
	docker push gcr.io/broad-ctsa/batch
