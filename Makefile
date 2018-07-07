build:
	docker build -t gcr.io/broad-ctsa/batch -t batch .

push:
	docker push gcr.io/broad-ctsa/batch

run:
	BATCH_USE_KUBE_CONFIG=1 docker run -i -v $HOME/.kube:/root/.kube -p 5000:5000 -t batch
