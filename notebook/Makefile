.PHONY: all build push deploy transform

PROJECT = $(shell gcloud config get-value project)

NAME:="notebook"
TAG = gcr.io/$(PROJECT)/$(NAME):$(shell docker images -q --no-trunc ${NAME} | awk -F':' '{print $$2}')

build:
	docker build -t $(NAME) ./

push:
	docker tag $(NAME) $(TAG)
	docker push $(TAG)

transform:
	cat deployment.yaml.in \
	| sed -e "s,@name@,${NAME}," \
	| sed -e "s,@image@,${TAG}," \
	| sed -e "s,@sha@,$(shell git rev-parse HEAD)," > deployment.yaml

deploy:
	kubectl -n default apply -f deployment.yaml

run-docker-local:
	docker run -p 5000:5000 -e BATCH_USE_KUBE_CONFIG=1 -v ~/:/root notebook

all: build push transform deploy
