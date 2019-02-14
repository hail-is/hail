.PHONY: all build push deploy transform

PROJECT = $(shell gcloud config get-value project)

SOURCE_NAME := notebook
TARGET_NAME = gcr.io/$(PROJECT)/$(SOURCE_NAME):$(shell docker images -q --no-trunc $(SOURCE_NAME) | awk -F':' '{print $$2}')

build:
	docker build -t $(SOURCE_NAME) ./

push:
	docker tag $(SOURCE_NAME) $(TARGET_NAME)
	docker push $(TARGET_NAME)

transform:
	cat deployment.yaml.in \
	| sed -e "s,@name@,$(SOURCE_NAME)," \
	| sed -e "s,@image@,$(TARGET_NAME)," \
	| sed -e "s,@sha@,$(shell git rev-parse HEAD)," > deployment.yaml

deploy:
	kubectl -n default apply -f deployment.yaml

run-docker-local:
	docker run -p 5000:5000 -e BATCH_USE_KUBE_CONFIG=1 -v ~/:/root notebook

all: build push transform deploy
