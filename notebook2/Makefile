.PHONY: build push deploy clean

PROJECT = $(shell gcloud config get-value project)
SOURCE_NAME := $(shell basename $(CURDIR))
TARGET_NAME = gcr.io/$(PROJECT)/$(SOURCE_NAME):$(shell docker images -q --no-trunc $(SOURCE_NAME) | awk -F':' '{print $$2}')
DEPLOYMENT_FILE := deployment.yaml

build:
	docker build -t $(SOURCE_NAME) ./

push: build
	docker tag $(SOURCE_NAME) $(TARGET_NAME)
	docker push $(TARGET_NAME)

$(DEPLOYMENT_FILE): build push
	sed < deployment.yaml.in \
	 -e "s,@name@,$(SOURCE_NAME)," \
	 -e "s,@image@,$(TARGET_NAME)," \
	 -e "s,@sha@,$(shell git rev-parse HEAD)," > $@

deploy: $(DEPLOYMENT_FILE)
	kubectl -n default apply -f $<

run-docker-local: build
	docker run -p 5000:5000 -e BATCH_USE_KUBE_CONFIG=1 -v ~/:/root $(SOURCE_NAME)

clean:
	rm -rf $(DEPLOYMENT_FILE)
