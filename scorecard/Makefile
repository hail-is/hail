.PHONY: build push run-docker run deploy

PROJECT = $(shell gcloud config get-value project)

SCORECARD_IMAGE = gcr.io/$(PROJECT)/scorecard:$(shell docker images -q --no-trunc scorecard:latest | sed -e 's,[^:]*:,,')

build:
	-docker pull gcr.io/$(PROJECT)/scorecard
	docker build . -t scorecard --cache-from scorecard,base,ubuntu:18.04

push: build
	docker tag scorecard $(SCORECARD_IMAGE)
	docker push $(SCORECARD_IMAGE)

run-docker: build
	docker run -i -p 5000:5000 -v `pwd`/secrets:/secrets -t scorecard

run:
	GITHUB_TOKEN_PATH=secrets/scorecard-github-access-token.txt python scorecard/scorecard.py

deploy: push
	sed -e "s,@sha@,$(shell git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$(SCORECARD_IMAGE)," \
	  < deployment.yaml.in > deployment.yaml
	kubectl -n default apply -f deployment.yaml
