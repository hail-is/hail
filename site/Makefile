.PHONY: build build-site build-run-letsencrypt push push-site push-run-letsencrypt \
  run-letsencrypt serve deploy-site

build: build-site build-run-letsencrypt

build-site:
	docker build -t site .

build-run-letsencrypt:
	docker build -t run-letsencrypt . -f Dockerfile.run-letsencrypt

push: push-site push-run-letsencrypt

push-site: IMAGE = gcr.io/broad-ctsa/site:$(shell docker images -q --no-trunc site | sed -e 's,[^:]*:,,')
push-site: build-site
	echo $(IMAGE) > site-image
	docker tag site $(IMAGE)
	docker push $(IMAGE)

push-run-letsencrypt: build-run-letsencrypt
	docker tag run-letsencrypt gcr.io/broad-ctsa/run-letsencrypt
	docker push gcr.io/broad-ctsa/run-letsencrypt

run-letsencrypt:
	run-letsencrypt.sh

serve:
	docker run -it -p 80:80 -p 443:443 -v $$(pwd)/letsencrypt:/etc/letsencrypt site

deploy-site:
	kubectl apply -f service.yaml
	sed -e "s,@sha@,$(shell git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$(shell cat site-image)," \
	  < site-deployment.yaml.in > site-deployment.yaml
	kubectl apply -f site-deployment.yaml
