.PHONY: build push deploy clean

SITE_IMAGE = gcr.io/$(PROJECT)/site:$(shell docker images -q --no-trunc site | sed -e 's,[^:]*:,,')
SITE_POLL_IMAGE = gcr.io/$(PROJECT)/site-poll:$(shell docker images -q --no-trunc site-poll | sed -e 's,[^:]*:,,')

deployment.yaml: build deployment.yaml.in
	sed -e "s,@sha@,$$(git rev-parse --short=12 HEAD),g" \
	  -e "s,@site_image@,$(SITE_IMAGE)," \
	  -e "s,@site_poll_image@,$(SITE_POLL_IMAGE)," \
	  < $< > $@

build: Dockerfile hail.nginx.conf Dockerfile.site-poll poll.sh poll-0.1.sh poll-0.2.sh
	docker build -t site .
	docker build -t site-poll -f Dockerfile.site-poll .

push: build
	docker tag site $(SITE_IMAGE)
	docker push $(SITE_IMAGE)
	docker tag site-poll $(SITE_POLL_IMAGE)
	docker push $(SITE_POLL_IMAGE)

deploy: push service.yaml deployment.yaml
	kubectl apply -f service.yaml deployment.yaml

clean:
	rm -rf deployment.yaml
