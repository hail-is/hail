include ../config.mk

.PHONY: deployment.yaml build push deploy clean test render render-hail-too watch watch-hail-too

IMAGE = $(DOCKER_PREFIX)/site:$(shell docker images -q --no-trunc site | sed -e 's,[^:]*:,,')
RENDER_COMMAND := PYTHONPATH=$$PYTHONPATH:../web_common:../gear python3 render.py

render:
	$(RENDER_COMMAND)

# For a good development experience, in one terminal window:
#
#     (cd www && python3 -m http.server)
#
# and in another terminal window:
#
#     make render-hail-too
#
# If you do not care about the hail docs,
#
#     make render
#
# will be faster.
render-hail-too:
	$(MAKE) -C ../hail hail-docs-no-test batch-docs
	rsync --archive --verbose ../hail/build/www/docs/ www/docs/
	$(RENDER_COMMAND)

watch:
	fswatch -o pages/ templates/ | xargs -n1 -I{} /bin/bash -c '$(RENDER_COMMAND)'

watch-hail-too:
	fswatch -o ../hail/python/ pages/ templates/ | xargs -n1 -I{} $(MAKE) render-hail-too

build: render
	-rm -rf www-out
	rsync --archive \
        --verbose \
        www/* \
        www-out
	-rm ../site-www.tar.gz
	cd www-out && tar -cvzf ../site-www.tar.gz *
	docker build -t site .

push: build
	docker tag site $(IMAGE)
	docker push $(IMAGE)

deploy: push
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	$(MAKE) -C ../docker hail-ubuntu
	python3 ../ci/jinja2_render.py '{"global": {"domain":"$(DOMAIN)"},"default_ns":{"name":"$(NAMESPACE)"},"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DEPLOY),"site_image":{"image":"$(IMAGE)"},"hail_ubuntu_image":{"image":"hail-ubuntu"}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out

clean:
	rm -rf deployment.yaml
