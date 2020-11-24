include ../config.mk

.PHONY: deployment.yaml build push deploy clean test render render-hail-too watch watch-all

IMAGE = gcr.io/$(PROJECT)/site:$(shell docker images -q --no-trunc site | sed -e 's,[^:]*:,,')
RENDER_COMMAND := PYTHONPATH=$$PYTHONPATH:../web_common:../gear python3 render.py 2>&1 | jq -c '{asctime, funcNameAndLine, message}'

render:
	mkdir -p docs
	$(RENDER_COMMAND)

# For a good development experience, in one terminal window:
#
#     (cd site && python3 -m http.server)
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
	rsync --archive ../hail/build/www/docs/ site/docs/
	$(RENDER_COMMAND)

watch:
	fswatch -o pages/ templates/ docs-templates/ docs-pages/ docs-styles/ dev-docs/ render.py \
    | xargs -n1 -I{} $(MAKE) render

watch-hail-too:
	fswatch -o ../hail/python/ pages/ templates/ | xargs -n1 -I{} $(MAKE) render-hail-too

build: render
	-rm -rf site-out
	rsync --archive \
        --verbose \
        site/* \
        site-out
	-rm ../site.tar.gz
	cd site-out && tar -cvzf ../site.tar.gz *
	docker build -t site .

push: build
	docker tag site $(IMAGE)
	docker push $(IMAGE)

deploy: push
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	$(MAKE) -C ../docker hail-ubuntu
	python3 ../ci/jinja2_render.py '{"global": {"domain":"hail.is"},"default_ns":{"name":"$(NAMESPACE)"},"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"},"deploy":$(DPELOY),"site_image":{"image":"$(IMAGE)"},"hail_ubuntu_image":{"image":"hail-ubuntu"}' deployment.yaml deployment.yaml.out
	kubectl -n $(NAMESPACE) apply -f deployment.yaml.out

clean:
	rm -rf deployment.yaml
