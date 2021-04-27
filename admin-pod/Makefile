include ../config.mk

SERVICE_BASE_IMAGE = $(DOCKER_PREFIX)/service-base:$(shell docker images -q --no-trunc service-base:latest | sed -e 's,[^:]*:,,')

.PHONY: deploy
deploy:
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	$(MAKE) -C ../docker push
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"$(SERVICE_BASE_IMAGE)"}}' admin-pod.yaml admin-pod.yaml.out
	kubectl -n $(NAMESPACE) apply -f admin-pod.yaml.out
