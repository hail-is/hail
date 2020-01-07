PROJECT := $(shell gcloud config get-value project)

SERVICE_BASE_IMAGE = gcr.io/$(PROJECT)/service-base:$(shell docker images -q --no-trunc service-base:latest | sed -e 's,[^:]*:,,')

.PHONY: deploy
deploy:
	make -C ../docker push
	python3 ../ci/jinja2_render.py '{"service_base_image":{"image":"$(SERVICE_BASE_IMAGE)"}}' admin-pod.yaml admin-pod.yaml.out
	kubectl -n default apply -f admin-pod.yaml.out
