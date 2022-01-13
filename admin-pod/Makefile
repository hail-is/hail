include ../config.mk

.PHONY: deploy
deploy:
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	$(MAKE) -C ../docker service-base
	python3 ../ci/jinja2_render.py '{"deploy":$(DEPLOY),"service_base_image":{"image":"'$$(cat ../docker/service-base-image-ref)'"}}' admin-pod.yaml admin-pod.yaml.out
	kubectl -n $(NAMESPACE) apply -f admin-pod.yaml.out
