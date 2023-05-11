include ../config.mk

.PHONY: build
build:
	$(MAKE) -C .. admin-pod-image

.PHONY: deploy
deploy: build
	! [ -z $(NAMESPACE) ]  # call this like: make deploy NAMESPACE=default
	python3 ../ci/jinja2_render.py '{"admin_pod_image":{"image":"'$$(cat ../admin-pod-image)'"}}' admin-pod.yaml admin-pod.yaml.out
	kubectl -n $(NAMESPACE) apply -f admin-pod.yaml.out
