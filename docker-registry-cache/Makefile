include ../config.mk

.PHONY: deploy

deploy:
	kubectl -n default apply -f service.yaml
	python3 ../ci/jinja2_render.py '{"code":{"sha":"$(shell git rev-parse --short=12 HEAD)"}}' deployment.yaml deployment.yaml.out
	kubectl -n default apply -f deployment.yaml.out
