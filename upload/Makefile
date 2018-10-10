.PHONY: build push redeploy

build:
	docker build . -t upload

push: IMAGE="gcr.io/broad-ctsa/upload:$(shell docker images -q --no-trunc upload | sed -e 's,[^:]*:,,')"
push: build
	docker tag upload $(IMAGE)
	docker push $(IMAGE)
	echo $(IMAGE) > upload-image

deploy:
	sed -e "s,@sha@,$SHA," \
	  -e "s,@image@,$$(cat upload-image)," \
	  < deployment.yaml.in > deployment.yaml
	kubectl apply -f deployment.yaml
