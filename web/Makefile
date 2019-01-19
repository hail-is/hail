.PHONY: build deploy all

PROJECT=$(shell gcloud config get-value project)

LOCAL_TAG_WEB=hail-web-client
PUBLISH_TAG_WEB=gcloud.io/$(PROJECT)/web/$(LOCAL_TAG_WEB)
SECRET_NAME_WEB=hail-web-client-secrets

build:
		-kubectl delete secret $(SECRET_NAME_WEB)
		kubectl create secret generic $(SECRET_NAME_WEB) --from-env-file=./packages/web-client/.env
		docker build packages/web-client -t $(LOCAL_TAG_WEB) --no-cache
		docker tag $(LOCAL_TAG_WEB) $(PUBLISH_TAG_WEB)

deploy:
	cat ./deployment.yaml.in \
	| sed -e "s,@web_client_image@,${PUBLISH_TAG_WEB}," \
	| sed -e "s,@web_client_secret@,${SECRET_NAME_WEB}," \
	> deployment.yaml
	kubectl apply -f deployment.yaml
		
all: build deploy