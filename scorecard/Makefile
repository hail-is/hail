.PHONY: build push run-docker run deploy

build:
	docker build . -t scorecard

push-site: IMAGE = gcr.io/broad-ctsa/scorecard:$(shell docker images -q --no-trunc site | sed -e 's,[^:]*:,,')
push: build
	echo $(IMAGE) > scorecard-image
	docker tag scorecard $(IMAGE)
	docker push $(IMAGE)

run-docker:
	docker run -i -p 5000:5000 -v secrets:/secrets -t scorecard

run:
	GITHUB_TOKEN_PATH=secrets/scorecard-github-access-token.txt python scorecard/scorecard.py

deploy:
	sed -e "s,@sha@,$(shell git rev-parse --short=12 HEAD)," \
	  -e "s,@image@,$(shell cat scorecard-image)," \
	  < site-deployment.yaml.in > site-deployment.yaml
	kubectl apply -f deployment.yaml
