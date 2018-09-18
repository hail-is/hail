build:
	docker build . -t scorecard

push:
	docker tag scorecard gcr.io/broad-ctsa/scorecard
	docker push gcr.io/broad-ctsa/scorecard

run-docker:
	docker run -i -p 5000:5000 -v secrets:/secrets -t scorecard

run:
	GITHUB_TOKEN_PATH=secrets/scorecard-github-access-token.txt python scorecard/scorecard.py

deploy:
	kubectl apply -f deployment.yaml
