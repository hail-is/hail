build:
	docker build . -t scorecard

push:
	docker tag scorecard gcr.io/broad-ctsa/scorecard
	docker push gcr.io/broad-ctsa/scorecard

run:
	docker run -i -p 5000:5000 -t scorecard

deploy:
	kubectl apply -f deployment.yaml
