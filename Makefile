build:
	docker build . -t scorecard

push:
	docker tag scorecard gcr.io/broad-ctsa/scorecard
