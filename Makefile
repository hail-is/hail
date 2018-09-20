build: build-site build-letsencrypt-run

build-site:
	docker build -t site .

build-letsencrypt-run:
	docker build -t letsencrypt-run . -f Dockerfile.run

push: push-site push-letsencrypt-run

push-site: build-site
	docker tag site gcr.io/broad-ctsa/site
	docker push gcr.io/broad-ctsa/site

push-letsencrypt-run: build-letsencrypt-run
	docker tag letsencrypt-run gcr.io/broad-ctsa/letsencrypt-run
	docker push gcr.io/broad-ctsa/letsencrypt-run

run:
	kubectl apply -f service.yaml
	kubectl delete pod --ignore-not-found=true letsencrypt-run
	echo 'FIXME wait to terminate'
	kubectl create -f letsencrypt-run-pod.yaml
	echo "Waiting for letsencrypt-run to complete..."
	EC=""
	while [ "$$EC" = "" ]; do \
	  EC=`kubectl get pod -o "jsonpath={.status.containerStatuses[0].state.terminated.exitCode}" letsencrypt-run`; \
	  echo EC=$$EC; \
	  sleep 5; \
	done
	kubectl logs letsencrypt-run
	if [ "$$EC" != 0 ]; then \
	  exit $$EC; \
	fi

deploy-site:
	kubectl apply -f service.yaml
	kubectl apply -f site-deployment.yaml
