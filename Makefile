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

letsencrypt-run:
# start service
	kubectl apply -f service.yaml
# stop existing site deployment
	kubectl delete --ignore-not-found=true -f site-deployment.yaml
	N=
	while [[ $$N != 0 ]]; do \
	  sleep 5; \
	  N=$(kubectl get pods -l app=site --ignore-not-found=true --no-headers | wc -l | tr -d '[:space:]'); \
	  echo N=$$N; \
	done
# stop existing letsencrypt-run pod
	kubectl delete pod --ignore-not-found=true letsencrypt-run
	N=
	while [[ $$N != 0 ]]; do \
	  sleep 5; \
	  N=$(kubectl get pod --ignore-not-found=true --no-headers letsencrypt-run | wc -l | tr -d '[:space:]'); \
	  echo N=$$N; \
	done
# run letsencrypt-run pod
	kubectl create -f letsencrypt-run-pod.yaml
	echo "Waiting for letsencrypt-run to complete..."
	EC=""
	while [[ $$EC = "" ]]; do \
	  sleep 5; \
	  EC=$(kubectl get pod -o "jsonpath={.status.containerStatuses[0].state.terminated.exitCode}" letsencrypt-run); \
	  echo EC=$$EC; \
	done
	kubectl logs letsencrypt-run
	if [[ $$EC != 0 ]]; then \
	  exit $$EC; \
	fi
# cleanup
	kubectl delete pod --ignore-not-found=true letsencrypt-run

serve:
	docker run -it -p 80:80 -p 443:443 -v $$(pwd)/letsencrypt:/etc/letsencrypt site

deploy-site:
	kubectl apply -f service.yaml
	kubectl apply -f site-deployment.yaml
