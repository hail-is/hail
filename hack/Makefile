PROJECT := broad-ctsa
ZONE := us-central1-a

PY_FILES := worker.py

flake8-stmp: $(PY_FILES)
	python3 -m flake8 worker
	touch $@

pylint-stmp: $(PY_FILES)
	python3 -m pylint --rcfile ../pylintrc worker --score=n
	touch $@

.PHONY: check
check: flake8-stmp pylint-stmp

.PHONY: clean
clean:
	rm -f flake8-stmp pylint-stmp


.PHONY: cleanup-image-builder
cleanup-image-builder:
	-gcloud -q compute --project $(PROJECT) instances delete cs-hack-image-builder

.PHONY: image-builder
image-builder: cleanup-image-builder
	gsutil cp image-builder-startup.sh gs://hail-cseed/cs-hack/
	gcloud -q compute --project $(PROJECT) instances create cs-hack-image-builder --zone=$(ZONE) --machine-type=n1-standard-1 --network=default --network-tier=PREMIUM --metadata=startup-script-url=gs://hail-cseed/cs-hack/image-builder-startup.sh --no-restart-on-failure --maintenance-policy=MIGRATE --service-account=842871226259-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --image=ubuntu-1904-disco-v20190724 --image-project=ubuntu-os-cloud --boot-disk-size=20GB --boot-disk-type=pd-ssd --boot-disk-device-name=cs-hack-image-builder

.PHONY: log-image-builder
log-image-builder:
	gcloud -q compute --project=$(PROJECT) instances get-serial-port-output cs-hack-image-builder --zone=$(ZONE)

.PHONY: cleanup-image
cleanup-image:
	-gcloud -q compute --project $(PROJECT) images delete cs-hack

.PHONY: image
image: cleanup-image
	gcloud -q compute --project $(PROJECT) images create cs-hack --source-disk=cs-hack-image-builder --source-disk-zone=$(ZONE)

master:
	gsutil -m cp master-startup.sh gs://hail-cseed/cs-hack/
	gcloud -q compute --project $(PROJECT) instances create cs-hack-master --zone=$(ZONE) --machine-type=n1-standard-8 --network=default --network-tier=PREMIUM --metadata=startup-script-url=gs://hail-cseed/cs-hack/master-startup.sh --no-restart-on-failure --maintenance-policy=MIGRATE --service-account=842871226259-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --image=cs-hack --image-project=$(PROJECT) --boot-disk-size=20GB --boot-disk-type=pd-ssd

copy-all:
	gsutil -m cp master-startup.sh worker-startup.sh run-worker.sh worker.py input.txt gs://hail-cseed/cs-hack/

check-all: check
	make -C ../hail/python check

push: check-all copy-all
	git add -u
	git commit -m 'wip'
	git push origin
