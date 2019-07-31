PROJECT := broad-ctsa
ZONE := us-central1-a

PY_FILES := __init__.py worker.py google_backend.py

flake8-stmp: $(PY_FILES)
	(cd .. && python3 -m flake8 google_backend)
	touch $@

pylint-stmp: $(PY_FILES)
	(cd .. && python3 -m pylint --rcfile ../../../../pylintrc google_backend --score=n)
	touch $@

.PHONY: check
check: flake8-stmp pylint-stmp

.PHONY: clean
clean:
	rm -f flake8-stmp pylint-stmp

.PHONY: cleanup-image-builder
cleanup-image-builder:
	-gcloud -q compute --project $(PROJECT) instances delete pipeline-image-builder

.PHONY: image-builder
image-builder: cleanup-image-builder
	gsutil cp image-builder-startup.sh gs://hail-common/dev/pipeline/
	gcloud -q compute --project $(PROJECT) instances create pipeline-image-builder --zone=$(ZONE) --machine-type=n1-standard-1 --network=default --network-tier=PREMIUM --metadata=startup-script-url=gs://hail-common/dev/pipeline/image-builder-startup.sh --no-restart-on-failure --maintenance-policy=MIGRATE --scopes=https://www.googleapis.com/auth/cloud-platform --image=ubuntu-1904-disco-v20190724 --image-project=ubuntu-os-cloud --boot-disk-size=20GB --boot-disk-type=pd-ssd --boot-disk-device-name=pipeline-image-builder

.PHONY: log-image-builder
log-image-builder:
	gcloud -q compute --project=$(PROJECT) instances get-serial-port-output pipeline-image-builder --zone=$(ZONE)

.PHONY: cleanup-image
cleanup-image:
	-gcloud -q compute --project $(PROJECT) images delete pipeline

.PHONY: image
image: cleanup-image
	gcloud -q compute --project $(PROJECT) images create pipeline --source-disk=pipeline-image-builder --source-disk-zone=$(ZONE)

master:
	gsutil cp image-builder-startup.sh gs://hail-common/dev/pipeline/
	gcloud -q compute --project $(PROJECT) instances create pipeline-master --zone=$(ZONE) --machine-type=n1-standard-8 --network=default --network-tier=PREMIUM --metadata=startup-script-url=gs://hail-common/dev/pipeline/master-startup.sh --no-restart-on-failure --maintenance-policy=MIGRATE --scopes=https://www.googleapis.com/auth/cloud-platform --image=pipeline --image-project=$(PROJECT) --boot-disk-size=20GB --boot-disk-type=pd-ssd

copy-all:
	gsutil -m cp master-startup.sh worker-startup.sh run-worker.sh worker.py gs://hail-common/dev/pipeline/

push: check copy-all
	git add -u
	git commit -m 'wip'
	git push origin
