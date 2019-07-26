
PROJECT := broad-ctsa
ZONE := us-central1-a

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

TEST_TASK_INSTANCE_NAME = cs-hack-test-task-7
test-task:
	gsutil -m cp task-startup.sh task-shutdown.sh run-task.sh run-task.py input.txt gs://hail-cseed/cs-hack/
	gsutil cp test-config.json gs://hail-cseed/cs-hack/tmp/foobar/config.json
	gcloud -q compute --project $(PROJECT) instances create $(TEST_TASK_INSTANCE_NAME) --zone=$(ZONE) --machine-type=n1-standard-1 --network=default --network-tier=PREMIUM --metadata=token=foobar,startup-script-url=gs://hail-cseed/cs-hack/task-startup.sh --no-restart-on-failure --maintenance-policy=MIGRATE --service-account=842871226259-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --image=cs-hack --image-project=$(PROJECT) --boot-disk-size=20GB --boot-disk-type=pd-ssd

copy-all:
	gsutil -m cp master-startup.sh task-startup.sh task-shutdown.sh run-task.sh run-task.py input.txt gs://hail-cseed/cs-hack/
	gsutil cp test-config.json gs://hail-cseed/cs-hack/tmp/foobar/config.json
