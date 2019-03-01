#!/bin/bash

set -ex

${CONDA_BINARY:=conda}

CLUSTER_NAME=${HAIL_CLUSTER_NAME:-vdc}
echo "configuring environment for ${CLUSTER_NAME} (override by setting HAIL_CLUSTER_NAME)"

PLATFORM="${HAIL_PLATFORM:-${OSTYPE}}"

case "$PLATFORM" in
    darwin*)
        install-docker() {
            brew cask install docker
            open /Applications/Docker.app  # opening Docker.app seems necessary to put docker on $PATH
        }
        install-conda() {
            tmpfile=$(mktemp)
            curl -L https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh > $tmpfile
            bash $tmpfile
        }
        install-gcloud() {
            brew cask install gcloud
        }
        ;;
    linux*)
        install-docker() {
            echo "installing docker on $PLATFORM is unsupported, please manually install: https://docs.docker.com/install/linux/docker-ce/ubuntu"
            exit 1
        }
        install-conda() {
            tmpfile=$(mktemp)
            curl -L https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh > $tmpfile
            bash $tmpfile
        }
        install-gcloud() {
            echo "installing gcloud on $PLATFORM is unsupported, please manually install: https://cloud.google.com/sdk/install"
            exit 1
        }
        ;;
    *)
        echo "unsupported platform $PLATFORM if you think this is the wrong platform, explicitly set $HAIL_PLATFORM"
        ;;
esac

docker version || install-docker
${CONDA_BINARY} -V || install-conda
gcloud -v || install-gcloud

gcloud auth login

PROJECT_NAME=$(gcloud config get-value project)
if [[ "$PROJECT_NAME" == "(unset)" ]]
then
    echo "no project configured, will set config to the default project: broad-ctsa"
    gcloud config set project broad-ctsa
    PROJECT_NAME="broad-ctsa"
fi

if [[ "$(gcloud config get-value compute/region)" == "(unset)" ]]
then
    echo "no compute/region configured, will set config to the default region: us-central"
    gcloud config set compute/region us-central
fi

kubectl version -c || gcloud components install kubectl
kubectl version || gcloud container clusters get-credentials $CLUSTER_NAME --region us-central1-a

for project in $(cat projects.yaml | grep '^- project: ' | sed 's/^- project: //')
do
    if [[ -e $project/environment.yml ]]
    then
        ${CONDA_BINARY} env create -f $project/environment.yml || ${CONDA_BINARY} env update -f $project/environment.yml
    fi
done

SVC_ACCT_NAME=${HAIL_SVC_ACCT_NAME:-$(whoami)-gke}
SVC_ACCT_FULL_NAME=${SVC_ACCT_NAME}@${PROJECT_NAME}.iam.gserviceaccount.com
KEY_FILE=~/.hail-dev/gke/svc-acct/${SVC_ACCT_NAME}.json
echo "Configuring service account with name ${SVC_ACCT_FULL_NAME} with key file stored in ${KEY_FILE}"
if [ ! -e "${KEY_FILE}" ]
then
    echo "${KEY_FILE} not found, will attempt to create a key file (and user if necessary)"
    gcloud iam service-accounts describe ${SVC_ACCT_FULL_NAME} || gcloud iam service-accounts create ${SVC_ACCT_NAME}
    gcloud projects add-iam-policy-binding \
           ${PROJECT_NAME} \
           --member "serviceAccount:${SVC_ACCT_FULL_NAME}" \
           --role "roles/owner"
    mkdir -p $(dirname ${KEY_FILE})
    gcloud iam service-accounts keys create \
           ${KEY_FILE} \
           --iam-account ${SVC_ACCT_FULL_NAME}
    sed -i '' '/^export GOOGLE_APPLICATION_CREDENTIALS=.*$/d' ~/.profile
    echo export GOOGLE_APPLICATION_CREDENTIALS="${KEY_FILE}" >> ~/.profile
    echo "please run source ~/.profile or start a new shell"
fi
