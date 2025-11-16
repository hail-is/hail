#!/bin/bash
# One-time script to read existing user service accounts from the database & grant the batch user access to them.
# Should only be run once per project.

set -e
source $HAIL/devbin/functions.sh

if [ $# -eq 0 ]; then
  echo 'ERROR: Must provide a GCP project name.'
  exit 1
fi

PROJECT=$1
ADMIN_POD=$(kfind1 admin-pod)
ACCOUNTS=$(kubectl -n default exec -it $ADMIN_POD -- mysql -NBe "SELECT hail_identity FROM auth.users WHERE state != 'deleted'" | grep -Eo "(\w|-)+@$PROJECT.iam.gserviceaccount.com" | awk '{print $1}')

echo "Found the following user service accounts in $PROJECT:"
for ACCOUNT in $ACCOUNTS; do
  echo "    $ACCOUNT"
done
confirm 'This script will add the batch user as a service account user to these accounts.'

if [ $? -ne 0 ]; then
  exit 0
fi

for ACCOUNT in $ACCOUNTS; do
  gcloud iam service-accounts --project $PROJECT add-iam-policy-binding $ACCOUNT --member "serviceAccount:batch2-agent@$PROJECT.iam.gserviceaccount.com" --role "roles/iam.serviceAccountUser"
  sleep 1
done

echo 'Finished updating permissions.'
