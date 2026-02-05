#!/bin/bash
# One-time script to read existing user service accounts from the database and add them to the pet_service_accounts group
# Only needed once per project, as part of switching over to adding all user service accounts to the pet_service_accounts group.
# For new users, the pet_service_accounts group is automatically added to when the user is created.

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
confirm 'This script will:'
echo '  1. Add the batch user as a service account user to these accounts.'
echo '  2. Add these accounts to the pet_service_accounts group.'

if [ $? -ne 0 ]; then
  exit 0
fi

# Get organization domain from global-config
ORGANIZATION_DOMAIN=$(kubectl -n default get secret global-config -o jsonpath='{.data.organization_domain}' | base64 -d)
if [ -z "$ORGANIZATION_DOMAIN" ]; then
  echo 'ERROR: Could not find organization_domain in global-config secret.'
  exit 1
fi

GROUP_EMAIL="pet_service_accounts@${ORGANIZATION_DOMAIN}"

for ACCOUNT in $ACCOUNTS; do
  echo "Processing $ACCOUNT..."
  
  # Add to pet_service_accounts group
  if ! gcloud identity groups memberships add --group-email=$GROUP_EMAIL --member-email=$ACCOUNT --roles=MEMBER; then
    echo "  Warning: Failed to add $ACCOUNT to group (may already be a member or group not found)"
  fi
  
  sleep 1
done

echo 'Finished updating permissions and group memberships.'
