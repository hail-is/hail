#!/bin/bash

source ../bootstrap_utils.sh

function create_secrets_provider_config() {
  cat >csi.yaml <<EOF
apiVersion: secrets-store.csi.x-k8s.io/v1
kind: SecretProviderClass
metadata:
  name: aws-secrets
spec:
  provider: aws
  parameters:
    objects: |
      - objectName: "global-config"
        objectType: "secretsmanager"
EOF
  kubectl apply -f csi.yaml
}

"$@"
