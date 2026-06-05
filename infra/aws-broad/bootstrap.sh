#!/bin/bash

source ../bootstrap_utils.sh

function update_cluster_auth() {
  local role_arn=$1
  cat >aws-auth.yaml <<EOF
  apiVersion: v1
  kind: ConfigMap
  metadata:
    name: aws-auth
    namespace: kube-system
  data:
    mapRoles: |
      - rolearn: ${role_arn}
        username: system:node:{{EC2PrivateDNSName}}
        groups:
          - system:bootstrappers
          - system:nodes
EOF
  kubectl apply -f aws-auth.yaml
}

"$@"
