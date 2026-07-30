#!/bin/bash

source ../bootstrap_utils.sh

function install_external_secrets() {
    EXTERNAL_SECRETS_ROLE=$1
    REGION=${2:-"us-east-1"}

    helm repo add external-secrets https://charts.external-secrets.io
    helm install external-secrets \
        external-secrets/external-secrets \
        -n external-secrets \
        --create-namespace \
        --set installCRDs=true

    kubectl annotate serviceaccount -n external-secrets external-secrets eks.amazonaws.com/role-arn="$EXTERNAL_SECRETS_ROLE"

    cat >external-secret-config.yaml <<EOF
apiVersion: external-secrets.io/v1
kind: ClusterSecretStore
metadata:
  name: "aws-secret-store"
spec:
  provider:
    aws:
      service: SecretsManager
      region: $REGION
      auth:
        jwt:
          serviceAccountRef:
            name: "external-secrets"
            namespace: "external-secrets"
---
apiVersion: external-secrets.io/v1
kind: ExternalSecret
metadata:
  name: "global-config-synced"
  namespace: "default"
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: "aws-secret-store"
    kind: ClusterSecretStore
  dataFrom:
    - extract:
        key: "global-config"
EOF
  kubectl apply -f external-secret-config.yaml
}

function install_load_balancer_controller() {
    ACCOUNT_ID=$1
    LOAD_BALANCER_CONTROLLER_ROLE=$2

    helm repo add eks https://aws.github.io/eks-charts
    helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
        -n kube-system \
        --set serviceAccount.name=aws-load-balancer-controller

    curl -o iam-policy.json https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.1.2/docs/install/iam_policy.json
    aws iam create-policy \
        --policy-name AWSLoadBalancerControllerIAMPolicy \
        --policy-document file://iam-policy.json
    aws iam attach-role-policy \
        --role-name "$LOAD_BALANCER_CONTROLLER_ROLE" \
        --policy-arn "arn:aws:iam::$ACCOUNT_ID:policy/AWSLoadBalancerControllerIAMPolicy"

    kubectl annotate serviceaccount -n kube-system aws-load-balancer-controller eks.amazonaws.com/role-arn="$LOAD_BALANCER_CONTROLLER_ROLE"
}

"$@"
