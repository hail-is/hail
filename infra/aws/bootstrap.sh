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

    kubectl -n external-secrets rollout status deployments --selector=app.kubernetes.io/instance=external-secrets
    kubectl annotate serviceaccount -n external-secrets external-secrets eks.amazonaws.com/role-arn="$EXTERNAL_SECRETS_ROLE"
    sleep 10s       # CRD creation can lag behind deployment rollout before they're discoverable

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
  name: "global-config"
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
    CLUSTER_NAME=$2
    LOAD_BALANCER_CONTROLLER_ROLE=$3

    curl -o iam-policy.json https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/main/docs/install/iam_policy.json
    aws iam create-policy \
        --policy-name AWSLoadBalancerControllerIAMPolicy \
        --policy-document file://iam-policy.json
    aws iam attach-role-policy \
        --role-name "$LOAD_BALANCER_CONTROLLER_ROLE" \
        --policy-arn "arn:aws:iam::$ACCOUNT_ID:policy/AWSLoadBalancerControllerIAMPolicy"

    cat >load-balancer-sa.yaml <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  annotations:
    eks.amazonaws.com/role-arn: "arn:aws:iam::$ACCOUNT_ID:role/$LOAD_BALANCER_CONTROLLER_ROLE"
  name: aws-load-balancer-controller
  namespace: kube-system
EOF
    kubectl apply -f load-balancer-sa.yaml

    helm repo add eks https://aws.github.io/eks-charts
    helm repo update eks
    helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
        -n kube-system \
        --set clusterName="$CLUSTER_NAME" \
        --set serviceAccount.create=false \
        --set serviceAccount.name=aws-load-balancer-controller

    kubectl -n kube-system rollout status deployments --selector=app.kubernetes.io/instance=aws-load-install_load_balancer_controller
}

"$@"
