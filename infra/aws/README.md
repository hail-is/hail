# Hail on AWS

This document is a WIP as we build out Hail infrastructure on AWS.

## Prerequisites:

- Install the AWS CLI and pass it credentials using `aws configure` or `aws login`.
- Set the default region for the AWS CLI using `aws configure`. The Hail backend is deployed in `us-east-1.`
- Check out the Hail repository and switch to the `$HAIL/infra/aws` directory.
- Create a new directory for your instance of Hail. Copy `hail-is/parameters.json` into this directory and edit the parameter values to match the desired configuration for your cluster.
- Create an IAM role for the CloudFormation stack. CloudFormation will use this role to manage all infrastructure:
```
export STACK_ROLE_NAME=hail-cloudformation-role
export STACK_ROLE_ARN=$(aws iam create-role --role-name $STACK_ROLE_NAME --assume-role-policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"Service\":\"cloudformation.amazonaws.com\"},\"Action\":\"sts:AssumeRole\"}]}" --query "Role.Arn" --output text)
aws iam attach-role-policy --role-name $STACK_ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AdministratorAccess
```

## Creating the Cluster

Create the cluster infrastructure using CloudFormation:

```
export INSTANCE_NAME=<name for this instance of Hail>
aws cloudformation create-stack --stack-name hail-vdc --template-body file://main.yaml --role-arn $STACK_ROLE_ARN --parameters file://${INSTANCE_NAME}/parameters.json --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND --disable-rollback
```

The stack will take 15 to 20 minutes to finish creating.
