# Hail on AWS

This document is a WIP as we build out Hail infrastructure on AWS.

## Prerequisites:

- Install the AWS CLI and pass it credentials using `aws login`.
- Set the default region for the AWS CLI using `aws configure`. The Hail backend is deployed in `us-east-1.`
- Check out the Hail repository and switch to the `$HAIL/infra/aws-broad` directory.
- Edit the default parameter values in `parameters.json` to match the desired configuration for your cluster.

## Creating the Cluster

Create the cluster infrastructure using CloudFormation:

```
aws cloudformation create-stack --stack-name hail-vdc --template-body file://main.yml --parameters file://parameters.json --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND --disable-rollback
```

The stack will take 15 to 20 minutes to finish creating. Once it's complete, create the instance the bootstrap scripts will be run from:

```
aws cloudformation create-stack --stack-name bootstrap --template-body file://bootstrap-vm.yml --capabilities CAPABILITY_NAMED_IAM --disable-rollback
```

When the stack reaches `CREATE_COMPLETE`, the bootstrap instance is ready. Connect to the instance using EC2 Instance Connect:
```
BOOTSTRAP_INSTANCE_ID=$(aws cloudformation describe-stack-resources --stack-name bootstrap --logical-resource-id BootstrapInstance --query "StackResources[0].PhysicalResourceId" --output text)
aws ec2-instance-connect ssh --os-user ubuntu --instance-id $BOOTSTRAP_INSTANCE_ID
```

