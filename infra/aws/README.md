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
export STACK_ROLE_ARN=$(aws iam create-role --role-name $STACK_ROLE_NAME --assume-role-policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"Service\":[\"cloudformation.amazonaws.com\",\"ec2.amazonaws.com\"]},\"Action\":\"sts:AssumeRole\"}]}" --query "Role.Arn" --output text)
aws iam attach-role-policy --role-name $STACK_ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AdministratorAccess
```

## Creating the Cluster

Create the cluster infrastructure using CloudFormation:

```
export INSTANCE_NAME=<name for this instance of Hail>
aws cloudformation create-stack --stack-name hail-vdc --template-body file://main.yaml --role-arn $STACK_ROLE_ARN --parameters file://${INSTANCE_NAME}/parameters.json --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND --disable-rollback
```

The stack will take 15 to 20 minutes to finish creating. Once it's complete, create the instance the bootstrap scripts will be run from:

```
aws cloudformation create-stack --stack-name bootstrap --template-body file://bootstrap-vm.yaml --role-arn $STACK_ROLE_ARN --parameters "ParameterKey=CloudFormationRoleName,ParameterValue=$STACK_ROLE_NAME" --capabilities CAPABILITY_NAMED_IAM --disable-rollback
```

When the stack reaches `CREATE_COMPLETE`, the bootstrap instance is ready. Connect to the instance using EC2 Instance Connect:
```
BOOTSTRAP_INSTANCE_ID=$(aws cloudformation describe-stack-resources --stack-name bootstrap --logical-resource-id BootstrapInstance --query "StackResources[0].PhysicalResourceId" --output text)
aws ec2-instance-connect ssh --os-user root --instance-id $BOOTSTRAP_INSTANCE_ID
```

On the bootstrap instance, set up environment variables and activate the virtual environment Hail was installed in:
```
export HAIL=$HOME/hail
export NAMESPACE=default
export GITHUB_ORGANIZATION=<your GitHub organization, e.g. "hail-is">
cd $HAIL/infra/aws
source .venv/bin/activate
```

By default, EKS restricts cluster access to the principal used to create the cluster, which in this case is the CloudFormation role. You must add an access entry for any role (recommended) or user that will need access to the cluster through the console or `kubectl`:

```
export ADMIN_PRINCIPAL=<role or user ARN to be used for admin access>
aws eks create-access-entry --cluster-name vdc --principal-arn "$ADMIN_PRINCIPAL" --type STANDARD
aws eks associate-access-policy --cluster-name vdc --principal-arn "$ADMIN_PRINCIPAL" --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy --access-scope type=cluster
```

By default, EKS restricts cluster access to the principal used to create the cluster, which in this case is the CloudFormation role. You must add an access entry for any role (recommended) or user that will need access to the cluster through the console or `kubectl`:

```
export ADMIN_PRINCIPAL=<role or user ARN to be used for admin access>
aws eks create-access-entry --cluster-name vdc --principal-arn "$ADMIN_PRINCIPAL" --type STANDARD
aws eks associate-access-policy --cluster-name vdc --principal-arn "$ADMIN_PRINCIPAL" --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy --access-scope type=cluster
```

Edit `$HAIL/letsencrypt/subdomains.txt` to include just the services you plan to use in this deployment, e.g. `auth`, `batch` and `batch-driver`.

Deploy unmanaged resources by running
```
./bootstrap.sh deploy_unmanaged
```

TODO: Create the batch worker VM image. Run:
```
NAMESPACE=default $HAIL/batch/aws-create-worker-image.sh
```

Download the global-config to be used by `bootstrap.py`.
```
sudo mkdir /global-config
source $HAIL/devbin/functions.sh
download-secret global-config
sudo cp contents/* /global-config/
cd -
sudo chmod +r /global-config/*
```

Bootstrap the cluster.
```
./bootstrap.sh bootstrap $GITHUB_ORGANIZATION/hail:<BRANCH> deploy_batch
```
