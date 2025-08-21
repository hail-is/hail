# Background concepts
The following are tools and resources that are helpful to get familiar
with when hacking on Hail Batch.

- Containers
  - [High and low level runtimes](https://www.ianlewis.org/en/container-runtimes-part-1-introduction-container-r)
  - [Network namespaces](https://lwn.net/Articles/580893/)
- Kubernetes
  - [Tutorials](https://kubernetes.io/docs/tutorials/)
  - [Cluster networking](https://kubernetes.io/docs/concepts/cluster-administration/networking/)
- Load balancers / Reverse Proxies
  - [Envoy](https://www.envoyproxy.io/docs/envoy/v1.30.4/intro/arch_overview/intro/terminology)
- Terraform


# Set up

Make sure you've run the follwing:

```
make install-dev-requirements
make -C hail install-editable
hailctl auth login
```

Examine your developer configuration

```
> hailctl dev config list

  location: external   <- Your laptop is not in the cluster
  default_namespace: default   <- Pointing to production
  domain: hail.is   <- Of our GCP Broad instance
```


Look at your user configuration

```
> hailctl auth user

{
    "username": "dgoldste",
    "email": "dgoldste@broadinstitute.org",  <- deprecated
    "gsa_email": "dgoldste-cwv2i@hail-vdc.iam.gserviceaccount.com",  <- deprecated
    "hail_identity": "dgoldste-cwv2i@hail-vdc.iam.gserviceaccount.com", <- Robot GSA
    "login_id": "dgoldste@broadinstitute.org", <- User account
    "display_name": "dgoldste-cwv2i@hail-vdc.iam.gserviceaccount.com"
}
```

# Deploying

Stand up your development version of Batch

```
hailctl dev deploy -b hail-is/hail:main -s deploy_batch,add_developers,upload_query_jar
```

<<< While dev deploys are running >>>

# Examine Production namespace

Application instances are run in containers in `Pods`

```
kubectl get pods
kubectl get pods -l app=batch
```

Pods are provisioned and accessed through the following resources
1. `Deployments`
2. `StatefulSets`
3. `Services`

```
kubectl get deployments
kubectl get statefulsets
kubectl get services

kubectl describe deployment batch
```

# How is traffic routed

<<< Look at gateways.md >>>

Visit ci.hail.is/envoy-config/gateway

<<< Look at alternatives to dev deploy >>>


# Deploy a batch to your namespace per instructions


# Try load testing
1. Go to the `standard` pool of your `batch-driver` page. Set
max instances to 5
2. Submit a 10_000 job batch of 0.25-core `true` jobs
3. Go to `grafana.hail.is`, the `Performance` tab. You should see API requests,
  SQL queries, and Batch Driver CPU utilization climb
4. Go to ci.hail.is/namespaces and adjust the rate limit for the `batch-driver`
  in your namespace. If you lower it, you should see rate limiting increase
  and the load on your namespace decrease. Increasing it should increase
  throughput until the Batch Driver or your database is fully saturated.
