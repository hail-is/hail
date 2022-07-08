# Kubernetes Operations

## Altering a Node Pool

### When managing node pools manually

We will have the old node pool and the new node pool active simultaneously. We will use `cordon` and
`drain` to move all load from the old node pool to the new node pool. Then we will delete the old
node pool.

1. Add a new node pool to the cluster. You can use the UI or `gcloud`. We have two kinds of node
   pools: non-preemptible and preemptible, their names should always be non-preemptible-pool-N and
   preemptible-pool-N, respectively. When you re-create the nodepool, increment the number by
   one. Take care to copy the taints and tags correctly.

2. Wait for the new nodepool to be ready.

3. Disable auto-scaling on the old nodepool.

4. Cordon all nodes in the old nodepool. This prevents pods from being newly scheduled on these
   nodes.

```
kubectl cordon --selector="cloud.google.com/gke-nodepool=$OLD_POOL_NAME"
```

5. Drain the nodes in the old nodepool. This moves pods from the old nodepool to the new node pool.

```
kubectl drain --ignore-daemonsets --selector="cloud.google.com/gke-nodepool=$OLD_POOL_NAME"
```

This will likely fail because the metrics-server uses some local disk to store metrics. If that is
the *only* pod listed, then you can re-run the command with `--delete-emptydir-data`. You may lose a
short period of k8s metrics. This will also impair the HorizontalPodAutoscaler.

Other pods, such as grafana and memory may also be listed here. You can use `--delete-emptydir-data`
to force them to be deleted as well. Deleting memory will cause a loss of cache for Hail Query on
Batch jobs using memory. Neither of these are catastrophic to delete.

```
kubectl drain --delete-emptydir-data --ignore-daemonsets --selector="cloud.google.com/gke-nodepool=$OLD_POOL_NAME"
```

6. The old node pool will still have nodes present. The autoscaler will, in all likelihood, not
   remove the nodes because they contain certain unmoveable kube-system pods. Instead, you can
   verify all relevant pods have moved by running the drain command again. You should see no pod
   names printed except for kube-system daemon sets. You will see all the nodes printed with the
   message "drained".

7. Delete the old node pool.

```
gcloud container node-pools delete $OLD_POOL_NAME --cluster $CLUSTER_NAME
```

### When using terraform
If using terraform to manage the node pools, we use terraform to create and delete
the pools. Assume we are replacing a pool whose terraform resource name is
`vdc_preemptible_pool`. NOTE: the following names apply to the *terraform resource*,
not the names of the node pools themselves, which should adhere to the naming
conventions outlined above and specified as terraform variables.

To complete step 1, copy the existing node pool resource
under a new name, `vdc_preemptible_pool_2`, make the desired changes to the new
resource and apply the terraform. This should not alter existing node pools.

Once draining is complete, take the following steps to remove the old node pool
and restore a clean terraform state:
1. Delete the resource `vdc_preemptible_pool` and apply. This should delete the old node pool.
2. Move the state of the new resource into the old one. For example, if in Azure, run

```
terraform state mv \
module.vdc.azurerm_kubernetes_cluster_node_pool.vdc_preemptible_pool_2 \
module.vdc.azurerm_kubernetes_cluster_node_pool.vdc_preemptible_pool
```

3. Rename `vdc_preemptible_pool_2` to `vdc_preemptible_pool`. If you try
to `terraform apply`, there should be no planned changes and the git history
should be clean.


## Troubleshooting

### Terraform Kubernetes provider dialing localhost
Occasionally, the `kubernetes` provider can initialize before fetching necessary
state (as the credentials are themselves terraform resources) and fall back to
dialing localhost. This can occur if you are switching between Hail installations
and the local mirror of the terraform state needs to be sync'd from remote storage
at the start of `terraform apply`.

As of writing, this
[remains an issue](https://github.com/hashicorp/terraform-provider-kubernetes/issues/1028)
with the kubernetes provider. A workaround to fully initialize the state is instead
of just running `terraform apply` for the entire module, to instead target just
the resources that generate the kubernetes configuration but do not themselves
rely on the kubernetes provider. Run `terraform apply -var-file=global.tfvars -target=module.vdc`
to correctly sync local terraform state, and subsequent invocations of `terraform apply`
should work as expected.
