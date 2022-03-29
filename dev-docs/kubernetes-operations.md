# Kubernetes Operations

## Altering a Node Pool

We will have the old node pool and the new node pool active simultaneously. We will use `cordon` and
`drain` to move all load from the old node pool to the new node pool. Then we will delete the old
node pool.

1. Add a new node pool to the cluster. You can use the UI or `gcloud`. We have two kinds of node
   pools: non-preemptible and preemptible, their names should always be non-preemptible-pool-N and
   prremptible-pool-N, respectively. When you re-create the nodepool, increment the number by
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
