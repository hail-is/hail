# FAQ when developing Hail Batch

#### I messed up the Batch database in my dev namespace. How do I start fresh?

If you only want to delete the Batch database and leave the other databases alone,
you can submit a dev deploy using the `delete_batch_tables` job. The following
will create a dev deploy that removes your current Batch database and redeploys
Batch with a fresh one:

```bash
hailctl dev deploy -b <github_username>/hail:<your branch> -s delete_batch_tables,deploy_batch
```

If you want to start totally clean, another option is to delete your dev namespace's
database completely by deleting the underlying Kubernetes resources.
The database is a Kubernetes `StatefulSet`, with the data stored in a
persistent disk owned by a `PersistentVolumeClaim`. Deleting the `StatefulSet` will
delete the MySQL pod, but not the underlying claim/data.
So to get a totally clean slate, you must delete both resources:

```bash
kubectl -n <my_namespace> delete statefulset db
# When that's done...
kubectl -n <my_namespace> delete pvc mysql-persistent-storage-db-0
```

The next dev deploy will set up a new database:

```bash
hailctl dev deploy -b <github_username>/hail:<your branch> -s deploy_batch,add_developers
```

#### My namespace scaled down overnight. How do I get them back?

There is a Kubernetes `CronJob` that runs in the evenings that scales down
development namespaces. To scale back up, you need to use `kubectl scale`,
or you can use the devbin function `kscale`, like

```bash
kscale <your_namespace> up
```

If you want to manually scale down your namespace when not using it, run
`kscale <your_namespace> down`.
