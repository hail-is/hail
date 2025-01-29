# Setting the `kubectl` Context

This is a useful building block in many other activities.

The hail project itself maintains two kubernetes clusters, one in GCP and one in
Azure.

If you have authenticated with `kubectl` to the appropriate cluster and `docker`
for the corresponding container registry, then you should only need to set the
current `kubectl` context by running:

```
kubectl config use-context <CONTEXT NAME>
```

The contexts can be listed with:
```
kubectl config get-contexts
```

If you are not authenticated, then you can run the following functions from
[`devbin/functions.sh`](/devbin/functions.sh):

```
# for GCP
gcpsetcluster <PROJECT>
# for Azure
azsetcluster <RESOURCE_GROUP>
```
