# ksync

https://github.com/ksync/ksync

## Installation

```
curl https://ksync.github.io/gimme-that/gimme.sh | bash
```

## Setting it up on hail-vdc

This only needs to be done once on a cluster.

1. First make sure you have a cluster admin role

```
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: is-cluster-admin
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-admin
subjects:
- apiGroup: rbac.authorization.k8s.io
  kind: User
  name: <user_email>
```

Apply this config to the cluster.

2. Run ksync init

```
ksync init --context gke_hail-vdc_us-central1-a_vdc
```

3. Add a toleration to the ksync daemon set that has been created in the kube-system namespace

```
kubectl -n kube-system patch daemonset ksync --type merge -p '
{
   "spec": {
      "template": {
         "spec": {
            "tolerations": [
               {
                  "effect": "NoSchedule",
                  "key": "preemptible",
                  "operator": "Equal",
                  "value": "true"
               }
            ]
         }
      }
   }
}
'
```

4. Check the ksync pods are running

```
kubectl -n kube-system get pods -l app=ksync
```

## Setting it up locally

1. Run ksync watch in a new terminal window

```
ksync watch
```

2. Make sure app is deployed in the namespace you want (dev deploy)

3. Create a spec in ~/.ksync/ksync.yaml using the create operation

```
ksync create --local-read-only -l app=auth --name <NAMESPACE>-<APP> -n jigold $(pwd)/<APP>/ /usr/local/lib/python3.9/dist-packages/<APP>/
```

4. Use ksync get to make sure the pods are being watched

```
ksync get
```

5. When you're done syncing, kill the process with ksync watch

6. To remove a spec to watch, run the delete command

```
ksync delete <NAMESPACE>-<APP>
```

## Notes

- ksync is updating all pods specified in ~/.ksync/ksync.yaml
when it is running (`ksync watch`)
