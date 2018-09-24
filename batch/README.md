Getting Started
---

Start a `minikube` k8s cluster and configure your `kubectl` to point at that k8s
cluster:

```
minikube start
```

If you get a weird minikube error, try

```
minikube delete
rm -rf ~/.minikube
brew cask reinstall minikube # or equivalent on your OS
minikube start
```

When you want to return to using a google k8s cluster, you can run this:

```
gcloud container clusters get-credentials CLUSTER_NAME
```

Set some environment variables so that docker images are placed in the
`minikube` cluster's docker registry:

```
eval $(minikube docker-env)
```

Build the batch and test image

```
make build-batch build-test
```

edit the `deployment.yaml` so that the container named `batch` has
`imagePullPolicy: Never`. This ensures that k8s does not go look for the image
in the Google Container Registry and instead uses the local image cache (which
you just updated when you ran `make build-batch build-test`).

Give way too many privileges to the default service account so that `batch` can
start new pods:

```
kubectl create clusterrolebinding \
  cluster-admin-default \
  --clusterrole cluster-admin \
  --serviceaccount=default:default
```

Create a batch service:

```
kubectl create -f deployment.yaml
```

If you ever need to shutdown the service, execute:

```
kubectl delete -f deployment.yaml
```

Look for the newly created batch pod:

```
kubectl get pods
```

And create a port forward from the k8s cluster to your local machine (this works
for clusters in GKE too):

```
kubectl port-forward POD_NAME 5000:5000
```

The former port is the local one and the latter port is the remote one (i.e. in
the k8s pod). Now you can load the conda environment for testing and run the
tests against this deployment:

```
conda env create -f environment.yaml
conda activate hail-batch
make test-local
```



---

Kubernetes [Python client](https://github.com/kubernetes-client/python/blob/master/kubernetes/README.md)
 - [V1Pod](https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/V1Pod.md)
 - [create_namespaced_pod](https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/CoreV1Api.md#create_namespaced_pod)
 - [delete_namespaced_pod](https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/CoreV1Api.md#delete_namespaced_pod)
 - 

To get kubectl credentials for a GKE cluster:

```
$ gcloud container clusters get-credentials <cluster>
```

To authorize docker to push to GCR:

```
$ gcloud auth configure-docker
```

To run batch locally, using the local kube credentials:

```
$ docker run -i -v $HOME/.kube:/root/.kube -p 5000:5000 -t batch
```

On OSX, the port will be accessible on the docker-machine:

```
$(docker-machine ip default):5000
```

Get a shell in a running pod:

```
$ kubectl exec -it <pod> -- /bin/sh
```

Hit a Flask REST endpoint with Curl:

```
$ curl -X POST -H "Content-Type: application/json" -d <data> <url>
$ curl -X POST -H "Content-Type: application/json" -d '{"name": "batchtest", "image": "gcr.io/broad-ctsa/true"}' batch/jobs/create
```

Give default:default serviceaccount cluster-admin privileges:

```
$ kubectl create clusterrolebinding cluster-admin-default --clusterrole cluster-admin --serviceaccount=default:default
```

Run an image in a new pod:

```
$ kubectl run <name> --restart=Never --image <image> -- <cmd>
```

For example, run a shell in an new pod:

```
$ kubectl run -i --tty apline --image=alpine --restart=Never -- sh
```

Forward from a local port to a port on pod:

```
$ kubectl port-forward jupyter-deployment-5f54cff675-msr85 8888:8888 # <local port>:<remote port>
```

Run container with a given hostname:

$ docker run -d --rm --name spark-m -h spark-m -p 8080:8080 -p 7077:7077 spark-m

List all containers, included stopped containers:

$ docker ps -a

Remove all stopped containers:

$ docker ps -aq --no-trunc -f status=exited | xargs docker rm

Run a docker container linked to another:

$ docker run -d --rm --cpus 0.5 --name spark-w-0 --link spark-m spark-w -c 1 -m 2g

Get IP of container:

$ docker inspect <container-id> | grep IPAddress

---

The following will set some environment variables so that future invocations of
`docker build` will make images available to the minikube cluster. This allows
you to test images without pushing them to a remote container registry.

```
eval $(minikube docker-env)
make build-batch build-test
```

NB: you must also set the `imagePullPolicy` of any `container` you `kubectl
create` to `Never` if you're using the `:latest` image tag (which is implicitly
used if no tag is specified on the image name). Otherwise, k8s will always try
to check if there is a newer version of the image. Even if `imagePullPolicy`
is set to `NotIfPresent`, k8s will still check for a newer image if you use the
`:latest` tag.
