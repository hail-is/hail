Kubernetes [Python client](https://github.com/kubernetes-client/python/blob/master/kubernetes/README.md
 - [V1Pod](https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/V1Pod.md)
 - [create_namespaced_pod](https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/CoreV1Api.md#create_namespaced_pod)
 - [delete_namespaced_pod](https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/CoreV1Api.md#delete_namespaced_pod)
 - 

To get kubectl credentials for a GKE cluster:

$ gcloud container clusters get-credentials <cluster>

To authorize docker to push to GCR:

$ gcloud auth configure-docker

To run batch locally, using the local kube credentials:

$ docker run -i -v $HOME/.kube:/root/.kube -p 5000:5000 -t batch

On OSX, the port will be accessible on the docker-machine:

$(docker-machine ip default):5000

Get a shell in a running pod:

$ kubectl exec -it <pod> -- /bin/sh

Hit a Flask REST endpoint with Curl:

$ curl -X POST -H "Content-Type: application/json" -d <data> <url>

$ curl -X POST -H "Content-Type: application/json" -d '{"name": "batchtest", "image": "gcr.io/broad-ctsa/true"}' batch/schedule

Give default:default serviceaccount cluster-admin privileges:

$ kubectl create clusterrolebinding cluster-admin-default --clusterrole cluster-admin --serviceaccount=default:default

Run an image in a new pod:

$ kubectl run <name> --restart=Never --image <image. -- <cmd>

For example, run a shell in an new pod:

$ kubectl run -i --tty apline --image=alpine --restart=Never -- sh

Forward from a local port to a port on pod:

$ kubectl port-forward jupyter-deployment-5f54cff675-msr85 8888:8888 # <local port>:<remote port>

