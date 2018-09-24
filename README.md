<<<<<<< HEAD
# Hail

[![Zulip](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://hail.zulipchat.com?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

[Hail](https://hail.is) is an open-source, scalable framework for exploring and analyzing genomic data. 

The Hail project began in Fall 2015 to empower the worldwide genetics community to [harness the flood of genomes](https://www.broadinstitute.org/blog/harnessing-flood-scaling-data-science-big-genomics-era) to discover the biology of human disease. Since then, Hail has expanded to enable analysis of large-scale datasets beyond the field of genomics. 

Here are two examples of projects powered by Hail:

- The [gnomAD](http://gnomad.broadinstitute.org/) team uses Hail as its core analysis platform. gnomAD is among the most comprehensive catalogues of human genetic variation in the world, and one of the largest genetic datasets. Analysis results are shared publicly and have had sweeping impact on biomedical research and the clinical diagnosis of genetic disorders.
- The Neale Lab at the Broad Institute used Hail to perform QC and stratified association analysis of 4203 phenotypes at each of 13M variants in 361,194 individuals from the UK Biobank in about a day. Results and code are [here](http://www.nealelab.is/uk-biobank).

For genomics applications, Hail can:

 - flexibly [import and export](https://hail.is/docs/devel/methods/impex.html) to a variety of data and annotation formats, including [VCF](https://samtools.github.io/hts-specs/VCFv4.2.pdf), [BGEN](http://www.well.ox.ac.uk/~gav/bgen_format/bgen_format_v1.2.html) and [PLINK](https://www.cog-genomics.org/plink2/formats)
 - generate variant annotations like call rate, Hardy-Weinberg equilibrium p-value, and population-specific allele count; and import annotations in parallel through the [annotation database](https://hail.is/docs/stable/annotationdb.html), [VEP](https://useast.ensembl.org/info/docs/tools/vep/index.html), and [Nirvana](https://github.com/Illumina/Nirvana/wiki)
 - generate sample annotations like mean depth, imputed sex, and TiTv ratio
 - generate new annotations from existing ones as well as genotypes, and use these to filter samples, variants, and genotypes
 - find Mendelian violations in trios, prune variants in linkage disequilibrium, analyze genetic similarity between samples, and compute sample scores and variant loadings using PCA
 - perform variant, gene-burden and eQTL association analyses using linear, logistic, and linear mixed regression, and estimate heritability
 - lots more!

Hail's functionality is exposed through **[Python](https://www.python.org/)** and backed by distributed algorithms built on top of **[Apache Spark](https://spark.apache.org/docs/latest/index.html)** to efficiently analyze gigabyte-scale data on a laptop or terabyte-scale data on a cluster. 

Users can script pipelines or explore data interactively in [Jupyter notebooks](http://jupyter.org/) that combine Hail's methods, PySpark's scalable [SQL](https://spark.apache.org/docs/latest/sql-programming-guide.html) and [machine learning algorithms](https://spark.apache.org/docs/latest/ml-guide.html), and Python libraries like [pandas](http://pandas.pydata.org/)'s [scikit-learn](http://scikit-learn.org/stable/) and [Matplotlib](https://matplotlib.org/). Hail also provides a flexible domain language to express complex quality control and analysis pipelines with concise, readable code.

To learn more, you can view our talks at [Spark Summit East](https://spark-summit.org/east-2017/events/scaling-genetic-data-analysis-with-apache-spark/) and [Spark Summit West](https://spark-summit.org/2017/events/scaling-genetic-data-analysis-with-apache-spark/) (below).

[![Hail talk at Spark Summit West 2017](https://storage.googleapis.com/hail-common/hail_spark_summit_west.png)](https://www.youtube.com/watch?v=pyeQusIN5Ao&list=PLlMMtlgw6qNjROoMNTBQjAcdx53kV50cS)

### Getting Started

There are currently two versions of Hail: `0.1` (stable) and `0.2 beta` (development).
We recommend that new users install `0.2 beta`, since this version is already radically improved from `0.1`,
 the file format is stable, and the interface is nearly stable.

To get started using Hail `0.2 beta` on your own data or on [public data](https://console.cloud.google.com/storage/browser/genomics-public-data/):

- install Hail using the instructions in [Installation](https://hail.is/docs/devel/getting_started.html)
- read the [Overview](https://hail.is/docs/devel/overview.html) for a broad introduction to Hail
- follow the [Tutorials](https://hail.is/docs/devel/tutorials-landing.html) for examples of how to use Hail
- check out the [Python API](https://hail.is/docs/devel/api.html) for detailed information on the programming interface

You can download phase 3 of the [1000 Genomes dataset](http://www.internationalgenome.org/about) in Hail's native matrix table format [here](https://console.cloud.google.com/storage/browser/hail-datasets/hail-data/?project=broad-ctsa).

As we work toward a stable `0.2` release, additional improvements to the interface may require users to modify their pipelines
when updating to the latest patch. All such breaking changes will be logged [here](http://discuss.hail.is/t/log-of-breaking-changes-in-0-2-beta/454).

See the [Hail 0.1 docs](https://hail.is/docs/stable/index.html) to get started with `0.1`. The [Annotation Database](https://hail.is/docs/stable/annotationdb.html) and [gnomAD distribution](http://gnomad-beta.broadinstitute.org/downloads) are currently only directly available
for `0.1` but will be updated for `0.2` soon.

### User Support

There are many ways to get in touch with the Hail team if you need help using Hail, or if you would like to suggest improvements or features. We also love to hear from new users about how they are using Hail.

- chat with the Hail team in our [Zulip chatroom](https://hail.zulipchat.com).
- post to the [Discussion Forum](http://discuss.hail.is) for user support and feature requests, or to share your Hail-powered science 
- please report any suspected bugs to [github issues](https://github.com/hail-is/hail/issues)

Hail uses a continuous deployment approach to software development, which means we frequently add new features. We update users about changes to Hail via the Discussion Forum. We recommend creating an account on the Discussion Forum so that you can subscribe to these updates.

### Contribute

Hail is committed to open-source development. Our [Github repo](https://github.com/hail-is/hail) is publicly visible. If you'd like to contribute to the development of methods or infrastructure, please: 

- see the [For Software Developers](https://hail.is/docs/devel/getting_started_developing.html) section of the installation guide for info on compiling Hail
- chat with us about development in our [Zulip chatroom](https://hail.zulipchat.com)
- visit the [Development Forum](http://dev.hail.is) for longer-form discussions
<!--- - read [this post]() (coming soon!) for tips on submitting a successful Pull Request to our repository --->


### Hail Team

The Hail team is embedded in the [Neale lab](https://nealelab.squarespace.com/) at the [Stanley Center for Psychiatric Research](http://www.broadinstitute.org/scientific-community/science/programs/psychiatric-disease/stanley-center-psychiatric-research/stanle) of the [Broad Institute of MIT and Harvard](http://www.broadinstitute.org) and the [Analytic and Translational Genetics Unit](https://www.atgu.mgh.harvard.edu/) of [Massachusetts General Hospital](http://www.massgeneral.org/).

Contact the Hail team at <a href="mailto:hail@broadinstitute.org"><code>hail@broadinstitute.org</code></a>.

Follow Hail on Twitter <a href="https://twitter.com/hailgenetics">@hailgenetics</a>.

### Citing Hail

If you use Hail for published work, please cite the software:

 - Hail, https://github.com/hail-is/hail

##### Acknowledgements

We would like to thank <a href="https://zulipchat.com/">Zulip</a> for supporting
open-source by providing free hosting, and YourKit, LLC for generously providing
free licenses for <a href="https://www.yourkit.com/java/profiler/">YourKit Java
Profiler</a> for open-source development.

<img src="https://www.yourkit.com/images/yklogo.png" align="right" />
=======
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
>>>>>>> batch/master
