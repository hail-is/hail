# cloud-tools
Scripts for working with Hail on Google Cloud Dataproc service from your local machine.

To get started, clone this repository:
```
$ git clone https://github.com/Nealelab/cloud-tools.git`
```

If you'd like command line shortcuts to these scripts, add these lines to `~/.bash_profile` or `~/.bashrc`: 
```
alias start-cluster=< local cloud-tools directory >/start_cluster.py
alias submit-cluster=< local cloud-tools directory >/submit_cluster.py
alias connect-cluster=< local cloud-tools directory >/connect_cluster.py
```
The `start_cluster.py` script references copies of the initialization scripts `init_default.py` and `init_notebook.py` that live in the bucket `gs://hail-common/`. You don't need copies of these initialization scripts on your local machine to use the other scripts, but they're hosted in this repository as a reference.

## Example workflows

### Batch-style job submission:

One way to use the Dataproc service is to write complete Python scripts that use Hail and then submit those scripts to the Dataproc cluster. Assuming you've created the command aliases above, an example of using the scripts in this repository in that manner would be:
```
$ start-cluster --name mycluster
...wait for cluster to start...
$ submit-cluster --name mycluster myhailscript.py
...Hail job output...
Job [...] finished successfully.
```

This snippet starts a cluster with the defaults included in `start_cluster.py` and submits a Hail Python script as a job. While your job is running, you can monitor its progress through the SparkUI.

To connect to the SparkUI (which is running on the master machine in your cluster) from your local machine, use:
```
$ connect-cluster --name mycluster
```

### Interactive Hail with Jupyter Notebook

Another way to use the Dataproc service is through a Jupyter notebook running on the cluster's master machine. An example workflow would be:
```
$ start-cluster --name mycluster --notebook
...
