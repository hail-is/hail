# cloud-tools
Scripts for working with Hail on Google Cloud Dataproc service from your local machine.

To get started, clone this repository:
```
$ git clone https://github.com/Nealelab/cloud-tools.git
```

If you'd like command line shortcuts to these scripts, add these lines to `~/.bash_profile` or `~/.bashrc`: 
```
alias start-cluster=< local cloud-tools directory >/start_cluster.py
alias submit-cluster=< local cloud-tools directory >/submit_cluster.py
alias connect-cluster=< local cloud-tools directory >/connect_cluster.py
alias stop-cluster=< local cloud-tools directory >/stop_cluster.py
```

To use the scripts here, you'll also need the Google Cloud SDK -- see quickstart instructions for Mac OS X [here](https://cloud.google.com/sdk/docs/quickstart-mac-os-x).

The `start_cluster.py` script references copies of the initialization scripts `init_default.py` and `init_notebook.py` that live in the publically-readable Google Storage bucket `gs://hail-common/`. You don't need copies of these initialization scripts on your local machine to use the other scripts, but they're hosted in this repository as a reference. 

Alternatively, you can use the initialization scripts with your own cluster start-up script -- though note that `init_notebook.py` depends on the Anaconda Python distribution installed on the master machine in `init_default.py`.

**REMINDER:** As always, don't forget to shut down your cluster when you're done with it! You can do this through the Google Cloud Console or on the command line with `gcloud dataproc clusters delete mycluster`.

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

This snippet starts a cluster with the defaults included in `start_cluster.py` and submits a Hail Python script as a job. 

The default initialization script installs the Anaconda Python2.7 distribution on the cluster's master machine, so you'll have access to pandas, numpy, etc. in your scripts.

While your job is running, you can monitor its progress through the SparkUI running on the cluster's master machine. To connect to the SparkUI from your local machine, use:
```
$ connect-cluster --name mycluster
```

**NOTE:** The `connect_cluster.py` script assumes that Google Chrome is installed on your local machine in the (default) location: `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`.

### Interactive Hail with Jupyter Notebook

Another way to use the Dataproc service is through a Jupyter notebook running on the cluster's master machine. An example workflow would be:
```
$ start-cluster --name mycluster --notebook
...wait for cluster to start...
$ connect-cluster --name mycluster --notebook
```
When your browser opens with the connection to the Jupyter notebook server, you should see the Google Storage home directory of the project your cluster was launched in, with all of the project's buckets listed. 

Select the bucket you'd like to work in, and you should see all of the files and directories in that bucket. You can either resume working on an existing `.ipynb` file in the bucket, or create a new notebook by selecting `Hail` from the `New` notebook drop-down in the upper-right corner.

From the notebook, you can use Hail the same way that you would in a complete job script:
```
import hail
hc = hail.HailContext()
...
```

When you save your notebooks using either `File -> Save and Checkpoint` or `command + s`, they should be saved automatically to the bucket.
