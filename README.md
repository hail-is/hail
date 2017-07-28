# cloud-tools
This is a small collection of scripts for working with Hail on Google Cloud Dataproc service from your local machine.

To use the scripts here, you'll also need the Google Cloud SDK -- see quickstart instructions for Mac OS X [here](https://cloud.google.com/sdk/docs/quickstart-mac-os-x). (Tip: to determine if your Mac type is x86 or x86_64, type `set | grep MACHTYPE` in the terminal.) The scripts in this repository are wrappers around the functionality in the SDK.

To get started, clone this repository:
```
$ git clone https://github.com/Nealelab/cloud-tools.git
```

If you'd like command line shortcuts to these scripts, add these lines to `~/.bash_profile` or `~/.bashrc`, changing the paths to reflect the directory to which you've cloned the repository on your local machine: 
```
alias start-cluster=/path/to/cloud-tools/start_cluster.py
alias submit-cluster=/path/to/cloud-tools/submit_cluster.py
alias connect-cluster=/path/to/cloud-tools/connect_cluster.py
alias stop-cluster=/path/to/cloud-tools/stop_cluster.py
```

The `start_cluster.py` script references a copy of the initialization script `init_notebook.py` that lives in the publicly-readable Google Storage bucket `gs://hail-common/`. 

You don't need a copy of this initialization script on your local machine to use the other scripts, but it's hosted in this repository as a reference. You can also use the initialization script with your own cluster start-up script. 

**REMINDER:** As always, don't forget to shut down your cluster when you're done with it! You can do this through the Google Cloud Console, using `stop-cluster --name mycluster` after setting up the aliases above, or using the Google Cloud SDK directly with `gcloud dataproc clusters delete mycluster`.

## Example workflows

### Batch-style job submission

One way to use the Dataproc service is to write complete Python scripts that use Hail and then submit those scripts to the Dataproc cluster. Assuming you've created the command aliases above, an example of using the scripts in this repository in that manner would be:
```
$ start-cluster --name mycluster
...wait for cluster to start...
$ submit-cluster --name mycluster myhailscript.py
...Hail job output...
Job [...] finished successfully.
```

This snippet starts a cluster with the defaults included in `start_cluster.py` and submits a Hail Python script as a job. 

The notebook initialization script installs an Anaconda Python2.7 distribution on the cluster's master machine, so you'll have access to pandas, numpy, etc. in your scripts.

To setup your cluster with VEP annotation capabilities, add the flag `--vep` to the command:
```
$ start-cluster --name mycluster --vep
```

By default, `submit_cluster.py` will use the latest Hail build in `gs://hail-common/`. To use a previous build found in that bucket, specify the build hash using the `--hash` flag:
```
$ submit-cluster --name mycluster --hash some_other_hash myhailscript.py
```

You can also use custom Hail jar and zip files located in Google Storage buckets with the `--jar` and `--zip` flags:
```
$ submit-cluster --name mycluster --jar gs://mybucket/hail-all-spark.jar --zip gs://mybucket/hail-python.zip myhailscript.py
```

While your job is running, you can monitor its progress through the Spark Web UI running on the cluster's master machine. To connect to the SparkUI from your local machine, use:
```
$ connect-cluster --name mycluster
```
This will open an SSH tunnel to the master node of your cluster and open a Google Chrome browser configured to view processes running on the master node. 

Once the browser is open and you have a Hail job running, you can monitor the progress of your job by navigating to `localhost:4040` to view the Spark Web UI. Each `HailContext` has its own Web UI, so if you have multiple contexts running, they may also be found on ports `4041`, `4042`, etc.

If you'd like to troubleshoot jobs that have failed, you can view the details of jobs that are no longer active by accessing the Spark history server at `localhost:18080`.

**NOTE:** The `connect_cluster.py` script assumes that Google Chrome is installed on your local machine in the (default) location: `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`.

### Interactive Hail with Jupyter Notebook

Another way to use the Dataproc service is through a Jupyter notebook running on the cluster's master machine. By default, the `start_cluster.py` script in this repository uses the `init_notebook.py` initialization action, which sets up and starts a Jupyter server process - complete with a Hail kernel - on the master machine. 

If you'd like to use a Jupyter notebook to run Hail commands, connect to the cluster using the `connect_cluster.py` script as described above and then navigate to `localhost:8123` (by default, the Jupyter notebook process runs on port `8123` of the master node).

After navigating to `localhost:8123`, you should see the Google Storage home directory of the project your cluster was launched in, with all of the project's buckets listed.

Select the bucket you'd like to work in, and you should see all of the files and directories in that bucket. You can either resume working on an existing `.ipynb` file in the bucket, or create a new Hail notebook by selecting `Hail` from the `New` notebook drop-down in the upper-right corner.

From the notebook, you can use Hail the same way that you would in a complete job script:
```
import hail
hc = hail.HailContext()
...
```
To read or write files stored in a Google bucket outside of Hail-specific commands, use Hail's `hadoop_read()` and `hadoop_write()` helper functions. For example, to read in a file from Google storage to a pandas dataframe:
```
import hail
import pandas as pd

hc = hail.HailContext()

with hail.hadoop_read('gs://mybucket/mydata.tsv') as f:
    df = pd.read_csv(f, sep='\t')
```

When you save your notebooks using either `File -> Save and Checkpoint` or `command + s`, they'll be saved automatically to the bucket you're working in.

By default, the Jupyter notebook will use the latest Hail build found in `gs://hail-common/`. To use a previous build found in that bucket, you'll need to indicate the hash of the build you'd like to use when starting your cluster:
```
$ start-cluster --name mycluster --hash some_other_hash
```
