# cloudtools

[![PyPI](https://img.shields.io/pypi/v/cloudtools.svg)]()

cloudtools is a small collection of command line tools intended to make using [Hail](https://hail.is) on clusters running in Google Cloud's Dataproc service simpler. 

These tools are written in Python and mostly function as wrappers around the `gcloud` suite of command line tools included in the Google Cloud SDK. 

## Installation

Prerequisites:
- Mac OS X
- Python 2
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/quickstart-mac-os-x)
- (Optional) Google Chrome installed in the (default) location `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`

cloudtools can be installed from the Python package index using the pip installer: `pip install cloudtools`

## Usage

All functionality in cloudtools is accessed through the `cluster` module.

There are 5 commands within the `cluster` module:
- `cluster name start [optional-args]`
- `cluster name submit [optional-args]`
- `cluster name connect [optional-args]`
- `cluster name diagnose [optional-args]`
- `cluster name stop`

where `name` is the required, user-supplied name of the Dataproc cluster.

**REMINDER:** Don't forget to shut down your cluster when you're done! You can do this using `cluster name stop`, through the Google Cloud Console, or using the Google Cloud SDK directly with `gcloud dataproc clusters delete name`.

## Examples

### Script submission

One way to use the Dataproc service is to write complete Python scripts that use Hail, and then submit those scripts to the Dataproc cluster. An example of using cloudtools to interact with Dataproc in this way would be:
```
$ cluster testcluster start -p 6
...wait for cluster to start...
$ cluster testcluster submit myhailscript.py
...Hail job output...
Job [...] finished successfully.
```
where `myhailscript.py` lives on your computer in your current working directory and looks something like:
```
from hail import *
hc = HailContext()
...
```

This snippet starts a cluster named "testcluster" with the 1 master machine, 2 worker machines (the minimum/default), and 6 additional preemptible worker machines. Then, after the cluster is started (this can take a few minutes), a Hail script is submitted to the cluster "testcluster".

### Interactive Hail with Jupyter Notebooks

Another way to use the Dataproc service is through a Jupyter notebook running on the cluster's master machine. By default, `cluster name start` sets up and starts a Jupyter server process - complete with a Hail kernel - on the master machine of the cluster. 

To use Hail in a Jupyter notebook, you'll need to have Google Chrome installed on your computer as described in the installation section above. Then, use
```
cluster testcluster connect notebook
```
to open a connection to the cluster "testcluster" through Chrome. 

A new browser will open with the address `localhost:8123` -- this is port 8123 on the cluster's master machine, which is where the Jupyter notebook server is running. You should see the Google Storage home directory of the project your cluster was launched in, with all of the project's buckets listed.

Select the bucket you'd like to work in, and you should see all of the files and directories in that bucket. You can either resume working on an existing `.ipynb` file in the bucket, or create a new Hail notebook by selecting `Hail` from the `New` notebook drop-down in the upper-right corner.

From the notebook, you can use Hail the same way that you would in a complete job script:
```
from hail import *
hc = HailContext()
...
```
To read or write files stored in a Google bucket outside of Hail-specific commands, use Hail's `hadoop_read()` and `hadoop_write()` helper functions. For example, to read in a TSV file from Google storage to a nested Python list:
```
from hail import *

hc = HailContext()

with hadoop_read('gs://mybucket/mydata.tsv') as f:
    rows = [x.strip().split('\t') for x in f.readlines()]
```

When you save your notebooks using either `File -> Save and Checkpoint` or `command + s`, they'll be saved automatically to the bucket you're working in.

### Monitoring Hail jobs

While your job is running, you can monitor its progress through the Spark Web UI running on the cluster's master machine at port 4040. To connect to the SparkUI from your local machine, use
```
cluster testcluster connect spark-ui
```
If you've attempted to start multiple Hail/Spark contexts, you may find that the web UI for a particular job is accessible through ports 4041 or 4042 instead. To connect to these ports, use
```
cluster testcluster connect spark-ui1
```
to connect to 4041, or
```
cluster testcluster connect spark-ui2
```
to connect to 4042.  

To view details on a job that has completed, you can access the Spark history server running on port 18080 with
```
cluster testcluster connect spark-history
```

### Optional command arguments

```
cluster name start --help
...
optional arguments:
  -h, --help            show this help message and exit
  --hash HASH           Hail build to use for notebook initialization.
  --spark {2.0.2,2.1.0}
                        Spark version used to build Hail.
  --version {0.1,devel}
                        Hail version to use
  --master-machine-type MASTER_MACHINE_TYPE, --master MASTER_MACHINE_TYPE, -m MASTER_MACHINE_TYPE
                        Master machine type.
  --master-boot-disk-size MASTER_BOOT_DISK_SIZE
                        Disk size of master machine (in GB).
  --num-master-local-ssds NUM_MASTER_LOCAL_SSDS
                        Number of local SSDs to attach to the master machine.
  --num-preemptible-workers NUM_PREEMPTIBLE_WORKERS, --n-pre-workers NUM_PREEMPTIBLE_WORKERS, -p NUM_PREEMPTIBLE_WORKERS
                        Number of preemptible worker machines.
  --num-worker-local-ssds NUM_WORKER_LOCAL_SSDS
                        Number of local SSDs to attach to each worker machine.
  --num-workers NUM_WORKERS, --n-workers NUM_WORKERS, -w NUM_WORKERS
                        Number of worker machines.
  --preemptible-worker-boot-disk-size PREEMPTIBLE_WORKER_BOOT_DISK_SIZE
                        Disk size of preemptible machines (in GB).
  --worker-boot-disk-size WORKER_BOOT_DISK_SIZE
                        Disk size of worker machines (in GB).
  --worker-machine-type WORKER_MACHINE_TYPE, --worker WORKER_MACHINE_TYPE
                        Worker machine type.
  --zone ZONE           Compute zone for the cluster.
  --properties PROPERTIES
                        Additional configuration properties for the cluster.
  --metadata METADATA   Comma-separated list of metadata to add:
                        KEY1=VALUE1,KEY2=VALUE2...
  --jar JAR             Hail jar to use for Jupyter notebook.
  --zip ZIP             Hail zip to use for Jupyter notebook.
  --init INIT           Comma-separated list of init scripts to run.
  --vep
  ```
  ```
  cluster name submit --help
  ...
  optional arguments:
  -h, --help            show this help message and exit
  --hash HASH           Hail build to use for notebook initialization.
  --spark {2.0.2,2.1.0}
                        Spark version used to build Hail.
  --version {0.1,devel}
                        Hail version to use.
  --jar JAR             Custom Hail jar to use.
  --zip ZIP             Custom Hail zip to use.
  --properties PROPERTIES, -p PROPERTIES
  ```
  ```
  cluster name connect --help
  ...
  positional arguments:
  name                  User-supplied name of Dataproc cluster.
  {start,submit,connect,diagnose,stop}
                        cloudtools command.
  {spark-ui,spark-ui1,spark-ui2,spark-history,notebook}

  optional arguments:
  -h, --help            show this help message and exit
  --port PORT, -p PORT  Local port to use for SSH tunnel to master node.
  --zone ZONE, -z ZONE  Compute zone for Google cluster.
  ```
  ```
  cluster name diagnose --help
  ...
  optional arguments:
  -h, --help            show this help message and exit
  --dest DEST, -d DEST  Directory for diagnose output -- must be local
  --hail-log HAIL_LOG, -l HAIL_LOG
                        Path for hail.log file
  --overwrite           Delete dest directory before adding new files
  --no-diagnose         Do not run gcloud dataproc clusters diagnose
  --compress, -z        GZIP all files
  --workers [WORKERS [WORKERS ...]]
                        Specific workers to get log files from
  --take TAKE           Only download logs from the first N workers
  ```
  ```
  cluster name stop --help
  ...
  optional arguments:
  -h, --help            show this help message and exit
  ```
