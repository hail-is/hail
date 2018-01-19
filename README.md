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
- `cluster start <name> [args]`
- `cluster submit <name> [args]`
- `cluster connect <name> [args]`
- `cluster diagnose <name> [args]`
- `cluster stop <name>`

where `<name>` is the required, user-supplied name of the Dataproc cluster.

**REMINDER:** Don't forget to shut down your cluster when you're done! You can do this using `cluster stop <name>`, through the Google Cloud Console, or using the Google Cloud SDK directly with `gcloud dataproc clusters delete name`.

## Examples

### Script submission

One way to use the Dataproc service is to write complete Python scripts that use Hail, and then submit those scripts to the Dataproc cluster. An example of using cloudtools to interact with Dataproc in this way would be:
```
$ cluster start testcluster -p 6
...wait for cluster to start...
$ cluster submit testcluster myhailscript.py
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

You can also pass arguments to the Hail script using the `--args` argument:
```
$ cluster submit testcluster myhailscript.py --args "arg1 arg2"
```
where `myhailscript.py` is
```
import sys
from hail import *
hc = HailContext()
print('First argument: ', sys.argv[1])
print('Second argument: ', sys.argv[2])
```
would print
```
First argument: arg1
Second argument: arg2
```

### Interactive Hail with Jupyter Notebooks

Another way to use the Dataproc service is through a Jupyter notebook running on the cluster's master machine. By default, `cluster name start` sets up and starts a Jupyter server process - complete with a Hail kernel - on the master machine of the cluster. 

To use Hail in a Jupyter notebook, you'll need to have Google Chrome installed on your computer as described in the installation section above. Then, use
```
cluster connect testcluster notebook
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
To read or write files stored in a Google bucket outside of Hail-specific commands, use Hail's `hadoop_read()` and `hadoop_write()` helper functions. For example, to read in a TSV file from Google storage to a pandas dataframe:
```
from hail import *
import pandas as pd

hc = HailContext()

with hadoop_read('gs://mybucket/mydata.tsv') as f:
    df = pd.read_table(f)
```

where pandas was included as a package to be installed in the cluster start command:
```
cluster start testcluster --pkgs pandas
```

When you save your notebooks using either `File -> Save and Checkpoint` or `command + s`, they'll be saved automatically to the bucket you're working in.

### Monitoring Hail jobs

While your job is running, you can monitor its progress through the Spark Web UI running on the cluster's master machine at port 4040. To connect to the SparkUI from your local machine, use
```
cluster connect testcluster ui
```
If you've attempted to start multiple Hail/Spark contexts, you may find that the web UI for a particular job is accessible through ports 4041 or 4042 instead. To connect to these ports, use
```
cluster connect testcluster ui1
```
to connect to 4041, or
```
cluster connect testcluster ui2
```
to connect to 4042.  

To view details on a job that has completed, you can access the Spark history server running on port 18080 with
```
cluster connect testcluster spark-history
```

### Module usage

```
$ cluster -h
usage: cluster [-h] {start,submit,connect,diagnose,stop} ...

Deploy and monitor Google Dataproc clusters to use with Hail.

positional arguments:
  {start,submit,connect,diagnose,stop}
    start               Start a Dataproc cluster configured for Hail.
    submit              Submit a Python script to a running Dataproc cluster.
    connect             Connect to a running Dataproc cluster.
    diagnose            Diagnose problems in a Dataproc cluster.
    stop                Shut down a Dataproc cluster.
    
optional arguments:
  -h, --help            show this help message and exit
```

```
$ cluster start -h
usage: cluster start [-h] [--hash HASH] [--spark {2.0.2,2.2.0}]
                     [--version {0.1,devel}]
                     [--master-machine-type MASTER_MACHINE_TYPE]
                     [--master-boot-disk-size MASTER_BOOT_DISK_SIZE]
                     [--num-master-local-ssds NUM_MASTER_LOCAL_SSDS]
                     [--num-preemptible-workers NUM_PREEMPTIBLE_WORKERS]
                     [--num-worker-local-ssds NUM_WORKER_LOCAL_SSDS]
                     [--num-workers NUM_WORKERS]
                     [--preemptible-worker-boot-disk-size PREEMPTIBLE_WORKER_BOOT_DISK_SIZE]
                     [--worker-boot-disk-size WORKER_BOOT_DISK_SIZE]
                     [--worker-machine-type WORKER_MACHINE_TYPE] [--zone ZONE]
                     [--properties PROPERTIES] [--metadata METADATA]
                     [--packages PACKAGES] [--jar JAR] [--zip ZIP]
                     [--init INIT] [--vep]
                     name

Start a Dataproc cluster configured for Hail.

positional arguments:
  name                  Cluster name.

optional arguments:
  -h, --help            show this help message and exit
  --hash HASH           Hail build to use for notebook initialization
                        (default: latest).
  --spark {2.0.2,2.2.0}
                        Spark version used to build Hail (default: 2.0.2)
  --version {0.1,devel}
                        Hail version to use (default: 0.1).
  --master-machine-type MASTER_MACHINE_TYPE, --master MASTER_MACHINE_TYPE, -m MASTER_MACHINE_TYPE
                        Master machine type (default: n1-highmem-8).
  --master-boot-disk-size MASTER_BOOT_DISK_SIZE
                        Disk size of master machine, in GB (default: 100).
  --num-master-local-ssds NUM_MASTER_LOCAL_SSDS
                        Number of local SSDs to attach to the master machine
                        (default: 0).
  --num-preemptible-workers NUM_PREEMPTIBLE_WORKERS, --n-pre-workers NUM_PREEMPTIBLE_WORKERS, -p NUM_PREEMPTIBLE_WORKERS
                        Number of preemptible worker machines (default: 0).
  --num-worker-local-ssds NUM_WORKER_LOCAL_SSDS
                        Number of local SSDs to attach to each worker machine
                        (default: 0).
  --num-workers NUM_WORKERS, --n-workers NUM_WORKERS, -w NUM_WORKERS
                        Number of worker machines (default: 2).
  --preemptible-worker-boot-disk-size PREEMPTIBLE_WORKER_BOOT_DISK_SIZE
                        Disk size of preemptible machines, in GB (default:
                        40).
  --worker-boot-disk-size WORKER_BOOT_DISK_SIZE
                        Disk size of worker machines, in GB (default: 40).
  --worker-machine-type WORKER_MACHINE_TYPE, --worker WORKER_MACHINE_TYPE
                        Worker machine type (default: n1-standard-8, or
                        n1-highmem-8 with --vep).
  --zone ZONE           Compute zone for the cluster (default: us-central1-b).
  --properties PROPERTIES
                        Additional configuration properties for the cluster
  --metadata METADATA   Comma-separated list of metadata to add:
                        KEY1=VALUE1,KEY2=VALUE2...
  --packages PACKAGES, --pkgs PACKAGES
                        Comma-separated list of Python packages to be
                        installed on the master node.
  --jar JAR             Hail jar to use for Jupyter notebook.
  --zip ZIP             Hail zip to use for Jupyter notebook.
  --init INIT           Comma-separated list of init scripts to run.
  --vep                 Configure the cluster to run VEP.
```

```
$ cluster submit -h
usage: cluster submit [-h] [--hash HASH] [--spark {2.0.2,2.2.0}]
                      [--version {0.1,devel}] [--jar JAR] [--zip ZIP]
                      [--properties PROPERTIES]
                      name script

Submit a Python script to a running Dataproc cluster.

positional arguments:
  name                  Cluster name.
  script

optional arguments:
  -h, --help            show this help message and exit
  --hash HASH           Hail build to use for notebook initialization
                        (default: latest).
  --spark {2.0.2,2.2.0}
                        Spark version used to build Hail (default: 2.0.2).
  --version {0.1,devel}
                        Hail version to use (default: 0.1).
  --jar JAR             Custom Hail jar to use.
  --zip ZIP             Custom Hail zip to use.
  --properties PROPERTIES, -p PROPERTIES
                        Extra Spark properties to set.
  --args ARGS           Quoted string of arguments to pass to the Hail script
                        being submitted.
```

```
$ cluster connect -h
usage: cluster connect [-h] [--port PORT] [--zone ZONE]
                       name
                       {notebook,nb,spark-ui,ui,spark-ui1,ui1,spark-ui2,ui2,spark-history,hist}

Connect to a running Dataproc cluster.

positional arguments:
  name                  Cluster name.
  {notebook,nb,spark-ui,ui,spark-ui1,ui1,spark-ui2,ui2,spark-history,hist}
                        Web service to launch.

optional arguments:
  -h, --help            show this help message and exit
  --port PORT, -p PORT  Local port to use for SSH tunnel to master node
                        (default: 10000).
  --zone ZONE, -z ZONE  Compute zone for Dataproc cluster (default: us-
                        central1-b).
```

```
$ cluster diagnose -h
usage: cluster diagnose [-h] --dest DEST [--hail-log HAIL_LOG] [--overwrite]
                        [--no-diagnose] [--compress]
                        [--workers [WORKERS [WORKERS ...]]] [--take TAKE]
                        name

Diagnose problems in a Dataproc cluster.

positional arguments:
  name                  Cluster name.

optional arguments:
  -h, --help            show this help message and exit
  --dest DEST, -d DEST  Directory for diagnose output -- must be local.
  --hail-log HAIL_LOG, -l HAIL_LOG
                        Path for hail.log file.
  --overwrite           Delete dest directory before adding new files.
  --no-diagnose         Do not run gcloud dataproc clusters diagnose.
  --compress, -z        GZIP all files.
  --workers [WORKERS [WORKERS ...]]
                        Specific workers to get log files from.
  --take TAKE           Only download logs from the first N workers.
```

```
$ cluster stop -h
usage: cluster stop [-h] name

Shut down a Dataproc cluster.

positional arguments:
  name        Cluster name.

optional arguments:
  -h, --help  show this help message and exit
```
