# cloud-tools
Scripts for working with Hail on Google Cloud Dataproc service on your local machine.

Recommended lines to add in `~/.bash_profile` or `~/.bashrc`:
```
alias start-cluster=< cloud-tools directory >/start_cluster.py
alias submit-cluster=< cloud-tools directory >/submit_cluster.py
alias connect-cluster=< cloud-tools directory >/connect_cluster.py
```
The `start_cluster.py` script references copies of the initialization scripts `init_default.py` and `init_notebook.py` that live in the bucket `gs://hail-common/`. You don't need to download versions of these initialization scripts onto your local machine, but they're hosted in this repository as a reference.
