# cloud-tools
Scripts for working with Hail on Google Cloud Dataproc service from your local machine.

To get started, clone this repository:
`git clone https://github.com/Nealelab/cloud-tools.git`

Recommended lines to add in `~/.bash_profile` or `~/.bashrc` if you'd like shortcuts to these scripts:
```
alias start-cluster=< local cloud-tools directory >/start_cluster.py
alias submit-cluster=< local cloud-tools directory >/submit_cluster.py
alias connect-cluster=< local cloud-tools directory >/connect_cluster.py
```
The `start_cluster.py` script references copies of the initialization scripts `init_default.py` and `init_notebook.py` that live in the bucket `gs://hail-common/`. You don't need copies of these initialization scripts on your local machine to use the other scripts, but they're hosted in this repository as a reference.
