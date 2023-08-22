# Container runtime

Containers in batch are run using the [crun](https://github.com/containers/crun) container runtime.
`crun` is a low-level container runtime like `runc` (what Docker uses) which implements the
Open Container Initiative (OCI) specification for running containers given an image's filesystem and a
[runtime configuration](https://github.com/opencontainers/runtime-spec/blob/master/config.md). The JSON
configuration specifies, among other things, the linux namespaces and cgroups under which to run the container
and the user command to run.

All images run on a worker are preprocessed by extracting their root filesystem into `/host/rootfs/` and
storing any additional image configuration like environment variables and users in memory in the worker
process. These root filesystems are immutable and job containers cannot write to them. All directories and
files relating to a user's job except for the underlying rootfs are stored under the job's scratch directory.
The scratch directory contains directories for each container in the job (input, main, output) and an `io`
directory that is mounted into each container. Each container directory contains
- The upper, merged, and work directories for the overlay filesystem used in the container. For
    a great explanation of how overlayfs works, see
    [here](https://jvns.ca/blog/2019/11/18/how-containers-work--overlayfs/).
- Any volumes specified in the user's image that are mounted into the container
- The container's `config.json` that the worker creates and passes to `crun`.

Batch uses [xfs_quota](https://man7.org/linux/man-pages/man8/xfs_quota.8.html) to enforce storage
limits for jobs. Each job receives its own XFS project rooted at the scratch directory. Any writes from
the main, input and output containers into their root filesystems contribute to the overall job storage quota.
Storage in `/io` is subject to the user's quota *unless* `/io` is mounted from an external disk.

Below is the layout of job's scratch directory on the worker. NOTE: Since the underlying image/root filesystem
is not stored per-job, it does not contribute toward a job's storage quota.

```
scratch/
├─ io/ (potentially mounted from an external disk)
├─ input/
│  ├─ rootfs_overlay/
│  │  ├─ upperdir/ (writeable layer)
│  │  ├─ merged/ (what the container sees as its root)
│  │  │  ├─ bin/ (from the overlay's lowerdir [the image's rootfs])
│  │  │  ├─ etc/ (from the overlay's lowerdir [the image's rootfs])
│  │  │  ├─ ...
│  │  │  ├─ io/ (bind mount)
│  │  ├─ workdir/
│  ├─ volumes/
│  ├─ config.json
├─ main/
│  ├─ rootfs_overlay/
│  │  ├─ upperdir/ (writeable layer)
│  │  ├─ merged/ (what crun/the container sees as its root)
│  │  │  ├─ bin/ (from the overlay's lowerdir [the image's rootfs])
│  │  │  ├─ etc/ (from the overlay's lowerdir [the image's rootfs])
│  │  │  ├─ ...
│  │  │  ├─ io/ (bind mount)
│  │  │  ├─ image/specified/volume/ (bind mount from volumes/)
│  │  ├─ workdir/
│  ├─ volumes/
│  │  ├─ image/specified/volume/
│  ├─ config.json
├─ output/
│  ├─ ...
```
