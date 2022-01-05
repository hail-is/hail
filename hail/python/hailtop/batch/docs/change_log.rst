.. _sec-change-log:

Change Log
==========

**Version 0.2.79**

- The `bucket` parameter in the ``ServiceBackend`` has been deprecated. Use `remote_tmpdir` instead.

**Version 0.2.75**

- Fixed a bug introduced in 0.2.74 where large commands were not interpolated correctly
- Made resource files be represented as an explicit path in the command rather than using environment
  variables
- Fixed ``Backend.close`` to be idempotent
- Fixed ``BatchPoolExecutor`` to always cancel all batches on errors

**Version 0.2.74**

- Large job commands are now written to GCS to avoid Linux argument length and number limitations.

**Version 0.2.72**

- Made failed Python Jobs have non-zero exit codes.

**Version 0.2.71**

- Added the ability to set values for ``Job.cpu``, ``Job.memory``, ``Job.storage``, and ``Job.timeout`` to `None`

**Version 0.2.70**

- Made submitting ``PythonJob`` faster when using the ``ServiceBackend``

**Version 0.2.69**

- Added the option to specify either `remote_tmpdir` or `bucket` when using the ``ServiceBackend``

**Version 0.2.68**

- Fixed copying a directory from GCS when using the ``LocalBackend``
- Fixed writing files to GCS when the bucket name starts with a "g" or an "s"
- Fixed the error "Argument list too long" when using the ``LocalBackend``
- Fixed an error where memory is set to None when using the ``LocalBackend``

**Version 0.2.66**

- Removed the need for the ``project`` argument in ``Batch()`` unless you are creating a PythonJob
- Set the default for ``Job.memory`` to be 'standard'
- Added the `cancel_after_n_failures` option to ``Batch()``
- Fixed executing a job with ``Job.memory`` set to 'lowmem', 'standard', and 'highmem' when using the
  ``LocalBackend``
- Fixed executing a ``PythonJob`` when using the ``LocalBackend``

**Version 0.2.65**

- Added ``PythonJob``
- Added new ``Job.memory`` inputs `lowmem`, `standard`, and `highmem` corresponding to ~1Gi/core, ~4Gi/core, and ~7Gi/core respectively.
- ``Job.storage`` is now interpreted as the desired extra storage mounted at `/io` in addition to the default root filesystem `/` when
  using the ServiceBackend. The root filesystem is allocated 5Gi for all jobs except 1.25Gi for 0.25 core jobs and 2.5Gi for 0.5 core jobs.
- Changed how we bill for storage when using the ServiceBackend by decoupling storage requests from CPU and memory requests.
- Added new worker types when using the ServiceBackend and automatically select the cheapest worker type based on a job's CPU and memory requests.

**Version 0.2.58**

- Added concatenate and plink_merge functions that use tree aggregation when merging.
- BatchPoolExecutor now raises an informative error message for a variety of "system" errors, such as missing container images.

**Version 0.2.56**

- Fix ``LocalBackend.run()`` succeeding when intermediate command fails

**Version 0.2.55**

- Attempts are now sorted by attempt time in the Batch Service UI.

**Version 0.2.53**

- Implement and document ``BatchPoolExecutor``.

**Version 0.2.50**

- Add ``requester_pays_project`` as a new parameter on batches.

**Version 0.2.43**

- Add support for a user-specified, at-most-once HTTP POST callback when a Batch completes.

**Version 0.2.42**

- Fixed the documentation for job memory and storage requests to have default units in bytes.
