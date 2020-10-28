.. _sec-change-log:

Change Log
==========

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
