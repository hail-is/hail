.. role:: python(code)
   :language: python
   :class: highlight

.. role:: bash(code)
   :language: bash
   :class: highlight

.. _sec-configuration-reference:

Configuration Reference
=======================

Configuration variables can be set for Hail Query by:

#. passing them as keyword arguments to :func:`.init`,
#. running a command of the form :bash:`hailctl config set <VARIABLE_NAME> <VARIABLE_VALUE>` from the command line, or
#. setting them as shell environment variables by running a command of the form
   :bash:`export <VARIABLE_NAME>=<VARIABLE_VALUE>` in a terminal, which will set the variable for the current terminal
   session.

Each method for setting configuration variables listed above overrides variables set by any and all methods below it.
For example, setting a configuration variable by passing it to :func:`.init` will override any values set for the
variable using either :bash:`hailctl` or shell environment variables.

.. warning::
    Some environment variables are shared between Hail Query and Hail Batch. Setting one of these variables via
    :func:`.init`, :bash:`hailctl`, or environment variables will affect both Query and Batch. However, when
    instantiating a class specific to one of the two, passing configuration to that class will not affect the other.
    For example, if one value for :python:`gcs_bucket_allow_list` is passed to :func:`.init`, a different value
    may be passed to the constructor for Batch's :python:`ServiceBackend`, which will only affect that instance of the
    class (which can only be used within Batch), and won't affect Query.

Supported Configuration Variables
---------------------------------

.. list-table:: GCS Bucket Allowlist
    :widths: 50 50

    * - Keyword Argument Name
      - :python:`gcs_bucket_allow_list`
    * - Keyword Argument Format
      - :python:`["bucket1", "bucket2"]`
    * - :bash:`hailctl` Variable Name
      - :bash:`gcs/bucket_allow_list`
    * - Environment Variable Name
      - :bash:`HAIL_GCS_BUCKET_ALLOW_LIST`
    * - :bash:`hailctl` and Environment Variable Format
      - :bash:`bucket1,bucket2`
    * - Effect
      - Prevents Hail Query from erroring if the default storage policy for any of the given locations is to use cold storage.
    * - Shared between Query and Batch
      - Yes
