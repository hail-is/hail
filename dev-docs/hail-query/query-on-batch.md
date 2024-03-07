# Query-on-Batch

In Query-on-Batch there are three components: client, driver, and workers. The client is a Python
process living either on the user's laptop or in a Batch job. The driver and workers are all JVM
processes living in Batch jobs.

We do not currently support the client process living in a job in the same batch as the driver and
workers. The driver expects total control over the batch within which it runs.

For almost all communication between its components, Query-on-Batch stores data in cloud
storage. The cloud storage bucket is always a user-owned bucket. We expect users to either use a
bucket which automatically deletes old files or to periodically delete the files we create.

## Client

The client is a Python process using the `hail` Python library. It makes "remote procedure calls,"
if you will, by creating jobs in one long-running batch. Unless a job fails (cancelling the batch
and rendering it unusable) the same batch is used for the lifetime of the Python process. The remote
procedure calls are defined by the `ActionTag` whose values must match the constant values in
`ServiceBackendAPI`. These, at time of writing nine, operations are the API between the client and
the driver. The client never speaks to the worker.

The data (whether it is IR to run or a table path from which we want to get the type) is serialized
into cloud storage and then deserialized by the driver. We do not encode the data into the "command"
of the job the way we do for normal Batch bash jobs. Serializing small operations in the "command"
of the job is a good idea that I did not have time to pursue. It saves one round trip from the
driver to cloud storage.

The result of most remote procedure calls is serialized in terms of JSON. The mapping from JSON to
Python objects is defined in backend.py and usually delegates to `_from_json` methods.

The result of the EXECUTE procedure call is potentially somewhat large (though small enough to fit
in memory on the client and the driver). We do not serialize this as JSON for size reasons (consider
that an array of objects repeats the field names). Instead, we select, for each type exactly one
EType which we implement in both Scala and Python. The implementation in Python is the
`_from_encoding` method of the Type class hierarchy.

There is exactly one place where Hail values are sent in the opposite direction from Python to
Scala: `EncodedLiteral`. Encoded literals are used to transfer large objects like a large dictionary
or a long list from Python to Scala. The value is serialized to the same EType mentioned above using
the `_to_encoding` method of the Type class hierarchy. A encoded literal is created by `hl.literal`
for any non-primitive, non-missing value.

## Driver

The entry point to the driver is the `ServiceBackendAPI` which is instantiated once per remote
procedure call. Every call corresponds to exactly one batch job. Each job is a JVM Job which is
described in more detail below under "JVM Jobs". For most of the operations, the driver quickly
executes a simple function and returns its value without starting any workers.

The EXECUTE operation is the only operation which starts workers. For EXECUTE, the driver parses the
IR, reads metadata about the datasets, plans a query, compiles "small data" code, and then executes
each partition of the query in its own job.

## Worker

The entry point to the worker is `Worker`. It simply reads the function object, reads the context
element, classloads the function, invokes the function on the context element, and serializes the
result (or an exception) to cloud storage.

Conceptually, a worker can either:

1. Complete successfully and serialize a small Hail value to cloud storage.
2. Raise an exception while executing the function and serialize that exception to cloud storage.
3. Raise an exception at any other point.

Both (2) and (3) will cause the job to appear failed in the Hail Batch UI, but if (3) happens, the
driver will fail because an output file is missing. (3) is almost certainly the result of a bug in
`Worker`.

## Backend

In terms of code, Query-on-Batch is implemented as a `Backend` which defines how to serialize
configuration, execute jobs, and deserialize results. The three key operations of `Backend` are:

- `broadcast`
- `addReference`
- `parallelizeAndComputeWithIndex`

There are many other operations like `tableType` which are best thought of as remote procedure calls
from Python to the JVM but where the procedure is executed in a Hail Batch job. This was perhaps a
mistake, but it freed the users from installing Java and empowered us to fully control the run-time
environment of the driver (for example, its core count and memory).

### `broadcast`

Broadcast explicitly serializes a value for use in jobs. In Spark, broadcasted files are
deserialized once per JVM. In Query-on-Batch, broadcasted files are just serialized along with the
code. Improving this requires Batch to support some form of shared/cached data. At time of writing,
the only caching/sharing that Batch supports is sharing of container images.

### `addReference`

A reference is a relatively large (a few gigabytes) sequence of DNA bases. Hail treats them as a
first class object that is used with the Locus datatype. In Spark, these are downloaded onto the
filesystem once per JVM. In Query-on-Batch, the bucket containing the references is mounted via
cloudfuse (gcsfuse in GCP, blobfuse in Azure). Note that gcsfuse requires the use of a Class A
Operation for each directory in a path making this more expensive than the ideal.

### `parallelizeAndComputeWithIndex`

This creates one job for each element of `collection`. Each job executes `f` on its element. If no
job raises an exception, the results are returned in the same order as `collection`. In Spark, we
implement this operation with an RDD. In Query-on-Batch, `submitAndWaitForBatch`:

1. Uploads the bytecode of the function to one object.
2. Uploads the contexts and an index thereof to another object.
3. Creates one job per context encoding in the command an index into the context array (as well as
   the length of the context lengths).

The memory and cores of the workers are controlled by `hl.init` parameters: `worker_memory` and
`worker_cores`. Hail Query workers cannot use more than one core but larger core counts effectively
make more memory available to the job.

The jobs are always added to the same batch in which the driver exists.

`submitAndWaitForBatch` polls the Batch API until all the jobs are complete. At time of writing,
Query-on-Batch does not use job groups. Instead, it assumes that a batch contains exactly one
Query-on-Batch driver (itself). The driver considers the distributed execution complete when the
number of complete jobs is one less than the number of jobs (nb: the driver itself is still
running).

## JVM Jobs

Query-on-Batch does not use normal Hail Batch jobs. Instead, it uses "JVM jobs". A JVM job is built
to enable Query-on-Batch. Every Hail Batch keeps 31 JVMs running: 1 16-core, 2 8-core, ..., and 16
1-core JVMs. Each JVM is running the `jvm-entryway`, which classloads a JAR (if not already loaded),
instantiates a given class, and invokes its `main` method.

Hail Batch maintains an allow-list of JAR locations which are only writable by the Hail team and
CI. A JAR for every main commit of Hail is uploaded to this location. In `batch/batch/worker.py`,
the main class is hard-coded to `is.hail.backend.service.Main`. `Main` dispatches to either
`is.hail.backend.service.Worker.main` or `is.hail.backend.service.ServiceBackendAPI.main`. The
latter is for driver jobs. The name is unfortunate and ought to be changed.

We keep JVMs running on the workers because a cold JVM executes Hail Query code substantially slower
than a warm JVM. A particularly glaring example is the Google Cloud Storage library: it is about an
order of magnitude slower on a cold JVM than a warm one. We are exploring the possibility of
removing JVM Jobs and replacing them with normal jobs which run a warmed and CRaC-checkpointed JVM.
