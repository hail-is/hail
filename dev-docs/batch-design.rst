============
Batch Design
============

.. sectnum::
.. contents::

********
Overview
********

Hail Batch is a multi-tenant batch job processing system. The Hail
team maintains deployments in GCP and Azure. There are also a few
deployments outside of the control of the Hail team as well as alpha
support in Terra. Hail Batch has two main use cases: (1) a batch job
processing system that executes arbitrary bash or Python code in
containerized environments that are generated using a Python client
library that handles file localization and job dependencies in a
user-friendly manner (hailtop.batch) and (2) as the backend for
running Hail Query on Batch (QoB) inside containers running Hail team
approved JVM byte code.

Typical users of hailtop.batch are looking to execute code for a
stand-alone scientific tool that can be run massively in parallel such
as across samples in a dataset and regions in a genome. Their
workloads usually consist of a single scatter layer with no
dependencies between jobs with sizes on the order of 100s to 100Ks of
jobs. The largest batch that has been processed by the Hail Batch
system is ~16 million jobs. Likewise, QoB consists of a single,
nonpreemptible driver job and subsequent sets of updates of jobs to
the directed acyclic graph (DAG) for subsequent stages of worker
jobs. There is a single job per partition within a stage. The number
of jobs within a stage can be on the order of 100K jobs. 


****************************
How the Current System Works
****************************

The Batch system is a set of services and infrastructure components
that work in concert to allow users to submit requests describing
workloads or sets of jobs to run and then executes the jobs on a set
of worker VMs. There is both a UI and a REST API for interacting with
Batch. The infrastructure required for a working Hail Batch system
consists of a Kubernetes cluster, a container registry, blob storage,
a MySQL database, and virtual machines (VMs). In this document, we describe
the purpose of each infrastructural component and how they all work in
concert to create a working Batch system. We also expand on how both
of the Batch Python web servers are implemented in detail such as
database representations, how cancellation works, how the autoscaler
works, and how billing works. Lastly, we describe what happens on the
worker VMs.



Infrastructure
==============

The Batch system consists of the following Kubernetes
services and cloud infrastructure components:

- Kubernetes Services
  - Gateway (gateway)
  - Internal Gateway (internal-gateway)
  - Auth (auth)
  - Auth Driver (auth-driver)
  - Batch Front End (batch)
  - Batch Driver (batch-driver)
- Worker VMs
- MySQL Database
- Cloud Storage
- Container Registry


Kubernetes Services
-------------------


Gateway
^^^^^^^

Gateway is a Kubernetes service and associated cloud-provider-managed
external load balancer. It is associated with a statically
known external IP Address. This is the entry point in which external
users send requests to the Batch system such as submitting batches and
getting information on their jobs. There is a an Envoy server behind
the load balancer that forwards requests to the appropriate service.


Internal Gateway
^^^^^^^^^^^^^^^^

Internal Gateway is a Kubernetes service and associated cloud-provider-managed
internal load balancer. Unlike the Gateway, the Internal
Gateway is associated with a statically known **internal** IP address
that is only accessible from virtual machines within our private
network. This endpoint is how Batch worker VMs are able to talk to the
Batch Driver Kubernetes Service directly without going through the public
internet.


Auth / Auth-Driver
^^^^^^^^^^^^^^^^^^

The Auth Kubernetes service is responsible for creating new users,
logging in existing users, authenticating requests from logged in
users, verifying developer status for accessing protected services
like a batch deployment in a developer namespace. We will soon be
changing how authentication / authorization is implemented. Currently,
for REST API requests, a user provides an authorization bearer header
with a Hail-issued token. This token is generated when users login and
has a default expiration date for 30 days. UI web requests have an
associated cookie that includes the token. The Auth Driver service is
responsible for creating new user resources such as service accounts,
secondary Kubernetes namespaces for developers, Kubernetes secrets
that store the user's active Hail authorization token and their Google
service account or Azure service principal certificates, which allows
users to access their resources required to execute jobs such as
Docker images and data stored in Google Cloud Storage or Azure Blob
Storage. When a user is deleted, their corresponding resources are
deleted as well.


Batch Front End
^^^^^^^^^^^^^^^

The Batch Front End is a Kubernetes service responsible for handling
user requests such as creating batches, updating batches, and viewing
job logs. How the Batch Front End Python service works is described in
more detail later in this document. When users submit requests to
authenticated endpoints (everything except for /healthcheck), the
Batch service sends a request to the Auth service to see if the token
submitted in the request is valid and in exchange get information
about the user. The Batch Front End can also send requests to the
Batch Driver notifying the driver that a batch has been created or
needs to be cancelled ("push notification"). The application is stateless
and 3 copies are running simultaneously. The Front End
extensively updates and queries the MySQL database to obtain the
information necessary to fulfill user requests. It also writes job
specs to cloud storage for use downstream by the worker VMs.


Batch Driver
^^^^^^^^^^^^

The Batch Driver is a Kubernetes service responsible for provisioning
worker VMs in response to demand, scheduling jobs on free worker VMs,
and cancelling jobs that no longer should be run. The Driver is
stateless, but only 1 copy can be running at a single time. This is
because our current strategy for knowing how many free cores per VM
are available requires a single process to accurately update the
number of free cores when we schedule a job on a VM. The Driver
communicates with worker VMs when it schedules or unschedules
jobs. The worker VMs then communicate back to the Driver when a worker
is ready to activate itself and start receiving work, notifying a job
has been completed, and deactivating itself when it is idle. The Batch
Driver has a second container inside the pod that is an Envoy server
responsible for maintaining TLS handshakes so as to reduce the CPU
load on the actual Python web server.


Worker VMs
----------

Worker VMs are virtual machines that are created outside of the
Kubernetes cluster. They share a network with the Kubernetes VMs, but
not with the Kubernetes pods. They are created with a default service
account that has permissions to read and write files to cloud storage
such as job specs and job logs as well as delete VMs (so it can delete
itself). Virtual machines are created with a preconfigured boot disk
image that has Docker preinstalled. Startup scripts then initialize
the worker VM, download the worker server application image from a
container registry, and then create the worker Docker container. Once
the worker container is running, it notifies the Batch Driver that it
is active and starts executing jobs.


MySQL Database
--------------

All Batch and Auth state is stored in a cloud-provider managed MySQL
database. We use SSL certificates to secure communication between
Kubernetes services and the database. Worker VMs cannot talk directly
to the database.


Cloud Storage
-------------

Users store the data they want to compute on in Cloud Storage (Google
Cloud Storage or Azure Blob Storage). All Batch created files such as
user job specs, job log files, job status files, and job resource
usage monitoring files are stored in cloud storage.


Container Registry
------------------

Container images used to execute user jobs as well as the images used
in our Kubernetes services are stored in a cloud provider managed
Container Registry (Google Artifact Registry and Azure Container
Registry).


Terraform
---------

TBD.


Bootstrapping
-------------

TBD.


Application Details
===================

Batch Lifecycle
---------------

1. A user submits a request to the Batch front end service to create a
   batch along with job specifications.
2. The Batch front end service records the batch and job information
   into a MySQL database and writes the job specifications to cloud
   storage.
3. The Batch driver notices that there is work available either
   through a push request from the Batch front end or by polling the
   state in the MySQL database and spins up worker VMs.
4. The worker VMs startup and notify the Batch driver they are active
   and have resources to run jobs.
5. The Batch driver schedules jobs to run on the active workers.
6. The worker VM downloads the job specification from cloud storage,
   downloads any input files the job needs from cloud storage, creates
   a container for the job to execute in, executes the code inside the
   container, uploads any logs and output files that have been
   generated, and then notifies the Batch driver that the job has
   completed.
7. Once all jobs have completed, the batch is set to complete in the
   database. Any callbacks that have been specified on batch
   completion are called.
8. Meanwhile, the user can find the status of their batch through the
   UI or using a Python client library to get the batch status, cancel
   the batch, list the jobs in the batch and their statuses, and wait
   for the batch or an individual job to complete. The implementation
   of the wait operation is by continuously polling the Batch Front
   End until the batch state is "complete".


Data Model
----------

The core concepts in the Batch data model are billing projects,
batches, jobs, updates, attempts, and resources.

A **billing project** is a mechanism for cost accounting, cost control, and
enabling the ability to share information about batches and jobs
across users. Each billing project has a list of authorized users and
a billing limit. Any users in the billing project can view information
about batches created in that billing project. Developers can
add/delete users in a billing project and modify billing limits. Right
now, these operations are manually done after a Batch user submits a
formal request to the Hail team. Note that the Hail billing project is
different than a GCP billing project.

A **batch** is a set of **jobs**. Each batch is associated with a
single billing project. A batch also consists of a set of
**updates**. Each update contains a distinct set of jobs. Updates are
distinct submissions of jobs to an existing batch in the system. They
are used as a way to add jobs to a batch. A batch is always created
with 0 updates and 0 total jobs. To add jobs to a batch, an update
must be created with an additional API call and the number of jobs in
the update must be known at the time of the API call. The reason for
this is because an update reserves a block of job IDs in order to
allow multiple updates to a batch to be submitted simultaneously
without the need for locking as well as for jobs within the update to
be able to reference each other before the actual job IDs are
known. Once all of the jobs for a given batch update have been
submitted, the update must be committed in order for the jobs to be
visible in the UI and processed by the batch driver.

A job can have **attempts**. An attempt is an individual execution
attempt of a job running on a worker VM. There can be multiple
attempts if a job is preempted. If a job is cancelled before it has a
chance to run, it will have zero attempts. An attempt has the
**instance** name that it ran on, the start time, and the end
time. The end time must always be greater than the start time. All
billing tracking is done at the level of an attempt as different
attempts for the same job can have different resource pricing if the
VM configurations are different (4 core worker vs 16 core worker).

Billing is tracked by **resources**. A resource is a product (example:
preemptible n1-standard-16 VM in us-central1) combined with a version
tag. Each resource has a rate that is used to compute cost when
multiplied by the usage of the resource. Resource rates are in units
that are dependent on the type of resource. For example, VM rates are
denominated in USD per core-hour. Each attempt has a set of resources
associated with it along with their usage in a resource-dependent set
of units. For example, a 1 core job has a usage value of 1000 (this
value is in mCPU). To compute the aggregate cost of a job, we sum up
all of the usages multiplied by the rates and then multiplied by the
duration the attempt has been running.


State Diagram
-------------

A job can be in one of the following states:

- Pending: 1+ parent jobs have not completed yet
- Ready: No pending parent jobs.
- Creating: Creating a VM for job private jobs.
- Running: Job is running on a worker VM.
- Success: Job completed successfully.
- Failed: Job failed.
- Cancelled: Job was cancelled either by the system, by the user, or
  because at least one of its parents failed.
- Error: Job failed due to an error in creating the container, an out
  of memory error, or a Batch bug (ex: user tries to use a nonexistent
  image).

The allowed state transitions are: Pending -> Ready Ready ->
{Creating, Running, Cancelled} Creating -> {Running, Cancelled}
Running -> {Success, Failed, Error, Cancelled}

A job's initial state depends on the states of its parent jobs. If it
has no parent jobs, its initial state is Ready.

A batch can be in one of the following states:

- completed: All jobs are in a completed state {Success, Failed,
  Error, Cancelled}
- running: At least one job is in a non-completed state {Pending,
  Ready, Running}

The batch and job states are critical for database performance and
must be indexed appropriately.


Batch Front End
---------------

The Batch Front End service (batch) is a stateless web service that
handles requests from the user. The front end exposes a REST API
interface for handling user requests such as creating a batch,
updating a batch, creating jobs in a batch, getting the status of a
batch, getting the status of a job, listing all the batches in a
billing project, and listing all of the jobs in a batch. There are
usually 3 copies of the batch front end service running at a given
time to be able to handle requests to create jobs in a batch with a
high degree of parallelism. This is necessary for batches with more
than a million jobs.


Flow for Creating and Updating Batches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following flow is used to create a new batch or update an existing
batch with a set of job specifications:

1. The client library submits a POST request to create a new batch at
   ``/api/v1alpha/batches/create``. A new entry for the batch is
   inserted into the database along with any associated tables. For
   example, if a user provides attributes (labels) on the batch, that
   information is populated into the ``batch_attributes`` table. A new
   update is also created for that batch if the request contains a
   reservation with more than 1 job. The new batch id and possibly the
   new update id are returned to the client.

2. The client library submits job specifications in 6-way parallelism
   in groups of jobs, called bunches, for the newly created batch update as a POST
   request to
   ``/api/v1alpha/batches/{batch_id}/updates/{update_id}/jobs/create``. The
   front end service creates new entries into the jobs table as well
   as associated tables such as the table that stores the attributes
   for the job.

3. The user commits the update by sending a POST request to
   ``/api/v1alpha/batches/{batch_id}/updates/{update_id}/commit``. After
   this, no additional jobs can be submitted for that update. The
   front end service executes a SQL stored procedure in the database
   that does some bookkeeping to transition these staged jobs into
   jobs the batch driver will be able to process and run.

The flow for updating an existing batch is almost identical to the one
above except step 1 submits a request to
``/api/v1alpha/batches/{batch_id}/updates/create``.

There are also two fast paths for creating and updating batches when
all jobs fit in a single HTTP request. At time of writing, our client
code uses this path when there are fewer than 1,024 jobs and the
specifications fit in fewer than 1KiB. at
``/api/v1alpha/batches/{batch_id}/create-fast`` and
``/api/v1alpha/batches/{batch_id}/update-fast``.


Listing Batches and Jobs
^^^^^^^^^^^^^^^^^^^^^^^^

To find all matching batches and jobs either via the UI or the Python
client library, a user provides a query filtering string as well as an
optional starting ID. The server then sends the next 50 records in
response and it is up to the client to send the next request with the
ID of the last record returned in the subsequent request.


Batch Driver
------------

The Batch Driver is a Kubernetes service that creates a fleet of
worker VMs in response to user workloads and has mechanisms in place
for sharing resources fairly across users. It also has many background
processes to make sure orphaned resources such as disks and VMs are
cleaned up, billing prices for resources are up to date, and
cancelling batches with more than N failures if specified by the
user. The service can be located on a preemptible machine, but we use
a non-preemptible machine to minimize downtime, especially when the
cluster is large. There can only be one driver service in existence at
any one time. There is an Envoy side car container in the batch driver
pod to handle TLS handshakes to avoid excess CPU usage of the batch
driver.


Instance Collections
^^^^^^^^^^^^^^^^^^^^

The batch driver maintains two different types of collections of
workers. There are **pools** that are multi-tenant and have a
dedicated worker type that is shared across all jobs. Pools can
support both preemptible and nonpreemptible VMs. Right now, there are
three types of machine types we support that correspond to low memory
(~1GB memory / core), standard (~4GB memory / core), and high memory
(~8GB memory / core) machines. These are correspondingly the
"highcpu", "standard", and "highmem" pools. Each pool has its own
scheduler and autoscaler. In addition, there's a single job private
instance manager that creates a worker VM per job and is used if the
worker requests a specific machine type. This is used commonly for
jobs that require more memory than a 16 core machine can provide.


Fair Share
^^^^^^^^^^

In order to avoid having one user starve other users from getting
their jobs run, we use the following fair share algorithm. We start
with the user who has the fewest cores running. We then allocate as
many cores as possible that are live in the cluster until we reach the
number of cores the next user has currently running. We then divide up
the remaining cores equally amongst the two users until we reach the
number of cores the next user has running. We repeat until we have
either exhausted all free cores in the cluster or have satisfied all
user resource requests. The query to get the number of ready cores in the fair
share algorithm is fast because we aggregate across a global table
``user_inst_coll_resources`` that has a limited number of rows
maintaining counts of the number of ready cores per instance
collection and user.


Autoscaler
^^^^^^^^^^

At a high level, the autoscaler is in charge of figuring out how many
worker VMs are required to run all of the jobs that are ready to run
without wasting resources. The simplest autoscaler takes the number of
ready cores total across all users and divides up that amount by the
number of cores per worker to get the number of instances that are
required. It then spins up a configurable number of instances each
time the autoscaler runs to avoid cloud provider API rate limits. This
approach works well for large workloads that have long running
jobs. However, the autoscaler can produce more cores than the
scheduler can keep busy with work. This happens when there are many
jobs with a short execution time.

Due to differences in resource prices across regions and extra fees
for inter-region data transfer, the autoscaler needs to be aware of
the regions a job can run in when scaling up the cluster in order to
avoid suboptimal cluster utilization or jobs not being able to be
scheduled due to a lack of resources.

The current autoscaler works by running every 15 seconds and executing
the following operations to determine the optimal number of instances
to spin up per region:

1. Get the fair share resource allocations for each user across all
   regions and figure out the share for each user out of 300 (this
   represents number of scheduling opportunities this user gets
   relative to other users).
2. For every user, sort the "Ready" jobs by regions the job can run in
   and take the first N jobs where N is equal to the user share
   computed in (1) multiplied by the autoscaler window, which is
   currently set to 2.5 minutes. The logic behind this number is it
   takes ~2.5 minutes to spin up a new instance so we only want to
   look at a small window at a time to avoid spinning up too many
   instances. It also makes this query feasible to set a limit on it
   and only look at the head of the job queue.
3. Take the union of the result sets for all of the users in (2) in
   fair share order. Do another pass over the result set where we
   assign each job a scheduling iteration which represents an estimate
   of which iteration of the scheduler that job will be scheduled in
   assuming the user's fair share.
4. Sort the result set by user fair share and the scheduling iteration
   and the regions that job can run in. Aggregate the free cores by
   regions in order in the result set. This becomes the number of free
   cores to use when computing the number of required instances and
   the possible regions the instance can be spun up in.


Scheduler
^^^^^^^^^

The scheduler finds the set of jobs to schedule by iterating through
each user in fair share order and then scheduling jobs with a "Ready"
state until the user's fair share allocation has been met. The result
set for each user is sorted by regions so that the scheduler matches
what the autoscaler is trying to provision for. The logic behind
scheduling is not very sophisticated so it is possible to have a job
get stuck if for example it requires 8 cores, but two instances are
live with 4 cores each.

Once the scheduler has assigned jobs to their respective instances,
the scheduler performs the work necessary to grab any secrets from
Kubernetes, update the job state and add an attempt in the database,
and then communicate with the worker VM to start running the
job. There must be a timeout on this scheduling attempt that is short
(1 second) in order to ensure that a delay in one job doesn't cause
the scheduler to get stuck waiting for that one job to be finished
scheduling. We wait at the end of the scheduling iteration for all
jobs to finish scheduling. If we didn't wait, then we might try and
reschedule the same job multiple times before the original operation
to schedule the job in the database completes.


Job State Updates
^^^^^^^^^^^^^^^^^

There are three main job state update operations:
- SJ: Schedule Job
- MJS: Mark job started
- MJC: Mark job completed

SJ is a database operation (stored procedure) that happens on the
driver before the job has been scheduled on the worker VM. In the
stored procedure, we check whether an attempt already exists for this
job. If it does not, we create the attempt and subtract the free cores
from the instance in the database. If it does exist, then we don't do
anything. We check the batch has not been cancelled or completed and
the instance is active before setting the job state to Running.

MJS is a database operation that is initiated by the worker VM when
the job starts running. The worker sends the start time of the attempt
along with the resources it is using. If the attempt does not exist
yet, we create the attempt and subtract the free cores from the
instance in the database. We then update the job state to Running if
it is not already and not been cancelled or completed already. We then
update the start time of the attempt to that given by the
worker. Lastly, we execute a separate database query that inserts the
appropriate resources for that attempt into the database.

MJC is a database operation that is initiated by the worker VM when
the job completes. The worker sends the start and end time of the
attempt along with the resources it is using. If the attempt does not
exist yet, we create the attempt and subtract the free cores from the
instance in the database. We then update the job state to the
appropriate completed state if it is not already and not been
cancelled or completed already. We then update the start and end times
of the attempt to that given by the worker. We then find all of the
children of the completed job and subtract the number of pending
parents by one. If the child job(s) now have no pending parents, they
are set to have a state of Ready. We also check if this is the last
job in the batch to complete. If so, we change the batch state to
completed. Lastly, we execute a separate database query that inserts
the appropriate resources for that attempt into the database.

When we are looking at overall Batch performance, we look at the
metrics of SJ and MJC rates per second for heavy workloads (ex: 1000s
of no-op true jobs). We historically scheduled at 80 jobs per second. We
endeavor to schedule much faster.


Canceller
^^^^^^^^^

The canceller consists of three background loops that cancel any
ready, running, or creating jobs in batches that have been cancelled
or the job specifically has been cancelled (ie. a parent failed). Fair
share is computed by taking the number of cancellable jobs in each
category and dividing by the total number of cancellable jobs and
multiplying by 300 jobs to cancel in each iteration with a minimum of
20 jobs per user.


Billing Updates
^^^^^^^^^^^^^^^

To provide users with real time billing and effectively enforce
billing limits, we have the worker send us the job attempts it has
running as well as the current time approximately every 1 minute. We
then update the rollup_time for each job which is guaranteed to be
greater than or equal to the start time and less than or equal to the
end time. The rollup time is then used in billing calculations to
figure out the duration the job has been running thus far.


Quota Exhaustion
^^^^^^^^^^^^^^^^

There is a mechanism in GCP by which we monitor our current quotas and
assign jobs that can be run in any region to a different region if
we've exceeded our quota.



Cloud Price Monitoring
^^^^^^^^^^^^^^^^^^^^^^

We periodically call the corresponding cloud APIs to get up to date
billing information and update the current rates of each product used
accordingly.



Database
--------

The batch database has a series of tables, triggers, and stored
procedures that are used to keep track of the state of billing
projects, batches, jobs, attempts, resources, and instances. We
previously discussed how the database operations SJ, MJS, and MJC
work.

There are three key principles in how the database is structured.

1. Any values that are dynamic should be separated from tables that
have static state. For example, to represent that a batch is
cancelled, we have a separate ``batches_cancelled`` table rather
than adding a cancelled field to the ``batches`` table.

2. Any tables with state that is updated in parallel should be
"tokenized" in order to reduce contention for updating rows. For
example, when keeping track of the number of running jobs per user
per instance collection, we'll need to update this count for every
schedule job operation. If there is only one row representing this
value, we'll end up serializing the schedule operations as each one
waits for the exclusive write lock. To avoid this, we have up to
200 rows per value we want to represent where each row has a unique
"token". This way concurrent transactions can update rows
simultaneously and the probability of serialized writes is
equivalent to the birthday problem in mathematics. Note that there
is a drawback to this approach in that queries to obtain the actual
value are more complicated to write as they include an aggregation
and the number of rows to store this in the database can make
queries slower and data more expensive to store.

Key tables have triggers on them to support billing, job state counts,
and fast cancellation which will be described in more detail below.


Billing
^^^^^^^

Billing is implemented by keeping track of the resources each attempt
uses as well as the duration of time each attempt runs for. It is
trivial to write a query to compute the cost per attempt or even per
job. However, the query speed is linear in the number of total
attempts when computing the cost for a batch by scanning over the
entire table which is a non-starter for bigger batches. Therefore, we
keep an ``aggregated_batch_resources`` table where each update to the
attempt duration timestamps or inserting a new attempt resource
updates the corresponding batch in the table. This table is
"tokenized" as described above to prevent serialization of attempt
update events. Likewise, we have similar aggregation tables for
billing projects as well as billing project by date. There are two
triggers, one on each of the ``attempts`` and ``attempt_resources``
table that perform the usage updates and insert the appropriate rows
to these billing tables every time the attempt rollup time is changed
or a new resource is inserted for an attempt. Having these aggregation
tables means we can query the cost of a billing project, billing
project by date, batch, or job by scanning at most 200 records making
this query fast enough for a UI page. The workers send the driver
periodic updates every minute with the elapsed time jobs have been
running for such that we can have "real-time billing".


Job State Tracking
^^^^^^^^^^^^^^^^^^

To quickly be able to count the number of ready jobs, ready cores,
running jobs, running cores, creating jobs, and creating cores for
computing fair share, we maintain a very small "tokenized" table that
is parameterized by user and instance collection. The values in this
table are automatically updated as a job's state is changed through
the job state diagram. The updates to the ``user_inst_coll_resources``
table happen in a trigger on the ``jobs`` table.


Cancellation
^^^^^^^^^^^^

A user can trigger a cancellation of a batch via the cancel button in
the UI or a REST request. The batch system also monitors how much has
been spent in a billing project. Once that limit has been exceeded,
all running batches in the billing project are cancelled.

Cancellation is the most complicated part of the Batch system. The
goal is to make cancellation as fast as possible such that we don't
waste resources spinning up worker VMs and running user jobs that are
ultimately going to get cancelled. Therefore, we need a way of quickly
notifying the autoscaler and scheduler to not spin up resources or
schedule jobs for batches that have been cancelled. We set a "flag" in
the database indicating the batch has been cancelled via the
``batches_cancelled`` table. This allows the query the scheduler
executes to find Ready jobs to run to not read rows for jobs in batches that
have been cancelled thereby avoiding scheduling them in the first
place. We also execute a similar query for the autoscaler. The only
place where we need to quickly know how many cores we have that are
ready and have not been cancelled is in the fair share calculation via
the ``user_inst_coll_resources`` table. To accomplish a fast update of
this table, we currently keep track of the number of **cancellable**
resources per batch in a tokenized table
``batch_inst_coll_cancellable_resources`` such as the number of
cancellable ready cores. When we execute a cancellation operation, we
quickly count the number of cancellable ready cores or other similar
values from the ``batch_inst_coll_cancellable_resources`` table and
subtract those numbers from the ``user_inst_coll_resources`` table to
have an O(1) update such that the fair share computation can quickly
adjust to the change in demand for resources.

The background canceller loops iterate through the cancelled jobs as
described above and are marked as Cancelled in the database and
handled accordingly one by one.

Once a batch has been cancelled, no subsequent updates are allowed to
the batch.


Batch Workers
-------------

Workers are Python web servers running on virtual machines. The Python
web server activates itself with the Batch driver and then accepts
requests to execute jobs. Jobs can take the form of either Docker Jobs
or JVM Jobs. The Docker Jobs are regular jobs that use a user-defined
image and the user-defined source code. JVM jobs are specially
designed for the Query on Batch (QoB) use case. The worker downloads
an approved JAR file to execute a user's query that is stored in cloud
storage. All containers the worker creates are by using `crun` and not
Docker. When the worker has not received any work to do and no jobs
are currently running, it will deactivate itself and shut itself down.



Known Issues
------------

- The current database structure serializes MJC operations because the
  table ``batches_n_jobs_in_complete_states`` has one row per batch
  and each MJC operation tries to update the same row in this
  table.
- ``commit_update`` is slow for large updates because we have to
  compute the job states by scanning the states of all of a job's
  parents.
- If a large batch has multiple distinct regions specified that are not
  interweaved, the autoscaler and scheduler can deadlock.
