# Hail

This document gives an overview of the Hail architecture and source
code.

## Background

Hail source code is stored in a monolithic repository (monorepo) on
GitHub at hail-is/hail.  $HAIL will denote the repository root below.
Hail is open source and developed in the open.

Hail is written in Python, Java/Scala (both JVM langauges), and C/C++.

Hail has two user-facing components: Hail Query and Hail Batch.  Both
provide Python interfaces.  Hail Query is for distributed, out-of-core
manipulation and analysis of tabular and genomic data.  Hail Batch is
for the execution of graphs containers.

The Hail client libraries are deployed in the Python Package Index
(PyPI) hail package.  The Hail package exposes two Python modules:
`hail` and `hailtop`.  `hail` provides the Hail Query interface.
`hailtop.batch` provides the batch interface.  `hailtop` contains
various other infrastructure submodules (see below).

The hail package also contains the `hailctl` command line utility that
provides a number of features, including managing Google Dataproc (a
hosted Spark service), Hail service authorization, Hail configuration,
interacting with the Hail developer tools, and managing the Hail Batch
service.  `hailctl` is implemented in the `hailtop.hailctl` submodule.

Users can run Hail on their own system, or on the Hail service that is
operated by the Hail team.  The Hail service is implemented by a
collection of microservices that are deployed on Kubernetes (K8s) and
Google Cloud Platform (GCP).  We refer to K8s and GCP as our Virtual
Datacenter (VDC).

## Hail Query

The `hail` module is implemented in $HAIL/hail/python/hail, this is
the Hail Query Python interface.  The Hail Query interface is lazy:
executing a pipeline builds an intermediate representation
representing the query.  The IR is implemented in the `hail.ir`
submodule.  When a query is ready to be executed, it is sent to a
backend, implemented in `hail.backend`.  There are three backends:
SparkBackend, LocalBackend and ServiceBackend.  At the time of
writing, only SparkBackend is complete, the other two are works in
progress.

The Spark backend works as follows.  The IR is serialized and sent to
a JVM child process via [py4j](https://www.py4j.org/).  The entrypoint
for this is the Scala class is.hail.backend.spark.SparkBackend.  The
SparkBackend parses the IR, runs a query optimizer on it, generates
custom JVM bytecode for the query which is called from a Spark
computational graph, which it then submits to Spark for execution.
The result is then returned via py4j back to Python and the user.

The ServiceBackend is structured differently.  The IR is again
serialized, but is written to cloud storage and a job is submitted
to Hail Batch to execute the aforementioned IR.

The local backend is similar to the Spark backend, except the
execution is performed on the local machine in the JVM instead of
being submitted to Spark.

## Hail Batch

A batch is a graph of jobs with dependencies.  A job includes
configuration for a container to execute, including image, inputs,
outputs, command line, environment variables, etc.  The Hail Batch
Python interface provides and high-level interface for constructing
batches of jobs.  A batch can then be submitted to a backend for
execution.  There are two backends: LocalBackend and ServiceBackend.

The local backend executes the batch on the local machine.

The service backend serializes the batch as JSON and submits it to the
batch microservice at https://batch.hail.is/.  The Batch service then
executes the graph on a fleet of GCP virtual machines called workers.

## Source Code Organization

Here are some key source code locations:

Hail Query:

* $HAIL/hail/python/hail: the Hail Query Python interface
* $HAIL/hail/src: the Hail Query Java/Scala code

Hail Batch:

* $HAIL/batch: the batch service
* $HAIL/benchmark: Hail Query benchmarking tools
* $HAIL/hail/python/hailtop/batch: Python Batch interface
* $HAIL/hail/python/hailtop/batch_client: low-level Batch client library

Services (see below for descriptions):

* $HAIL/auth
* $HAIL/batch
* $HAIL/ci
* $HAIL/gateway
* $HAIL/internal-gateway
* $HAIL/site

Libraries for services:

* $HAIL/gear: gear services library
* $HAIL/hail/python/hailtop/aiogoogle: asyncio Google client libraries
* $HAIL/hail/python/hailtop/auth: user authorization library
* $HAIL/hail/python/hailtop/config: user and deployment configuration library
* $HAIL/hail/python/hailtop/tls.py: TLS utilities for services
* $HAIL/hail/python/hailtop/utils: various
* $HAIL/tls: for TLS configuration and deployment
* $HAIL/web_common: service HTML templates

Other:

* $HAIL/blog: blog configuration
* $HAIL/build.yaml: the Hail build, test and deployment configuration
* $HAIL/datasets: ETL code for the Hail Query Datasets API
* $HAIL/docker: base docker images used by services
* $HAIL/hail/python/hailtop/hailctl: implementation of the hailctl command-line tool
* $HAIL/ukbb-rg: UKBB genetic correlation browser configuration, available at https://ukbb-rg.hail.is/

## Hail Query Java/Scala Code Organization

This section is not complete.

* is.hail.annotation: For historical reasons, a Hail Query runtime
  value (e.g. an int, array, string, etc.) is called an annotation.
  In the JVM, there are two representations of runtime values: as JVM
  objects or as a pointer to explicitly managed memory off the Java
  heap called a region value.  Annotation also sometimes refer to just
  the JVM object representation.  Explicitly managed off-(Java-)heap
  values are also referred to as "unsafe".

* is.hail.asm4s: The Hail Query optimizer generates JVM bytecode to
  implement queries.  asm4s is a high-level Scala interface for
  generating JVM bytecode.

* is.hail.lir: lir is a low-level intermediate representation (IR) for
  JVM bytecode.  The high-level asm4s interface is implemented in
  terms of lir.  lir can generate raw JVM bytecode that can be loaded
  into the interpreter and invoked via reflection.

* is.hail.types: Hail Query has several different notions of types.
  For values, there are three kinds of type: virtual, physical and
  encoded types.  Virtual types are usual-visible types like int,
  array and string.  Physical types are implementations of virtual
  types.  For example, arrays might be stored densely or sparsely.
  Encoded types specify how to (de)serialize types for reading and
  writing.  There also higher-level types for Tables, MatrixTables and
  BlockMatrices.

* is.hail.expr: expr is a large, disorganized package that contains
  the Hail Query IR, query optimizer and IR code generator.

* is.hail.io.fs: fs contains an abstract filesystem interface, FS, and
  implementations for Google Storage and Hadoop.  We need to implement
  one for local UNIX filesystems.

* is.hail.io: This includes support for reading and writing a variety
  of file formats.

* is.hail.services: This package is related to the implementation of
  services.  This includes a Java/Scala client for Hail Batch, the
  shuffle service client and server, and support for TLS,
  authentication, deployment configuration, etc.

* is.hail.rvd: The fundamental abstraction in Spark is the resilient
  distributed dataset (RDD).  When Hail generates Spark pipelines to
  implement queries, it generates an RDD of off-(Java-)heap region
  values.  An RVD is a Region Value Dataset, an abstraction for a RDD
  of region values.

* is.hail.backend: each Python backend (Spark, local, service) has a
  corresponding backend in Scala

## Microservice Overview

The Hail service is implemented as a collection of microservices (for
the list, see below).  The services are implemented in Python (mostly)
and Java/Scala (or a mix, using py4j).  The code for the services can
be found at the top-level, e.g. the batch and batch-driver
implementation can be found in $HAIL/batch.  For publicly accessible
services, they are available at https://<service>.hail.is/, e.g. the
batch service is available at https://batch.hail.is/.

Python services are implemented using Python asyncio,
[aiohttp](https://docs.aiohttp.org/en/stable/) and Jinja2 for HTML
templating.

Some services rely on 3rd party services.  Those include:

* ci depends on GitHub

* batch, ci and auth depend on K8s

* batch depends on K8s and GCP

* site depends (client-side) on Algolia for search

Services store state in a managed MySQL Google CloudSQL instance.

There is a collection of libraries to facilitate service development:

* `hailtop.aiogoogle`: asyncio client libraries for Google services.
  There are currently libraries for GCE, GCR, IAM, Stackdriver Logging
  and (in progress) Google Storage

* `hailtop.config`: to manage user and deployment configuration

* `hailtop.auth`: to manage user authorization tokens

* `hailtop.utils`: various

* `gear`: verifying user credentials, CSRF protection, a database
  interface, logging and user session management

* `web_common`: common HTML templates for the services Web UI (except
  site)

* `tls`: infrastructure for encrypting internal communication between
  services

## List of Microservices

* auth and auth-driver: for user authorization, authorization token
  verification and account management

* batch and batch-driver: the Hail Batch service

* ci: We've implemented our own continuous integration and continuous
  deployed (CI/CD) system. It also hosts a developer status board
  at https://ci.hail.is/me.

* gateway: gateway is an nginx reverse proxy that terminates TLS
  connections and forwards requests to services in K8s.  It is
  possible to run multiple copies of the Hail services in K8s, each
  set in a separate K8s namespace.  gateway forwards requests to the
  K8s service in the appropriate namespace.

* internal-gateway: batch workers run on GCE VMs, not in K8s.  The
  internal-gateway is an nginx reverse proxy that terminates
  connections from the Google Virtual Private Cloud (VPC) network and
  connections to the services in K8s.

* site: site implements the main Hail website https://hail.is/
  including the landing page and Hail Query and Hail Batch
  documentation.

There are two types of services: managed and unmanaged.
CI handles deployment of managed services, while unmanaged services
are deployed by hand using their respective Makefiles. The
unmanaged services are gateway and internal-gateway.
