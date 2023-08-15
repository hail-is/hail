.. _sec-docker-resources:

================
Docker Resources
================

What is Docker?
---------------
Docker is a tool for packaging up operating systems, scripts, and environments in order to
be able to run the same code regardless of what machine the code is executing on. This packaged
code is called an image. There are three parts to Docker: a mechanism for building images,
an image repository called Docker Hub, and a way to execute code in an image
called a container. For using Batch effectively, we're only going to focus on building images.

Installation
------------

You can install Docker by following the instructions for either `Macs <https://docs.docker.com/docker-for-mac/install/>`__
or for `Linux <https://docs.docker.com/install/linux/docker-ce/ubuntu/>`__.


Creating a Dockerfile
---------------------

A Dockerfile contains the instructions for creating an image and is typically called `Dockerfile`.
The first directive at the top of each Dockerfile is `FROM` which states what image to create this
image on top of. For example, we can build off of `ubuntu:22.04` which contains a complete Ubuntu
operating system, but does not have Python installed by default. You can use any image that already
exists to base your image on. An image that has Python preinstalled is `python:3.6-slim-stretch` and
one that has `gcloud` installed is `google/cloud-sdk:slim`. Be careful when choosing images from
unknown sources!

In the example below, we create a Dockerfile that is based on `ubuntu:22.04`. In this file, we show an
example of installing PLINK in the image with the `RUN` directive, which is an arbitrary bash command.
First, we download a bunch of utilities that do not come with Ubuntu using `apt-get`. Next, we
download and install PLINK from source. Finally, we can copy files from your local computer to the
docker image using the `COPY` directive.


.. code-block:: text

    FROM 'ubuntu:22.04'

    RUN apt-get update && apt-get install -y \
        python3 \
        python3-pip \
        tar \
        wget \
        unzip \
        && \
        rm -rf /var/lib/apt/lists/*

    RUN mkdir plink && \
        (cd plink && \
         wget https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20200217.zip && \
         unzip plink_linux_x86_64_20200217.zip && \
         rm -rf plink_linux_x86_64_20200217.zip)

    # copy single script
    COPY my_script.py /scripts/

    # copy entire directory recursively
    COPY . /scripts/

For more information about Dockerfiles and directives that can be used see the following sources:

- https://docs.docker.com/develop/develop-images/dockerfile_best-practices/
- https://docs.docker.com/engine/reference/builder/


Building Images
---------------

To create a Docker image, use

.. code-block:: sh

    docker build -t us-docker.pkg.dev/<my-project>/<my-image>:<tag> -f Dockerfile .

    * `<dir>` is the context directory, `.` means the current working directory,
    * `-t <name>` specifies the image name, and
    * `-f <dockerfile>` specifies the Dockerfile file.
    * A more complete description may be found `here: <https://docs.docker.com/engine/reference/commandline/build/>`__.

For example, we can build an image named us-docker.pkg.dev/<my-project>/<my-image> based on the Dockerfile named Dockerfile, using the current working directory as the context:

.. code-block:: sh

    docker build -t us-docker.pkg.dev/<my-project>/<my-image>:<tag> -f Dockerfile .


In this example we prepend the image name with `us-docker.pkg.dev/<my-project>/` so that it may be pushed to the Google Container Registry, in the next step.

Pushing Images
--------------

To use an image with Batch, you need to upload it somewhere Batch can read it: either the `Google Container Registry <https://cloud.google.com/container-registry/docs/>`__ or
Docker Hub. Below is an example of pushing the image to the Google Container Registry. For more information about pulling and pushing images for the Google Container Registry, see
`here <https://cloud.google.com/container-registry/docs/pushing-and-pulling>`__.

.. code-block:: sh

    docker push us-docker.pkg.dev/<my-project>/<my-image>:<tag>


Now you can use your Docker image with Batch to run your code with the method :meth:`.BashJob.image`
specifying the image as `us-docker.pkg.dev/<my-project>/<my-image>:<tag>`!
