=========================
Install Hail on GNU/Linux
=========================

- Install Java 8.
- Install a recent version of the C and C++ standard libraries. GCC 5.0, LLVM
  version 3.4, or any later versions suffice.
- Install BLAS and LAPACK.
- Install Hail using pip.

On a Debian-like system, the following should suffice:

.. code-block:: sh

   apt-get install \
       openjdk-8-jre-jeadless \
       g++ \
       python3 python3-pip \
       libopenblas liblapack3
   python3 -m pip install hail

`Now let's take Hail for a spin! <try.rst>`__
