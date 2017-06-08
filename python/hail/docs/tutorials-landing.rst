.. _sec-tutorials:

=========
Tutorials
=========

To take Hail for a test drive, go through our tutorials. These can be viewed here in the documentation,
but we recommend instead that you run them yourself with Jupyter.

Download the Hail distribution from our :ref:`getting started page<sec-getting_started>`, and follow
the instructions there to set up the Hail. Inside the unzipped distribution folder, you'll find
a ``tutorials/`` directory. ``cd`` to this directory and run ``jhail`` to start the notebook
server, then click a notebook to begin!

Hail Overview
=============

This notebook is designed to provide a broad overview of Hail’s functionality, with emphasis on the
functionality to manipulate and query a genetic dataset. We walk through a genome-wide SNP association
test, and demonstrate the need to control for confounding caused by population stratification.

.. toctree::
    :maxdepth: 2

    Hail Overview <tutorials/hail-overview.ipynb>

Introduction to the expression language
=======================================

This notebook starts with the basics of the Hail expression language, and builds up practical experience
with the type system, syntax, and functionality. By the end of this notebook, we hope that you will be
comfortable enough to start using the expression language to slice, dice, filter, and query genetic data.

.. toctree::
    :maxdepth: 2

    Expression language <tutorials/introduction-to-the-expression-language.ipynb>

Expression language: query, annotate, and aggregate
===================================================

This notebook uses the Hail expression language to query, filter, and annotate the same thousand genomes
dataset from the overview. We also cover how to compute aggregate statistics from a dataset using the
expression language.


.. toctree::
    :maxdepth: 2

    Expression language 2: <tutorials/expression-language-part-2.ipynb>
