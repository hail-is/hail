.. _sec-tutorials2:

==============
Hail Tutorials
==============

.. raw:: html

    <!-- for some reason, Safari is confused if we do not include the download attribute in the
    anchor. At time of writing, there does not appear to be a way to tell Sphinx to include that
    attribute. -->
    <p>To take Hail for a test drive, go through our tutorials. These can be viewed here in the
    documentation, but we recommend instead that you run them yourself with Jupyter by
    <a class="reference external" href="tutorials.tar.gz" download>downloading the archive (.tar.gz)</a>
    and running the following:</p>

::

    pip install jupyter
    tar xf tutorials.tar.gz
    jupyter notebook tutorials/

.. toctree::
    :maxdepth: 1

        Genome-Wide Association Study (GWAS) Tutorial <tutorials/01-genome-wide-association-study.ipynb>
        Table Tutorial <tutorials/03-tables.ipynb>
        Aggregation Tutorial <tutorials/04-aggregation.ipynb>
        Filtering and Annotation Tutorial <tutorials/05-filter-annotate.ipynb>
        Table Joins Tutorial <tutorials/06-joins>
        MatrixTable Tutorial <tutorials/07-matrixtable.ipynb>
        Plotting Tutorial<tutorials/08-plotting.ipynb>
