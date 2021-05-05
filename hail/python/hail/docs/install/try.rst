=====================
Your First Hail Query
=====================

We recommend using IPython, a super-powered Python terminal:

.. code-block:: sh

   pip install ipython

Start an IPython session by copy-pasting the below into your Terminal.

.. code-block:: sh

   ipython

Let's randomly generate a dataset according to the Balding-Nichols
Model. The dataset has one-hundred variants and ten samples from three
populations.

.. code-block:: sh

    import hail as hl
    mt = hl.balding_nichols_model(n_populations=3,
                                  n_samples=10,
                                  n_variants=100)
    mt.show()

The last line, ``mt.show()``, displays the dataset in a tabular form.

.. code-block:: sh

    2020-05-09 19:08:07 Hail: INFO: Coerced sorted dataset
    +---------------+------------+------+------+------+------+
    | locus         | alleles    | 0.GT | 1.GT | 2.GT | 3.GT |
    +---------------+------------+------+------+------+------+
    | locus<GRCh37> | array<str> | call | call | call | call |
    +---------------+------------+------+------+------+------+
    | 1:1           | ["A","C"]  | 0/1  | 1/1  | 0/1  | 0/1  |
    | 1:2           | ["A","C"]  | 1/1  | 0/1  | 1/1  | 0/1  |
    | 1:3           | ["A","C"]  | 0/1  | 1/1  | 1/1  | 1/1  |
    | 1:4           | ["A","C"]  | 0/0  | 0/0  | 0/1  | 1/1  |
    | 1:5           | ["A","C"]  | 0/1  | 0/0  | 0/1  | 0/0  |
    | 1:6           | ["A","C"]  | 1/1  | 0/1  | 0/1  | 0/1  |
    | 1:7           | ["A","C"]  | 0/0  | 0/1  | 0/1  | 0/0  |
    | 1:8           | ["A","C"]  | 1/1  | 0/1  | 1/1  | 1/1  |
    | 1:9           | ["A","C"]  | 1/1  | 1/1  | 1/1  | 1/1  |
    | 1:10          | ["A","C"]  | 1/1  | 0/1  | 1/1  | 0/1  |
    | 1:11          | ["A","C"]  | 0/1  | 1/1  | 1/1  | 0/1  |
    +---------------+------------+------+------+------+------+
    showing top 11 rows
    showing the first 4 of 10 columns</code></pre>

Next Steps
""""""""""

- Get the `Hail cheatsheets <../cheatsheets.rst>`__
- Follow the Hail `GWAS Tutorial <../tutorials/01-genome-wide-association-study.rst>`__
- Learn how to use `Hail on Google Cloud <../cloud/google_cloud.rst>`__
