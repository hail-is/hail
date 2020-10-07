VCF Combiner
============

Hail has functionality for combining single-sample GVCFs into a multi-sample
matrix table. This process is sometimes called "joint calling", although the
implementation in Hail does not use cohort-level information to reassign individual
genotype calls.

The resulting matrix table is different from a matrix table imported from a project VCF;
see below for a synopsis of how to use this object.

Running the Hail GVCF combiner
------------------------------

The :func:`.run_combiner` function is the primary entry point to running the Hail GVCF
combiner. A typical script for running the combiner on Google Cloud Dataproc using
``hailctl dataproc`` might look like the below::


    import hail as hl
    hl.init(log='/home/hail/combiner.log')

    path_to_input_list = 'gs://path/to/input_files.txt'  # a file with one GVCF path per line

    inputs = []
    with hl.hadoop_open(path_to_input_list, 'r') as f:
        for line in f:
            inputs.append(line.strip())

    output_file = 'gs://path/to/combined/output.mt'  # output destination
    temp_bucket = 'gs://my-temp-bucket'  # bucket for storing intermediate files
    hl.experimental.run_combiner(inputs, out_file=output_file, tmp_path=temp_bucket, reference_genome='GRCh38')


A command-line tool is also provided as a convenient wrapper around this function. This
tool can be run using the below syntax::

    python3 -m hail.experimental.vcf_combiner SAMPLE_MAP OUT_FILE TMP_PATH [OPTIONAL ARGS...]

The below command is equivalent to the Python pipeline above::

    python3 -m hail.experimental.vcf_combiner \
        gs://path/to/input_files.txt \
        gs://path/to/combined/output.mt \
        gs://my-temp-bucket \
        --reference-genome GRCh38 \
        --log /home/hail/combiner.log

Pipeline structure
^^^^^^^^^^^^^^^^^^

The Hail GVCF combiner merges GVCFs hierarchically, parameterized by the `branch_factor`
setting. The number of rounds of merges is defined as ``math.ceil(math.log(N_GVCFS, BRANCH_FACTOR))``.
With the default branch factor of 100, merging between 101 and 10,000 inputs uses 2 rounds,
and merging between 10,001 and 1,000,000 inputs requires 3 rounds.

The combiner will print the execution plan before it runs: ::

    2020-04-01 08:37:32 Hail: INFO: GVCF combiner plan:
        Branch factor: 4
        Batch size: 4
        Combining 50 input files in 3 phases with 6 total jobs.
            Phase 1: 4 jobs corresponding to 13 intermediate output files.
            Phase 2: 1 job corresponding to 4 intermediate output files.
            Phase 3: 1 job corresponding to 1 final output file.


Pain points
^^^^^^^^^^^

The combiner can take some time on large numbers of GVCFs. This time is split
between single-machine planning and compilation work that happens only on the
driver machine, and jobs that take advantage of the entire cluster. For this
reason, its is recommended that clusters with autoscaling functionality are
used, to reduce the overall cost of the pipeline.

For users running with Google Dataproc, the full documentation for creating
autoscaling policies `can be found here <https://cloud.google.com/dataproc/docs/concepts/configuring-clusters/autoscaling#create_an_autoscaling_policy>`__.

A typical YAML policy might look like: ::

    basicAlgorithm:
      cooldownPeriod: 120s
      yarnConfig:
        gracefulDecommissionTimeout: 120s
        scaleDownFactor: 1.0
        scaleUpFactor: 1.0
    secondaryWorkerConfig:
      maxInstances: MAX_PREEMPTIBLE_INSTANCES
      weight: 1
    workerConfig:
      maxInstances: 2
      minInstances: 2
      weight: 1

For ``MAX_PREEMPTIBLE_INSTANCES``, you should fill in a value based on the number of GVCFs you are merging.
For sample sizes up to about 10,000, a value of 100 should be fine.

You can start a cluster with this autoscaling policy using ``hailctl``: ::

    hailctl dataproc start cluster_name ...args... --autoscaling-policy=policy_id_or_uri

Working with sparse matrix tables
---------------------------------

Sparse matrix tables are a new method of representing VCF-style data in a space
efficient way. The 'sparse' modifier refers to the fact that these datasets
contain sample-level reference blocks, just like the input GVCFs -- most of the
entries in the matrix are missing, because that entry falls within a reference
block defined at an earlier locus. While unfamiliar, this representation (1) is
incrementally mergeable with other sparse matrix tables, and (2) scales with the
``N_GVCFs``, not ``N_GVCFs^1.5`` as project VCFs do. The schema of a sparse
matrix table also differs from the schema of a dense project VCF imported to
matrix table. They do not have the same ``GT``, ``AD``, and ``PL`` fields found
in a project VCF, but instead have ``LGT``, ``LAD``, ``LPL`` that provide the
same information, but require additional functions to work with in combination
with a sample's local alleles, ``LA``.

Sample Level Reference Blocks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GVCFs represent blocks of homozygous reference calls of similar qualities using
one record. For example: ::

    #CHROM  POS    ID  REF  ALT  INFO       FORMAT    SAMPLE_1
    chr1    14523  .   C    .    END=15000  GT:DP:GQ  0/0:19:40

This record indicates that sample ``SAMPLE_1`` is homozygous reference until
position 15,000 with approximate ``GQ`` of 40 across the ~500-base-pair. In
short read sequencing, two adjacent loci in a sample's genome will be covered by
mostly the same reads so the quality information about these two loci is highly
correlated; reference blocks explicitly represent regions of reference alleles
with similar quality information.

A sparse matrix table has an entry field ``END`` that corresponds to the GVCF
``INFO`` field, ``END``. It has the same meaning, but only for the single column
where the END resides. In a sparse matrix table, there will be no defined
entries for this sample between ``chr1:14524`` and ``chr1:15000``, inclusive.

Local Alleles
^^^^^^^^^^^^^

The ``LA`` field constitutes a record's **local alleles**, or the alleles that
appeared in the original GVCF for that sample. ``LA`` is used to interpret the
values of ``LGT`` (local genotype), ``LAD`` (local allele depth), and ``LPL``
(local phred-scaled genotype likelihoods). This is best explained through
example: ::

    Variant Information
    -------------------
    locus: chr22:10678889
    alleles: ["CAT", "C", "TAT"]

    Sample1 (reference block, CAT/CAT)
    -------------------------
    DP: 8
    GQ: 21
    LA: [0]
    LGT: 0/0
    LAD: NA
    LPL: NA
    END: 10678898

    equivalent GT: 0/0
    equivalent AD: [8, 0, 0]
    equivalent PL: [0, 21, 42*, 21, 42*, 42*]

    Sample1 (called CAT/TAT)
    -------------------------
    DP: 9
    GQ: 77
    LA: [0, 2]
    LGT: 0/1
    LAD: [3, 6]
    LPL: [137, 0, 77]
    END: NA

    equivalent GT: 0/2
    equivalent AD: [3, 0, 6]
    equivalent PL: [137, 137*, 137*, 0, 137*, 77]

The ``LA`` field for the first sample only includes the reference allele (0),
since this locus was the beginning of a reference block in the original GVCF. In
a reference block, LA will typically be an array with only one value, ``0``. The
``LA`` field for the second sample, which contains a variant allele, includes
the reference and that allele (0 and 2, respectively). PL entries above marked
with an asterisk refer to genotypes with alleles not observed in the original
GVCF; the actual value produced by a tool like GATK will be large (a
low-likelihood value), but not exactly the above. As with standard ``GT``
fields, it is possible to use :meth:`.CallExpression.unphased_diploid_gt_index`
to compute the ``LGT``'s corresponding index into the ``LPL`` array.


Functions
~~~~~~~~~

There are a number of functions for working with sparse data. Of particular
importance is :func:`~.densify`, which transforms a sparse matrix table to a dense
project-VCF-like matrix table on the fly. While computationally expensive, this
operation is necessary for many downstream analyses, and should be thought of as
roughly costing as much as reading a matrix table created by importing a dense
project VCF.

.. currentmodule:: hail.experimental

.. autosummary::

    run_combiner
    densify
    sparse_split_multi
    lgt_to_gt

.. autofunction:: run_combiner
.. autofunction:: densify
.. autofunction:: sparse_split_multi
.. autofunction:: lgt_to_gt
