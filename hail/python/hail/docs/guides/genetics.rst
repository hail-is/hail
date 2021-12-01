Genetics
========

This page tailored how-to guides for small but commonly-used patterns
appearing in genetics pipelines. For documentation on the suite of
genetics functions built into Hail, see the :ref:`genetics methods page <methods_genetics>`.

Formatting
~~~~~~~~~~

Convert variants in string format to separate locus and allele fields
.....................................................................

..
    >>> # this sets up ht for doctest below
    >>> ht = hl.import_table('data/variant-lof.tsv')
    >>> ht = ht.transmute(variant = ht.v)

:**code**:

        >>> ht = ht.key_by(**hl.parse_variant(ht.variant))

:**dependencies**: :func:`.parse_variant`, :meth:`.key_by`

:**understanding**:

    .. container:: toggle

        .. container:: toggle-content

            If your variants are strings of the format 'chr:pos:ref:alt', you may want
            to convert them to separate locus and allele fields. This is useful if
            you have imported a table with variants in string format and you would like to
            join this table with other Hail tables that are keyed by locus and
            alleles.

            ``hl.parse_variant(ht.variant)`` constructs a :class:`.StructExpression`
            containing two nested fields for the locus and alleles. The ** syntax unpacks
            this struct so that the resulting table has two new fields, ``locus`` and
            ``alleles``.

.. _liftover_howto:

Liftover variants from one coordinate system to another
.......................................................

:**tags**: liftover

:**description**: Liftover a Table or MatrixTable from one reference genome to another.

:**code**:

    First, we need to set up the two reference genomes (source and destination):

    >>> rg37 = hl.get_reference('GRCh37')  # doctest: +SKIP
    >>> rg38 = hl.get_reference('GRCh38')  # doctest: +SKIP
    >>> rg37.add_liftover('gs://hail-common/references/grch37_to_grch38.over.chain.gz', rg38)  # doctest: +SKIP

    Then we can liftover the locus coordinates in a Table or MatrixTable (here, `ht`)
    from reference genome ``'GRCh37'`` to ``'GRCh38'``:

    >>> ht = ht.annotate(new_locus=hl.liftover(ht.locus, 'GRCh38'))  # doctest: +SKIP
    >>> ht = ht.filter(hl.is_defined(ht.new_locus))  # doctest: +SKIP
    >>> ht = ht.key_by(locus=ht.new_locus)  # doctest: +SKIP

    Note that this approach does not retain the old locus, nor does it verify
    that the allele has not changed strand. We can keep the old one for
    reference and filter out any liftover that changed strands using:

    >>> ht = ht.annotate(new_locus=hl.liftover(ht.locus, 'GRCh38', include_strand=True),
    ...                  old_locus=ht.locus)  # doctest: +SKIP
    >>> ht = ht.filter(hl.is_defined(ht.new_locus) & ~ht.new_locus.is_negative_strand)  # doctest: +SKIP
    >>> ht = ht.key_by(locus=ht.new_locus.result)  # doctest: +SKIP

:**dependencies**: :func:`.liftover`, :meth:`.add_liftover`, :func:`.get_reference`

Filtering and Pruning
~~~~~~~~~~~~~~~~~~~~~

Remove related individuals from a dataset
.........................................

:**tags**: kinship

:**description**: Compute a measure of kinship between individuals, and then
                  prune related individuals from a matrix table.

:**code**:

        >>> pc_rel = hl.pc_relate(mt.GT, 0.001, k=2, statistics='kin')
        >>> pairs = pc_rel.filter(pc_rel['kin'] > 0.125)
        >>> related_samples_to_remove = hl.maximal_independent_set(pairs.i, pairs.j,
        ...                                                        keep=False)
        >>> result = mt.filter_cols(
        ...     hl.is_defined(related_samples_to_remove[mt.col_key]), keep=False)

:**dependencies**: :func:`.pc_relate`, :func:`.maximal_independent_set`

:**understanding**:

    .. container:: toggle

        .. container:: toggle-content

            To remove related individuals from a dataset, we first compute a measure
            of relatedness between individuals using :func:`.pc_relate`. We filter this
            result based on a kinship threshold, which gives us a table of related pairs.

            From this table of pairs, we can compute the complement of the maximal
            independent set using :func:`.maximal_independent_set`. The parameter
            ``keep=False`` in ``maximal_independent_set`` specifies that we want the
            complement of the set (the variants to remove), rather than the maximal
            independent set itself. It's important to use the complement for filtering,
            rather than the set itself, because the maximal independent set will not contain
            the singleton individuals.

            Once we have a list of samples to remove, we can filter the columns of the
            dataset to remove the related individuals.

Filter loci by a list of locus intervals
........................................

From a table of intervals
+++++++++++++++++++++++++

:**tags**: genomic region, genomic range

:**description**: Import a text file of locus intervals as a table, then use
                  this table to filter the loci in a matrix table.

:**code**:

    >>> interval_table = hl.import_locus_intervals('data/gene.interval_list', reference_genome='GRCh37')
    >>> filtered_mt = mt.filter_rows(hl.is_defined(interval_table[mt.locus]))

:**dependencies**: :func:`.import_locus_intervals`, :meth:`.MatrixTable.filter_rows`

:**understanding**:

    .. container:: toggle

        .. container:: toggle-content

            We have a matrix table ``mt`` containing the loci we would like to filter, and a
            list of locus intervals stored in a file. We can import the intervals into a
            table with :func:`.import_locus_intervals`.

            Hail supports implicit joins between locus intervals and loci, so we can filter
            our dataset to the rows defined in the join between the interval table and our
            matrix table.

            ``interval_table[mt.locus]`` joins the matrix table with the table of intervals
            based on locus and interval<locus> matches. This is a StructExpression, which
            will be defined if the locus was found in any interval, or missing if the locus
            is outside all intervals.

            To do our filtering, we can filter to the rows of our matrix table where the
            struct expression ``interval_table[mt.locus]`` is defined.

            This method will also work to filter a table of loci, as well as a matrix
            table.

From a UCSC BED file
++++++++++++++++++++

:**description**: Import a UCSC BED file as a table of intervals, then use this
                  table to filter the loci in a matrix table.

:**code**:

    >>> interval_table = hl.import_bed('data/file1.bed', reference_genome='GRCh37')
    >>> filtered_mt = mt.filter_rows(hl.is_defined(interval_table[mt.locus]))

:**dependencies**: :func:`.import_bed`, :meth:`.MatrixTable.filter_rows`

Using ``hl.filter_intervals``
+++++++++++++++++++++++++++++

:**description**: Filter using an interval table, suitable for a small list of
                  intervals.

:**code**:

    >>> filtered_mt = hl.filter_intervals(mt, interval_table['interval'].collect())

:**dependencies**: :func:`.methods.filter_intervals`

Declaring intervals with ``hl.parse_locus_interval``
++++++++++++++++++++++++++++++++++++++++++++++++++++

:**description**: Filter to declared intervals.

:**code**:

    >>> intervals = ['1:100M-200M', '16:29.1M-30.2M', 'X']
    >>> filtered_mt = hl.filter_intervals(
    ...     mt,
    ...     [hl.parse_locus_interval(x, reference_genome='GRCh37') for x in intervals])

:**dependencies**: :func:`.methods.filter_intervals`, :func:`.parse_locus_interval`

Pruning Variants in Linkage Disequilibrium
..........................................

:**tags**: LD Prune

:**description**: Remove correlated variants from a matrix table.

:**code**:

    >>> biallelic_mt = mt.filter_rows(hl.len(mt.alleles) == 2)
    >>> pruned_variant_table = hl.ld_prune(mt.GT, r2=0.2, bp_window_size=500000)
    >>> filtered_mt = mt.filter_rows(
    ...     hl.is_defined(pruned_variant_table[mt.row_key]))

:**dependencies**: :func:`.ld_prune`

:**understanding**:

    .. container:: toggle

        .. container:: toggle-content

            Hail's :func:`.ld_prune` method takes a matrix table and returns a table
            with a subset of variants which are uncorrelated with each other. The method
            requires a biallelic dataset, so we first filter our dataset to biallelic
            variants. Next, we get a table of independent variants using :func:`.ld_prune`,
            which we can use to filter the rows of our original dataset.

            Note that it is more efficient to do the final filtering step on the original
            dataset, rather than on the biallelic dataset, so that the biallelic dataset
            does not need to be recomputed.

Analysis
~~~~~~~~

Linear Regression
.................

Single Phenotype
++++++++++++++++

:**tags**: Linear Regression

:**description**: Compute linear regression statistics for a single phenotype.

:**code**:

    Approach #1: Use the :func:`.linear_regression_rows` method

    >>> ht = hl.linear_regression_rows(y=mt.pheno.height,
    ...                                x=mt.GT.n_alt_alleles(),
    ...                                covariates=[1])

    Approach #2: Use the :func:`.aggregators.linreg` aggregator

    >>> mt_linreg = mt.annotate_rows(linreg=hl.agg.linreg(y=mt.pheno.height,
    ...                                                   x=[1, mt.GT.n_alt_alleles()]))

:**dependencies**: :func:`.linear_regression_rows`, :func:`.aggregators.linreg`

:**understanding**:

    .. container:: toggle

        .. container:: toggle-content

            The :func:`.linear_regression_rows` method is more efficient than using the :func:`.aggregators.linreg`
            aggregator. However, the :func:`.aggregators.linreg` aggregator is more flexible (multiple covariates
            can vary by entry) and returns a richer set of statistics.


Multiple Phenotypes
+++++++++++++++++++

:**tags**: Linear Regression

:**description**: Compute linear regression statistics for multiple phenotypes.

:**code**:

    Approach #1: Use the :func:`.linear_regression_rows` method for all phenotypes simultaneously

    >>> ht_result = hl.linear_regression_rows(y=[mt.pheno.height, mt.pheno.blood_pressure],
    ...                                       x=mt.GT.n_alt_alleles(),
    ...                                       covariates=[1])

    Approach #2: Use the :func:`.linear_regression_rows` method for each phenotype sequentially

    >>> ht1 = hl.linear_regression_rows(y=mt.pheno.height,
    ...                                 x=mt.GT.n_alt_alleles(),
    ...                                 covariates=[1])

    >>> ht2 = hl.linear_regression_rows(y=mt.pheno.blood_pressure,
    ...                                 x=mt.GT.n_alt_alleles(),
    ...                                 covariates=[1])

    Approach #3: Use the :func:`.aggregators.linreg` aggregator

    >>> mt_linreg = mt.annotate_rows(
    ...     linreg_height=hl.agg.linreg(y=mt.pheno.height,
    ...                                 x=[1, mt.GT.n_alt_alleles()]),
    ...     linreg_bp=hl.agg.linreg(y=mt.pheno.blood_pressure,
    ...                             x=[1, mt.GT.n_alt_alleles()]))

:**dependencies**: :func:`.linear_regression_rows`, :func:`.aggregators.linreg`

:**understanding**:

    .. container:: toggle

        .. container:: toggle-content

            The :func:`.linear_regression_rows` method is more efficient than using the :func:`.aggregators.linreg`
            aggregator, especially when analyzing many phenotypes. However, the :func:`.aggregators.linreg`
            aggregator is more flexible (multiple covariates can vary by entry) and returns a richer set of
            statistics. The :func:`.linear_regression_rows` method drops samples that have a missing value for
            any of the phenotypes. Therefore, Approach #1 may not be suitable for phenotypes with differential
            patterns of missingness. Approach #2 will do two passes over the data while Approaches #1 and #3 will
            do one pass over the data and compute the regression statistics for each phenotype simultaneously.

Using Variants (SNPs) as Covariates
+++++++++++++++++++++++++++++++++++

:**tags**: sample genotypes covariate

:**description**: Use sample genotype dosage at specific variant(s) as covariates in regression routines.

:**code**:

    Create a sample annotation from the genotype dosage for each variant of
    interest by combining the filter and collect aggregators:

    >>> mt_annot = mt.annotate_cols(
    ...     snp1 = hl.agg.filter(hl.parse_variant('20:13714384:A:C') == mt.row_key,
    ...                          hl.agg.collect(mt.GT.n_alt_alleles()))[0],
    ...     snp2 = hl.agg.filter(hl.parse_variant('20:17479730:T:C') == mt.row_key,
    ...                          hl.agg.collect(mt.GT.n_alt_alleles()))[0])

    Run the GWAS with :func:`.linear_regression_rows` using variant dosages as covariates:

    >>> gwas = hl.linear_regression_rows(  # doctest: +SKIP
    ...     x=mt_annot.GT.n_alt_alleles(),
    ...     y=mt_annot.pheno.blood_pressure,
    ...     covariates=[1, mt_annot.pheno.age, mt_annot.snp1, mt_annot.snp2])

:**dependencies**: :func:`.linear_regression_rows`, :func:`.aggregators.collect`, :func:`.parse_variant`, :func:`.variant_str`

Stratified by Group
+++++++++++++++++++

:**tags**: Linear Regression

:**description**: Compute linear regression statistics for a single phenotype stratified by group.

:**code**:

    Approach #1: Use the :func:`.linear_regression_rows` method for each group

    >>> female_pheno = (hl.case()
    ...                   .when(mt.pheno.is_female, mt.pheno.height)
    ...                   .or_missing())

    >>> linreg_female = hl.linear_regression_rows(y=female_pheno,
    ...                                           x=mt.GT.n_alt_alleles(),
    ...                                           covariates=[1])

    >>> male_pheno = (hl.case()
    ...                 .when(~mt.pheno.is_female, mt.pheno.height)
    ...                 .or_missing())

    >>> linreg_male = hl.linear_regression_rows(y=male_pheno,
    ...                                         x=mt.GT.n_alt_alleles(),
    ...                                         covariates=[1])

    Approach #2: Use the :func:`.aggregators.group_by` and :func:`.aggregators.linreg` aggregators

    >>> mt_linreg = mt.annotate_rows(
    ...     linreg=hl.agg.group_by(mt.pheno.is_female,
    ...                            hl.agg.linreg(y=mt.pheno.height,
    ...                                          x=[1, mt.GT.n_alt_alleles()])))

:**dependencies**: :func:`.linear_regression_rows`, :func:`.aggregators.group_by`, :func:`.aggregators.linreg`

:**understanding**:

    .. container:: toggle

        .. container:: toggle-content

            We have presented two ways to compute linear regression statistics for each value of a grouping
            variable. The first approach utilizes the :func:`.linear_regression_rows` method and must be called
            separately for each group even though it can compute statistics for multiple phenotypes
            simultaneously. This is because the :func:`.linear_regression_rows` method drops samples that have a
            missing value for any of the phenotypes. When the groups are mutually exclusive,
            such as 'Male' and 'Female', no samples remain! Note that we cannot define `male_pheno = ~female_pheno`
            because we subsequently need `male_pheno` to be an expression on the `mt_linreg` matrix table
            rather than `mt`. Lastly, the argument to `root` must be specified for both cases -- otherwise
            the 'Male' output will overwrite the 'Female' output.

            The second approach uses the :func:`.aggregators.group_by` and :func:`.aggregators.linreg`
            aggregators. The aggregation expression generates a dictionary where a key is a group
            (value of the grouping variable) and the corresponding value is the linear regression statistics
            for those samples in the group. The result of the aggregation expression is then used to annotate
            the matrix table.

            The :func:`.linear_regression_rows` method is more efficient than the :func:`.aggregators.linreg`
            aggregator and can be extended to multiple phenotypes, but the :func:`.aggregators.linreg`
            aggregator is more flexible (multiple covariates can be vary by entry) and returns a richer
            set of statistics.

PLINK Conversions
~~~~~~~~~~~~~~~~~

Polygenic Score Calculation
...........................

:**plink**:

    >>> plink --bfile data --score scores.txt sum # doctest: +SKIP

:**tags**: PRS

:**description**: This command is analogous to plink's --score command with the
                  `sum` option. Biallelic variants are required.

:**code**:

    >>> mt = hl.import_plink(
    ...     bed="data/ldsc.bed", bim="data/ldsc.bim", fam="data/ldsc.fam",
    ...     quant_pheno=True, missing='-9')
    >>> mt = hl.variant_qc(mt)
    >>> scores = hl.import_table('data/scores.txt', delimiter=' ', key='rsid',
    ...                          types={'score': hl.tfloat32})
    >>> mt = mt.annotate_rows(**scores[mt.rsid])
    >>> flip = hl.case().when(mt.allele == mt.alleles[0], True).when(
    ...     mt.allele == mt.alleles[1], False).or_missing()
    >>> mt = mt.annotate_rows(flip=flip)
    >>> mt = mt.annotate_rows(
    ...     prior=2 * hl.if_else(mt.flip, mt.variant_qc.AF[0], mt.variant_qc.AF[1]))
    >>> mt = mt.annotate_cols(
    ...     prs=hl.agg.sum(
    ...         mt.score * hl.coalesce(
    ...             hl.if_else(mt.flip, 2 - mt.GT.n_alt_alleles(),
    ...                        mt.GT.n_alt_alleles()), mt.prior)))

:**dependencies**:

    :func:`.import_plink`, :func:`.variant_qc`, :func:`.import_table`,
    :func:`.coalesce`, :func:`.case`, :func:`.cond`, :meth:`.Call.n_alt_alleles`







