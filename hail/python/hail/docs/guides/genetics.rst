Genetics
========

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

:**description**: Import a text file of locus intervals as a table, then use
                  this table to filter the loci in a matrix table.

:**code**:

    >>> interval_table = hl.import_locus_intervals('data/gene.interval_list')
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

            This method will also work to filter a table of loci, instead of
            a matrix table.

From a Python list
++++++++++++++++++

:**description**: Filter loci in a matrix table using a list of intervals.
                  Suitable for a small list of intervals.

:**dependencies**: :func:`.filter_intervals`

:**code**:

    >>> interval_table = hl.import_locus_intervals('data/gene.interval_list')
    >>> interval_list = [x.interval for x in interval_table.collect()]
    >>> filtered_mt = hl.filter_intervals(mt, interval_list)

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

Linear Regression Stratified by Group
.....................................

:**tags**: Linear Regression

:**description**: Compute linear regression statistics stratified by group such as 'male' and 'female'.

:**code**:

    Approach #1: Use the :func:`.linear_regression` method for each group

    >>> female_pheno = hl.case()
    ...                  .when(mt.pheno.is_female, mt.pheno.height)
    ...                  .or_missing()
    >>> mt_linreg = hl.linear_regression(y = female_pheno, x = [1, mt.GT.n_alt_alleles()], root='linreg_female')
    >>> male_pheno = hl.case()
    ...                .when(~mt_linreg.pheno.is_female, mt_linreg.pheno.height)
    ...                .or_missing()
    >>> mt_linreg = hl.linear_regression(y = male_pheno, x = [1, mt_linreg.GT.n_alt_alleles()], root='linreg_male')

    Approach #2: Use the :func:`.aggregators.linreg` and :func:`.aggregators.group_by` aggregators

    >>> mt_linreg = mt.annotate_rows(linreg = hl.agg.group_by(mt.pheno.is_female,
    ...                                                       hl.agg.linreg(mt.pheno.height, [1, mt.GT.n_alt_alleles()])))

:**dependencies**: :func:`.linear_regression`, :func:`.aggregators.linreg`, :func:`.aggregators.group_by`

:**understanding**:

        .. container:: toggle

        .. container:: toggle-content

            We have presented two ways to compute linear regression statistics for each value of a grouping
            variable. The first approach utilizes the :func:`.linear_regression` method and must be called
            separately for each group even though it can compute statistics for multiple phenotypes
            simultaneously. This is because the :func:`.linear_regression` method drops samples that have
            more than one missing value across all phenotypes, such as when the groups are mutually
            exclusive such as 'Male' and 'Female'. Note that the expressions for `female_pheno` and
            `male_pheno` cannot be computed at the same time because they are inputs to two different
            matrix tables. Lastly, the argument to `root` must be specified for both cases -- otherwise
            the output for the 'Male' grouping will overwrite the 'Female' output.

            The second approach uses the :func:`.aggregators.linreg` and :func:`.aggregators.group_by`
            aggregators. The aggregation expression generates a dictionary where the keys are the grouping
            variables and the values are the linear regression statistics for that group. The result of the
            aggregation expression is then used to annotate the matrix table.

PLINK Conversions
~~~~~~~~~~~~~~~~~

Polygenic Risk Score Calculation
................................

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
    ...     prior=2 * hl.cond(mt.flip, mt.variant_qc.AF[0], mt.variant_qc.AF[1]))
    >>> mt = mt.annotate_cols(
    ...     prs=hl.agg.sum(
    ...         mt.score * hl.coalesce(
    ...             hl.cond(mt.flip, 2 - mt.GT.n_alt_alleles(),
    ...                     mt.GT.n_alt_alleles()), mt.prior)))

:**dependencies**:

    :func:`.import_plink`, :func:`.variant_qc`, :func:`.import_table`,
    :func:`.coalesce`, :func:`.case`, :func:`.cond`, :meth:`.Call.n_alt_alleles`







