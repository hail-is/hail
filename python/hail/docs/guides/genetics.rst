Genetics
========

Formatting
~~~~~~~~~~

Convert variants in string format to separate locus and allele fields
.....................................................................

..
    >>> ht = hl.import_table('data/variant-lof.tsv')
    >>> ht = ht.transmute(variant = ht.v)

>>> ht = ht.key_by(**hl.parse_variant(ht.variant))

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

>>> pc_rel = hl.pc_relate(mt.GT, 0.001, k=2, statistics='kin')
>>> pairs = pc_rel.filter(pc_rel['kin'] > 0.125)
>>> related_samples_to_remove = hl.maximal_independent_set(pairs.i, pairs.j,
...                                                        keep=False)
>>> result = mt.filter_cols(
...     hl.is_defined(related_samples_to_remove[mt.col_key]), keep=False)

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

Filtering loci by a list of intervals
.....................................

>>> interval_table = hl.import_locus_intervals('data/gene.interval_list')
>>> filtered_mt = mt.filter_rows(hl.is_defined(interval_table[mt.locus]))

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

Alternatively, Hail has a :func:`.filter_intervals` method which can be used to
filter the rows of a table or matrix table using a list of intervals, instead of
a Table of intervals. To use this method, we would first need to collect the
intervals from our table into a list.

>>> interval_table = hl.import_locus_intervals('data/gene.interval_list')
>>> interval_list = [x.interval for x in interval_table.collect()]
>>> filtered_mt = hl.filter_intervals(mt, interval_list)

Both of these methods will also work to filter a table of loci, instead of a
matrix table.

Pruning Variants in Linkage Disequilibrium
..........................................

>>> biallelic_mt = mt.filter_rows(hl.len(mt.alleles) == 2)
>>> pruned_variant_table = hl.ld_prune(mt.GT, r2=0.2, bp_window_size=500000)
>>> filtered_mt = mt.filter_rows(
...     hl.is_defined(pruned_variant_table[mt.row_key]))

Hail's :func:`.ld_prune` method takes a matrix table and returns a table
with a subset of variants which are uncorrelated with each other. The method
requires a biallelic dataset, so we first filter our dataset to biallelic
variants. Next, we get a table of independent variants using :func:`.ld_prune`,
which we can use to filter the rows of our original dataset.

Note that it is more efficient to do the final filtering step on the original
dataset, rather than on the biallelic dataset, so that the biallelic dataset
does not need to be recomputed.

PLINK Conversions
~~~~~~~~~~~~~~~~~

Polygenic Risk Score Calculation
................................

>>> mt = hl.variant_qc(mt)
>>> mt = mt.annotate_cols(
...     prs=hl.agg.sum(
...         mt.score * hl.coalesce(mt.GT.n_alt_alleles(),
...                                2 * mt.variant_qc.AF[1])) / hl.agg.count())

This command is analogous to plink's --score command. This requires biallelic
variants.

The :func:`.coalesce` function takes any number of arguments and returns the
first non-missing one.

Note that plink will score whichever allele you specify in your input, whereas
Hail will score the alternate allele. Flip your alleles if the allele you want
to score is not the alternate.


