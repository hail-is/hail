from __future__ import print_function  # Python 2 and 3 print compatibility

import warnings

from decorator import decorator

from hail.expr import Type, TGenotype, TString, TVariant, TArray
from hail.typecheck import *
from hail.java import *
from hail.keytable import KeyTable
from hail.representation import Interval, Pedigree, Variant
from hail.utils import Summary, wrap_to_list, hadoop_read
from hail.kinshipMatrix import KinshipMatrix
from hail.ldMatrix import LDMatrix

warnings.filterwarnings(module=__name__, action='once')


@decorator
def requireTGenotype(func, vds, *args, **kwargs):
    if vds._is_generic_genotype:
        if vds.genotype_schema != TGenotype:
            coerced_vds = VariantDataset(vds.hc, vds._jvdf.toVDS())
            return func(coerced_vds, *args, **kwargs)
        else:
            raise TypeError("genotype signature must be Genotype, but found '%s'" % type(vds.genotype_schema))

    return func(vds, *args, **kwargs)


@decorator
def convertVDS(func, vds, *args, **kwargs):
    if vds._is_generic_genotype:
        if isinstance(vds.genotype_schema, TGenotype):
            vds = VariantDataset(vds.hc, vds._jvdf.toVDS())

    return func(vds, *args, **kwargs)

vds_type = lazy()

class VariantDataset(object):
    """Hail's primary representation of genomic data, a matrix keyed by sample and variant.

    Variant datasets may be generated from other formats using the :py:class:`.HailContext` import methods,
    constructed from a variant-keyed :py:class:`KeyTable` using :py:meth:`.VariantDataset.from_table`,
    and simulated using :py:meth:`~hail.HailContext.balding_nichols_model`.

    Once a variant dataset has been written to disk with :py:meth:`~hail.VariantDataset.write`,
    use :py:meth:`~hail.HailContext.read` to load the variant dataset into the environment.

    >>> vds = hc.read("data/example.vds")

    :ivar hc: Hail Context.
    :vartype hc: :class:`.HailContext`
    """

    def __init__(self, hc, jvds):
        self.hc = hc
        self._jvds = jvds

        self._globals = None
        self._sample_annotations = None
        self._colkey_schema = None
        self._sa_schema = None
        self._rowkey_schema = None
        self._va_schema = None
        self._global_schema = None
        self._genotype_schema = None
        self._sample_ids = None
        self._num_samples = None
        self._jvdf_cache = None

    @staticmethod
    @handle_py4j
    @typecheck(table=KeyTable)
    def from_table(table):
        """Construct a sites-only variant dataset from a key table.

        **Examples**

        Import a text table and construct a sites-only VDS:

        >>> table = hc.import_table('data/variant-lof.tsv', types={'v': TVariant()}).key_by('v')
        >>> sites_vds = VariantDataset.from_table(table)

        **Notes**

        The key table must be keyed by one column of type :py:class:`.TVariant`.

        All columns in the key table become variant annotations in the result.
        For example, a key table with key column ``v`` (*Variant*) and column
        ``gene`` (*String*) will produce a sites-only variant dataset with a
        ``va.gene`` variant annotation.

        :param table: Variant-keyed table.
        :type table: :py:class:`.KeyTable`

        :return: Sites-only variant dataset.
        :rtype: :py:class:`.VariantDataset`
        """
        jvds = scala_object(Env.hail().variant, 'VariantDataset').fromKeyTable(table._jkt)
        return VariantDataset(table.hc, jvds)

    @property
    def _jvdf(self):
        if self._jvdf_cache is None:
            if self._is_generic_genotype:
                self._jvdf_cache = Env.hail().variant.GenericDatasetFunctions(self._jvds)
            else:
                self._jvdf_cache = Env.hail().variant.VariantDatasetFunctions(self._jvds)
        return self._jvdf_cache

    @property
    def _is_generic_genotype(self):
        return self._jvds.isGenericGenotype()

    @property
    @handle_py4j
    def sample_ids(self):
        """Return sampleIDs.

        :return: List of sample IDs.
        :rtype: list of str
        """

        if self._sample_ids is None:
            self._sample_ids = jiterable_to_list(self._jvds.sampleIds())
        return self._sample_ids

    @property
    @handle_py4j
    def sample_annotations(self):
        """Return a dict of sample annotations.

        The keys of this dictionary are the sample IDs (strings).
        The values are sample annotations.

        :return: dict
        """

        if self._sample_annotations is None:
            zipped_annotations = Env.jutils().iterableToArrayList(
                self._jvds.sampleIdsAndAnnotations()
            )
            r = {}
            for element in zipped_annotations:
                r[element._1()] = self.sample_schema._convert_to_py(element._2())
            self._sample_annotations = r
        return self._sample_annotations

    @handle_py4j
    def num_partitions(self):
        """Number of partitions.

        **Notes**

        The data in a variant dataset is divided into chunks called partitions, which may be stored together or across a network, so that each partition may be read and processed in parallel by available cores. Partitions are a core concept of distributed computation in Spark, see `here <http://spark.apache.org/docs/latest/programming-guide.html#resilient-distributed-datasets-rdds>`__ for details.

        :rtype: int
        """

        return self._jvds.nPartitions()

    @property
    @handle_py4j
    def num_samples(self):
        """Number of samples.

        :rtype: int
        """

        if self._num_samples is None:
            self._num_samples = self._jvds.nSamples()
        return self._num_samples

    @handle_py4j
    def count_variants(self):
        """Count number of variants in variant dataset.

        :rtype: long
        """

        return self._jvds.countVariants()

    @handle_py4j
    def was_split(self):
        """True if multiallelic variants have been split into multiple biallelic variants.

        Result is True if :py:meth:`~hail.VariantDataset.split_multi` or :py:meth:`~hail.VariantDataset.filter_multi` has been called on this variant dataset,
        or if the variant dataset was imported with :py:meth:`~hail.HailContext.import_plink`, :py:meth:`~hail.HailContext.import_gen`,
        or :py:meth:`~hail.HailContext.import_bgen`, or if the variant dataset was simulated with :py:meth:`~hail.HailContext.balding_nichols_model`.

        :rtype: bool
        """

        return self._jvds.wasSplit()

    @handle_py4j
    def file_version(self):
        """File version of variant dataset.

        :rtype: int
        """

        return self._jvds.fileVersion()

    @handle_py4j
    @typecheck_method(key_exprs=oneof(strlike, listof(strlike)),
               agg_exprs=oneof(strlike, listof(strlike)))
    def aggregate_by_key(self, key_exprs, agg_exprs):
        """Aggregate by user-defined key and aggregation expressions to produce a KeyTable.
        Equivalent to a group-by operation in SQL.

        **Examples**

        Compute the number of LOF heterozygote calls per gene per sample:

        >>> kt_result = (vds
        ...     .aggregate_by_key(['Sample = s', 'Gene = va.gene'],
        ...                        'nHet = g.filter(g => g.isHet() && va.consequence == "LOF").count()')
        ...     .export("test.tsv"))

        This will produce a :class:`KeyTable` with 3 columns (`Sample`, `Gene`, `nHet`).

        :param key_exprs: Named expression(s) for which fields are keys.
        :type key_exprs: str or list of str

        :param agg_exprs: Named aggregation expression(s).
        :type agg_exprs: str or list of str

        :rtype: :class:`.KeyTable`
        """

        if isinstance(key_exprs, list):
            key_exprs = ",".join(key_exprs)
        if isinstance(agg_exprs, list):
            agg_exprs = ",".join(agg_exprs)

        return KeyTable(self.hc, self._jvds.aggregateByKey(key_exprs, agg_exprs))

    @handle_py4j
    @requireTGenotype
    @typecheck_method(expr=oneof(strlike, listof(strlike)),
               propagate_gq=bool)
    def annotate_alleles_expr(self, expr, propagate_gq=False):
        """Annotate alleles with expression.

        .. include:: requireTGenotype.rst

        **Examples**

        To create a variant annotation ``va.nNonRefSamples: Array[Int]`` where the ith entry of
        the array is the number of samples carrying the ith alternate allele:

        >>> vds_result = vds.annotate_alleles_expr('va.nNonRefSamples = gs.filter(g => g.isCalledNonRef()).count()')

        **Notes**

        This method is similar to :py:meth:`.annotate_variants_expr`. :py:meth:`.annotate_alleles_expr` dynamically splits multi-allelic sites,
        evaluates each expression on each split allele separately, and for each expression annotates with an array with one element per alternate allele. In the splitting, genotypes are downcoded and each alternate allele is represented
        using its minimal representation (see :py:meth:`split_multi` for more details).


        :param expr: Annotation expression.
        :type expr: str or list of str
        :param bool propagate_gq: Propagate GQ instead of computing from (split) PL.

        :return: Annotated variant dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        if isinstance(expr, list):
            expr = ",".join(expr)

        jvds = self._jvdf.annotateAllelesExpr(expr, propagate_gq)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @typecheck_method(expr=oneof(strlike, listof(strlike)))
    def annotate_genotypes_expr(self, expr):
        """Annotate genotypes with expression.

        **Examples**

        Convert the genotype schema to a :py:class:`~hail.expr.TStruct` with two fields ``GT`` and ``CASE_HET``:

        >>> vds_result = vds.annotate_genotypes_expr('g = {GT: g.gt, CASE_HET: sa.pheno.isCase && g.isHet()}')

        Assume a VCF is imported with ``generic=True`` and the resulting genotype schema
        is a ``Struct`` and the field ``GTA`` is a ``Call`` type. Use the ``.toGenotype()`` method in the
        expression language to convert a ``Call`` to a ``Genotype``. ``vds_gta`` will have a genotype schema equal to
        :py:class:`~hail.expr.TGenotype`

        >>> vds_gta = (hc.import_vcf('data/example3.vcf.bgz', generic=True, call_fields=['GTA'])
        ...                 .annotate_genotypes_expr('g = g.GTA.toGenotype()'))

        **Notes**

        :py:meth:`~hail.VariantDataset.annotate_genotypes_expr` evaluates the expression given by ``expr`` and assigns
        the result of the right hand side to the annotation path specified by the left-hand side (must
        begin with ``g``). This is analogous to :py:meth:`~hail.VariantDataset.annotate_variants_expr` and
        :py:meth:`~hail.VariantDataset.annotate_samples_expr` where the annotation paths are ``va`` and ``sa`` respectively.

        ``expr`` is in genotype context so the following symbols are in scope:

          - ``g``: genotype annotation
          - ``v`` (*Variant*): :ref:`variant`
          - ``va``: variant annotations
          - ``s`` (*Sample*): sample
          - ``sa``: sample annotations
          - ``global``: global annotations

        For more information, see the documentation on writing `expressions <overview.html#expressions>`__
        and using the `Hail Expression Language <exprlang.html>`__.

        .. warning::

            - If the resulting genotype schema is not :py:class:`~hail.expr.TGenotype`,
              subsequent function calls on the annotated variant dataset may not work such as
              :py:meth:`~hail.VariantDataset.pca` and :py:meth:`~hail.VariantDataset.linreg`.

            - Hail performance may be significantly slower if the annotated variant dataset does not have a
              genotype schema equal to :py:class:`~hail.expr.TGenotype`.

            - Genotypes are immutable. For example, if ``g`` is initially of type ``Genotype``, the expression
              ``g.gt = g.gt + 1`` will return a ``Struct`` with one field ``gt`` of type ``Int`` and **NOT** a ``Genotype``
              with the ``gt`` incremented by 1.

        :param expr: Annotation expression.
        :type expr: str or list of str

        :return: Annotated variant dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        if isinstance(expr, list):
            expr = ",".join(expr)

        jvds = self._jvdf.annotateGenotypesExpr(expr)
        vds = VariantDataset(self.hc, jvds)
        if isinstance(vds.genotype_schema, TGenotype):
            return VariantDataset(self.hc, vds._jvdf.toVDS())
        else:
            return vds

    @handle_py4j
    @typecheck_method(expr=oneof(strlike, listof(strlike)))
    def annotate_global_expr(self, expr):
        """Annotate global with expression.

        **Example**

        Annotate global with an array of populations:

        >>> vds = vds.annotate_global_expr('global.pops = ["FIN", "AFR", "EAS", "NFE"]')

        Create, then overwrite, then drop a global annotation:

        >>> vds = vds.annotate_global_expr('global.pops = ["FIN", "AFR", "EAS"]')
        >>> vds = vds.annotate_global_expr('global.pops = ["FIN", "AFR", "EAS", "NFE"]')
        >>> vds = vds.annotate_global_expr('global.pops = drop(global, pops)')

        The expression namespace contains only one variable:

        - ``global``: global annotations

        :param expr: Annotation expression
        :type expr: str or list of str

        :return: Annotated variant dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        if isinstance(expr, list):
            expr = ','.join(expr)

        jvds = self._jvds.annotateGlobalExpr(expr)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @typecheck_method(path=strlike,
                      annotation=anytype,
                      annotation_type=Type)
    def annotate_global(self, path, annotation, annotation_type):
        """Add global annotations from Python objects.

        **Examples**

        Add populations as a global annotation:
        
        >>> vds_result = vds.annotate_global('global.populations',
        ...                                     ['EAS', 'AFR', 'EUR', 'SAS', 'AMR'],
        ...                                     TArray(TString()))

        **Notes**

        This method registers new global annotations in a VDS. These annotations
        can then be accessed through expressions in downstream operations. The
        Hail data type must be provided and must match the given ``annotation``
        parameter.

        :param str path: annotation path starting in 'global'

        :param annotation: annotation to add to global

        :param annotation_type: Hail type of annotation
        :type annotation_type: :py:class:`.Type`

        :return: Annotated variant dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        annotation_type._typecheck(annotation)

        annotated = self._jvds.annotateGlobal(annotation_type._convert_to_j(annotation), annotation_type._jtype, path)
        assert annotated.globalSignature().typeCheck(annotated.globalAnnotation()), 'error in java type checking'
        return VariantDataset(self.hc, annotated)

    @handle_py4j
    @typecheck_method(expr=oneof(strlike, listof(strlike)))
    def annotate_samples_expr(self, expr):
        """Annotate samples with expression.

        **Examples**

        Compute per-sample GQ statistics for hets:

        >>> vds_result = (vds.annotate_samples_expr('sa.gqHetStats = gs.filter(g => g.isHet()).map(g => g.gq).stats()')
        ...     .export_samples('output/samples.txt', 'sample = s, het_gq_mean = sa.gqHetStats.mean'))

        Compute the list of genes with a singleton LOF per sample:

        >>> variant_annotations_table = hc.import_table('data/consequence.tsv', impute=True).key_by('Variant')
        >>> vds_result = (vds.annotate_variants_table(variant_annotations_table, root='va.consequence')
        ...     .annotate_variants_expr('va.isSingleton = gs.map(g => g.nNonRefAlleles()).sum() == 1')
        ...     .annotate_samples_expr('sa.LOF_genes = gs.filter(g => va.isSingleton && g.isHet() && va.consequence == "LOF").map(g => va.gene).collect()'))

        To create an annotation for only a subset of samples based on an existing annotation:

        >>> vds_result = vds.annotate_samples_expr('sa.newpheno = if (sa.pheno.cohortName == "cohort1") sa.pheno.bloodPressure else NA: Double')

        .. note::

            For optimal performance, be sure to explicitly give the alternative (``NA``) the same type as the consequent (``sa.pheno.bloodPressure``).

        **Notes**

        ``expr`` is in sample context so the following symbols are in scope:

        - ``s`` (*Sample*): sample
        - ``sa``: sample annotations
        - ``global``: global annotations
        - ``gs`` (*Aggregable[Genotype]*): aggregable of :ref:`genotype` for sample ``s``

        :param expr: Annotation expression.
        :type expr: str or list of str

        :return: Annotated variant dataset.
        :rtype: :class:`.VariantDataset`
        """

        if isinstance(expr, list):
            expr = ','.join(expr)

        jvds = self._jvds.annotateSamplesExpr(expr)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @typecheck_method(table=KeyTable,
                      root=nullable(strlike),
                      expr=nullable(strlike),
                      vds_key=nullable(strlike),
                      product=bool)
    def annotate_samples_table(self, table, root=None, expr=None, vds_key=None, product=False):
        """Annotate samples with a key table.

        **Examples**

        To annotates samples using `samples1.tsv` with type imputation::

        >>> table = hc.import_table('data/samples1.tsv', impute=True).key_by('Sample')
        >>> vds_result = vds.annotate_samples_table(table, root='sa.pheno')

        Given this file

        .. code-block:: text

            $ cat data/samples1.tsv
            Sample	Height	Status  Age
            PT-1234	154.1	ADHD	24
            PT-1236	160.9	Control	19
            PT-1238	NA	ADHD	89
            PT-1239	170.3	Control	55

        the three new sample annotations are ``sa.pheno.Height: Double``, ``sa.pheno.Status: String``, and ``sa.pheno.Age: Int``.

        To annotate without type imputation, resulting in all String types:

        >>> annotations = hc.import_table('data/samples1.tsv').key_by('Sample')
        >>> vds_result = vds.annotate_samples_table(annotations, root='sa.phenotypes')

        **Detailed examples**

        Let's import annotations from a CSV file with missing data and special characters

        .. code-block:: text

            $ cat data/samples2.tsv
            Batch,PT-ID
            1kg,PT-0001
            1kg,PT-0002
            study1,PT-0003
            study3,PT-0003
            .,PT-0004
            1kg,PT-0005
            .,PT-0006
            1kg,PT-0007

        In this case, we should:

        - Pass the non-default delimiter ``,``

        - Pass the non-default missing value ``.``

        >>> annotations = hc.import_table('data/samples2.tsv', delimiter=',', missing='.').key_by('PT-ID')
        >>> vds_result = vds.annotate_samples_table(annotations, root='sa.batch')

        Let's import annotations from a file with no header and sample IDs that need to be transformed. 
        Suppose the vds sample IDs are of the form ``NA#####``. This file has no header line, and the 
        sample ID is hidden in a field with other information.

        .. code-block:: text

            $ cat data/samples3.tsv
            1kg_NA12345   female
            1kg_NA12346   male
            1kg_NA12348   female
            pgc_NA23415   male
            pgc_NA23418   male

        To import it:

        >>> annotations = (hc.import_table('data/samples3.tsv', no_header=True)
        ...                  .annotate('sample = f0.split("_")[1]')
        ...                  .key_by('sample'))
        >>> vds_result = vds.annotate_samples_table(annotations,
        ...                             expr='sa.sex = table.f1, sa.batch = table.f0.split("_")[0]')

        **Notes** 

        This method takes as an argument a :class:`.KeyTable` object. Hail has a default join strategy
        for tables keyed by String, which is to join by sample ID. If the table is keyed by something else, like
        population or cohort, then the ``vds_key`` argument must be passed to describe the key in the dataset 
        to use for the join. This argument expects a list of Hail expressions whose types match, in order, the 
        table's key types.
        
        Each expression in the list ``vds_key`` has the following symbols in
        scope:

          - ``s`` (*String*): sample ID
          - ``sa``: sample annotations
        
        **The** ``root`` **and** ``expr`` **arguments**
        
        .. note::
        
            One of ``root`` or ``expr`` is required, but not both. 
            
        The ``expr`` parameter expects an annotation expression involving ``sa`` (the existing 
        sample annotations in the dataset) and ``table`` (a struct containing the columns in 
        the table), like ``sa.col1 = table.col1, sa.col2 = table.col2`` or ``sa = merge(sa, table)``.
        The ``root`` parameter expects an annotation path beginning in ``sa``, like ``sa.annotations``.
        Passing ``root='sa.annotations'`` is exactly the same as passing ``expr='sa.annotations = table'``.

        ``expr`` has the following symbols in scope:

          - ``sa``: sample annotations
          - ``table``: See note.

        .. note:: 
        
            The value of ``table`` inside root/expr depends on the number of values in the key table, 
            as well as the ``product`` argument. There are three behaviors based on the number of values
            and one branch for ``product`` being true and false, for a total of six modes:
            
            +-------------------------+-------------+--------------------+-----------------------------------------------+
            | Number of value columns | ``product`` | Type of  ``table`` | Value of  ``table``                           |
            +=========================+=============+====================+===============================================+
            | More than 2             | False       | ``Struct``         | Struct with an element for each column.       |
            +-------------------------+-------------+--------------------+-----------------------------------------------+
            | 1                       | False       | ``T``              | The value column.                             |
            +-------------------------+-------------+--------------------+-----------------------------------------------+
            | 0                       | False       | ``Boolean``        | Existence of any matching key.                |
            +-------------------------+-------------+--------------------+-----------------------------------------------+
            | More than 2             | True        | ``Array[Struct]``  | An array with a struct for each matching key. |
            +-------------------------+-------------+--------------------+-----------------------------------------------+
            | 1                       | True        | ``Array[T]``       | An array with a value for each matching key.  |
            +-------------------------+-------------+--------------------+-----------------------------------------------+
            | 0                       | True        | ``Int``            | The number of matching keys.                  |
            +-------------------------+-------------+--------------------+-----------------------------------------------+  

        **Common uses for the** ``expr`` **argument**

        Put annotations on the top level under ``sa``

        .. code-block:: text

            expr='sa = merge(sa, table)'

        Annotate only specific annotations from the table

        .. code-block:: text

            expr='sa.annotations = select(table, toKeep1, toKeep2, toKeep3)'

        The above is equivalent to

        .. code-block:: text

            expr='''sa.annotations.toKeep1 = table.toKeep1,
                sa.annotations.toKeep2 = table.toKeep2,
                sa.annotations.toKeep3 = table.toKeep3'''
                
        Finally, for more information about importing key tables from text, 
        see the documentation for :py:meth:`.HailContext.import_table`.

        :param table: Key table.
        :type table: :py:class:`.KeyTable`

        :param root: Sample annotation path to store text table. (This or ``expr`` required).
        :type root: str or None

        :param expr: Annotation expression. (This or ``root`` required).
        :type expr: str or None
        
        :param vds_key: Join key for the dataset, if not sample ID.
        :type vds_key: str, list of str, or None.
        
        :param bool product: Join with all matching keys (see note).

        :return: Annotated variant dataset.
        :rtype: :py:class:`.VariantDataset`
        """
        if vds_key:
            vds_key = wrap_to_list(vds_key)

        return VariantDataset(self.hc, self._jvds.annotateSamplesTable(table._jkt, vds_key, root, expr, product))

    @handle_py4j
    @typecheck_method(expr=oneof(strlike, listof(strlike)))
    def annotate_variants_expr(self, expr):
        """Annotate variants with expression.

        **Examples**

        Compute GQ statistics about heterozygotes per variant:

        >>> vds_result = vds.annotate_variants_expr('va.gqHetStats = gs.filter(g => g.isHet()).map(g => g.gq).stats()')

        Collect a list of sample IDs with non-ref calls in LOF variants:

        >>> vds_result = vds.annotate_variants_expr('va.nonRefSamples = gs.filter(g => g.isCalledNonRef()).map(g => s).collect()')

        Substitute a custom string for the rsID field:

        >>> vds_result = vds.annotate_variants_expr('va.rsid = str(v)')

        **Notes**

        ``expr`` is in variant context so the following symbols are in scope:

          - ``v`` (*Variant*): :ref:`variant`
          - ``va``: variant annotations
          - ``global``: global annotations
          - ``gs`` (*Aggregable[Genotype]*): aggregable of :ref:`genotype` for variant ``v``

        For more information, see the documentation on writing `expressions <overview.html#expressions>`__
        and using the `Hail Expression Language <exprlang.html>`__.

        :param expr: Annotation expression or list of annotation expressions.
        :type expr: str or list of str

        :return: Annotated variant dataset.
        :rtype: :class:`.VariantDataset`
        """

        if isinstance(expr, list):
            expr = ','.join(expr)

        jvds = self._jvds.annotateVariantsExpr(expr)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @typecheck_method(table=KeyTable,
                      root=nullable(strlike),
                      expr=nullable(strlike),
                      vds_key=nullable(oneof(strlike, listof(strlike))),
                      product=bool)
    def annotate_variants_table(self, table, root=None, expr=None, vds_key=None, product=False):
        """Annotate variants with a key table.

        **Examples**

        Add annotations from a variant-keyed tab separated file:

        >>> table = hc.import_table('data/variant-lof.tsv', impute=True).key_by('v')
        >>> vds_result = vds.annotate_variants_table(table, root='va.lof')
        
        Add annotations from a locus-keyed TSV:
        
        >>> kt = hc.import_table('data/locus-table.tsv', impute=True).key_by('Locus')
        >>> vds_result = vds.annotate_variants_table(table, root='va.scores')

        Add annotations from a gene-and-type-keyed TSV:
    
        >>> table = hc.import_table('data/locus-metadata.tsv', impute=True).key_by(['gene', 'type'])
        >>> vds_result = (vds.annotate_variants_table(table,
        ...       root='va.foo',
        ...       vds_key=['va.gene', 'if (va.score > 10) "Type1" else "Type2"']))

        Annotate variants with the target in a GATK interval list file:
        
        >>> intervals = KeyTable.import_interval_list('data/exons2.interval_list')
        >>> vds_result = vds.annotate_variants_table(intervals, root='va.exon')
        
        Annotate variants with all targets from matching intervals in a GATK interval list file:
        
        >>> intervals = KeyTable.import_interval_list('data/exons2.interval_list')
        >>> vds_result = vds.annotate_variants_table(intervals, root='va.exons', product=True)
        
        Annotate variants using a UCSC BED file, marking each variant true/false for an overlap with any interval:
        
        >>> intervals = KeyTable.import_bed('data/file2.bed')
        >>> vds_result = vds.annotate_variants_table(intervals, root='va.bed')
        
        **Notes**
        
        This method takes as an argument a :class:`.KeyTable` object. Hail has default join strategies
        for tables keyed by Variant, Locus, or Interval.
        
        **Join strategies:**
                  
        If the key is a ``Variant``, then a variant in the dataset will match a variant in the 
        table that is equivalent. Be careful, however: ``1:1:A:T`` does not match ``1:1:A:T,C``, 
        and vice versa. 
        
        If the key is a ``Locus``, then a variant in the dataset will match any locus in the table
        which is equivalent to ``v.locus`` (same chromosome and position).
            
        If the key is an ``Interval``, then a variant in the dataset will match any interval in 
        the table that contains the variant's locus (chromosome and position).
        
        If the key is not one of the above three types (a String representing gene ID, for instance),
        or if another join strategy should be used for a key of one of these three types (join with a 
        locus object in variant annotations, for instance) for these types, then the ``vds_key`` argument 
        should be passed. This argument expects a list of expressions whose types match, in order, 
        the table's key types. Note that using ``vds_key`` is slower than annotation with a standard 
        key type.

        Each expression in the list ``vds_key`` has the following symbols in
        scope:

          - ``v`` (*Variant*): :ref:`variant`
          - ``va``: variant annotations
        
        **The** ``root`` **and** ``expr`` **arguments**
        
        .. note::
        
            One of ``root`` or ``expr`` is required, but not both. 
            
        The ``expr`` parameter expects an annotation assignment involving ``va`` (the existing 
        variant annotations in the dataset) and ``table`` (the values(s) in the table),
        like ``va.col1 = table.col1, va.col2 = table.col2`` or ``va = merge(va, table)``.
        The ``root`` parameter expects an annotation path beginning in ``va``, like ``va.annotations``.
        Passing ``root='va.annotations'`` is the same as passing ``expr='va.annotations = table'``.

        ``expr`` has the following symbols in scope:

          - ``va``: variant annotations
          - ``table``: See note.

        .. note:: 
        
            The value of ``table`` inside root/expr depends on the number of values in the key table, 
            as well as the ``product`` argument. There are three behaviors based on the number of values
            and one branch for ``product`` being true and false, for a total of six modes:
            
            +-------------------------+-------------+--------------------+-----------------------------------------------+
            | Number of value columns | ``product`` | Type of  ``table`` | Value of  ``table``                           |
            +=========================+=============+====================+===============================================+
            | More than 2             | False       | ``Struct``         | Struct with an element for each column.       |
            +-------------------------+-------------+--------------------+-----------------------------------------------+
            | 1                       | False       | ``T``              | The value column.                             |
            +-------------------------+-------------+--------------------+-----------------------------------------------+
            | 0                       | False       | ``Boolean``        | Existence of any matching key.                |
            +-------------------------+-------------+--------------------+-----------------------------------------------+
            | More than 2             | True        | ``Array[Struct]``  | An array with a struct for each matching key. |
            +-------------------------+-------------+--------------------+-----------------------------------------------+
            | 1                       | True        | ``Array[T]``       | An array with a value for each matching key.  |
            +-------------------------+-------------+--------------------+-----------------------------------------------+
            | 0                       | True        | ``Int``            | The number of matching keys.                  |
            +-------------------------+-------------+--------------------+-----------------------------------------------+  
                      
        **Common uses for the** ``expr`` **argument**

        Put annotations on the top level under ``va``:

        .. code-block:: text

            expr='va = merge(va, table)'

        Annotate only specific annotations from the table:

        .. code-block:: text

            expr='va.annotations = select(table, toKeep1, toKeep2, toKeep3)'

        The above is roughly equivalent to:

        .. code-block:: text

            expr='''va.annotations.toKeep1 = table.toKeep1,
                va.annotations.toKeep2 = table.toKeep2,
                va.annotations.toKeep3 = table.toKeep3'''
                
        Finally, for more information about importing key tables from text, 
        see the documentation for :py:meth:`.HailContext.import_table`.

        :param table: Key table.
        :type table: :py:class:`.KeyTable`

        :param root: Variant annotation path to store text table. (This or ``expr`` required).
        :type root: str or None

        :param expr: Annotation expression. (This or ``root`` required).
        :type expr: str or None
        
        :param vds_key: Join key for the dataset. Much slower than default joins.
        :type vds_key: str, list of str, or None.
        
        :param bool product: Join with all matching keys (see note).

        :return: Annotated variant dataset.
        :rtype: :py:class:`.VariantDataset`
        """
        if vds_key:
            vds_key = wrap_to_list(vds_key)

        jvds = self._jvds.annotateVariantsTable(table._jkt, vds_key, root, expr, product)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @typecheck_method(other=vds_type,
                      expr=nullable(strlike),
                      root=nullable(strlike))
    def annotate_variants_vds(self, other, expr=None, root=None):
        '''Annotate variants with variant annotations from .vds file.

        **Examples**

        Import a second variant dataset with annotations to merge into ``vds``:

        >>> vds1 = vds.annotate_variants_expr('va = drop(va, anno1)')
        >>> vds2 = (hc.read("data/example2.vds")
        ...           .annotate_variants_expr('va = select(va, anno1, toKeep1, toKeep2, toKeep3)'))

        Copy the ``anno1`` annotation from ``other`` to ``va.annot``:

        >>> vds_result = vds1.annotate_variants_vds(vds2, expr='va.annot = vds.anno1')

        Merge the variant annotations from the two vds together and places them
        at ``va``:

        >>> vds_result = vds1.annotate_variants_vds(vds2, expr='va = merge(va, vds)')

        Select a subset of the annotations from ``other``:

        >>> vds_result = vds1.annotate_variants_vds(vds2, expr='va.annotations = select(vds, toKeep1, toKeep2, toKeep3)')

        The previous expression is equivalent to:

        >>> vds_result = vds1.annotate_variants_vds(vds2, expr='va.annotations.toKeep1 = vds.toKeep1, ' +
        ...                                       'va.annotations.toKeep2 = vds.toKeep2, ' +
        ...                                       'va.annotations.toKeep3 = vds.toKeep3')

        **Notes**

        Using this method requires one of the two optional arguments: ``expr``
        and ``root``. They specify how to insert the annotations from ``other``
        into the this vds's variant annotations.

        The ``root`` argument copies all the variant annotations from ``other``
        to the specified annotation path.

        The ``expr`` argument expects an annotation assignment whose scope
        includes, ``va``, the variant annotations in the current VDS, and ``vds``,
        the variant annotations in ``other``.

        VDSes with multi-allelic variants may produce surprising results because
        all alternate alleles are considered part of the variant key. For
        example:

        - The variant ``22:140012:A:T,TTT`` will not be annotated by
          ``22:140012:A:T`` or ``22:140012:A:TTT``

        - The variant ``22:140012:A:T`` will not be annotated by
          ``22:140012:A:T,TTT``

        It is possible that an unsplit variant dataset contains no multiallelic
        variants, so ignore any warnings Hail prints if you know that to be the
        case.  Otherwise, run :py:meth:`.split_multi` before :py:meth:`.annotate_variants_vds`.

        :param VariantDataset other: Variant dataset to annotate with.

        :param str root: Sample annotation path to add variant annotations.

        :param str expr: Annotation expression.

        :return: Annotated variant dataset.
        :rtype: :py:class:`.VariantDataset`
        '''

        jvds = self._jvds.annotateVariantsVDS(other._jvds, joption(root), joption(expr))

        return VariantDataset(self.hc, jvds)

    def annotate_variants_db(self, annotations, gene_key=None):
        """
        Annotate variants using the Hail annotation database.

        .. warning::

            Experimental. Supported only while running Hail on the Google Cloud Platform.

        Documentation describing the annotations that are accessible through this method can be found :ref:`here <sec-annotationdb>`.

        **Examples**

        Annotate variants with CADD raw and PHRED scores:

        >>> vds = vds.annotate_variants_db(['va.cadd.RawScore', 'va.cadd.PHRED']) # doctest: +SKIP

        Annotate variants with gene-level PLI score, using the VEP-generated gene symbol to map variants to genes: 

        >>> pli_vds = vds.annotate_variants_db('va.gene.constraint.pli') # doctest: +SKIP

        Again annotate variants with gene-level PLI score, this time using the existing ``va.gene_symbol`` annotation 
        to map variants to genes:

        >>> vds = vds.annotate_variants_db('va.gene.constraint.pli', gene_key='va.gene_symbol') # doctest: +SKIP

        **Notes**

        Annotations in the database are bi-allelic, so splitting multi-allelic variants in the VDS before using this 
        method is recommended to capture all appropriate annotations from the database. To do this, run :py:meth:`split_multi` 
        prior to annotating variants with this method:

        >>> vds = vds.split_multi().annotate_variants_db(['va.cadd.RawScore', 'va.cadd.PHRED']) # doctest: +SKIP

        To add VEP annotations, or to add gene-level annotations without a predefined gene symbol for each variant, the 
        :py:meth:`~.VariantDataset.annotate_variants_db` method runs Hail's :py:meth:`~.VariantDataset.vep` method on the 
        VDS. This means that your cluster must be properly initialized to run VEP.

        .. warning::

            If you want to add VEP annotations to your VDS, make sure to add the initialization action 
            :code:`gs://hail-common/vep/vep/vep85-init.sh` when starting your cluster.

        :param annotations: List of annotations to import from the database.
        :type annotations: str or list of str 

        :param gene_key: Existing variant annotation used to map variants to gene symbols if importing gene-level 
            annotations. If not provided, the method will add VEP annotations and parse them as described in the 
            database documentation to obtain one gene symbol per variant.
        :type gene_key: str

        :return: Annotated variant dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        # import modules needed by this function
        import sqlite3

        # collect user-supplied annotations, converting str -> list if necessary and dropping duplicates
        annotations = list(set(wrap_to_list(annotations)))

        # open connection to in-memory SQLite database
        conn = sqlite3.connect(':memory:')

        # load database with annotation metadata, print error if not on Google Cloud Platform
        try:
            f = hadoop_read('gs://annotationdb/ADMIN/annotationdb.sql')
        except FatalError:
            raise EnvironmentError('Cannot read from Google Storage. Must be running on Google Cloud Platform to use annotation database.')
        else:
            curs = conn.executescript(f.read())
            f.close()

        # parameter substitution string to put in SQL query
        like = ' OR '.join('a.annotation LIKE ?' for i in xrange(2*len(annotations)))

        # query to extract path of all needed database files and their respective annotation exprs 
        qry = """SELECT file_path, annotation, file_type, file_element, f.file_id
                 FROM files AS f INNER JOIN annotations AS a ON f.file_id = a.file_id
                 WHERE {}""".format(like)

        # run query and collect results in a file_path: expr dictionary
        results = curs.execute(qry, [x + '.%' for x in annotations] + annotations).fetchall()

        # all file_ids to be used
        file_ids = list(set([x[4] for x in results]))

        # parameter substitution string
        sub = ','.join('?' for x in file_ids)

        # query to fetch count of total annotations in each file
        qry = """SELECT file_path, COUNT(*)
                 FROM files AS f INNER JOIN annotations AS a ON f.file_id = a.file_id
                 WHERE f.file_id IN ({})
                 GROUP BY file_path""".format(sub)

        # collect counts in file_id: count dictionary
        cnts = {x[0]: x[1] for x in curs.execute(qry, file_ids).fetchall()}

        # close database connection
        conn.close()

        # collect dictionary of file_path: expr entries
        file_exprs = {}
        for r in results:
            expr = r[1] + '=' + 'table' if r[2] == 'table' and cnts[r[0]] < 2 else r[1] + '=' + r[2] + '.' + r[3]
            try:
                file_exprs[r[0]] += ',' + expr
            except KeyError:
                file_exprs[r[0]] = expr

        # are there any gene annotations?
        are_genes = 'gs://annotationdb/gene/gene.kt' in file_exprs #any([x.startswith('gs://annotationdb/gene/') for x in file_exprs])

        # subset to VEP annotations
        veps = any([x == 'vep' for x in file_exprs])

        # if VEP annotations are selected, or if gene-level annotations are selected with no specified gene_key, annotate with VEP
        if veps or (are_genes and not gene_key):

            # VEP annotate the VDS
            self = self.vep(config='/vep/vep-gcloud.properties', root='va.vep')

            # extract 1 gene symbol per variant from VEP annotations if a gene_key parameter isn't provided
            if are_genes:

                # hierarchy of possible variant consequences, from most to least severe
                csq_terms = [
                    'transcript_ablation',
                    'splice_acceptor_variant',
                    'splice_donor_variant',
                    'stop_gained',
                    'frameshift_variant',
                    'stop_lost',
                    'start_lost',
                    'transcript_amplification',
                    'inframe_insertion',
                    'inframe_deletion',
                    'missense_variant',
                    'protein_altering_variant',
                    'incomplete_terminal_codon_variant',
                    'stop_retained_variant',
                    'synonymous_variant',
                    'splice_region_variant',
                    'coding_sequence_variant',
                    'mature_miRNA_variant',
                    '5_prime_UTR_variant',
                    '3_prime_UTR_variant',
                    'non_coding_transcript_exon_variant',
                    'intron_variant',
                    'NMD_transcript_variant',
                    'non_coding_transcript_variant',
                    'upstream_gene_variant',
                    'downstream_gene_variant',
                    'TFBS_ablation',
                    'TFBS_amplification',
                    'TF_binding_site_variant',
                    'regulatory_region_ablation',
                    'regulatory_region_amplification',
                    'feature_elongation',
                    'regulatory_region_variant',
                    'feature_truncation',
                    'intergenic_variant'
                ]

                # add consequence terms as a global annotation
                self = self.annotate_global('global.csq_terms', csq_terms, TArray(TString()))

                # find 1 transcript/gene per variant using the following method:
                #   1. define the most severe consequence for each variant according to hierarchy
                #   2. subset to transcripts with that most severe consequence
                #   3. if one of the transcripts in the subset is canonical, take that gene/transcript,
                #      else just take the first gene/transcript in the subset
                self = (
                    self
                    .annotate_variants_expr(
                        """
                        va.gene.most_severe_consequence = 
                            let canonical_consequences = va.vep.transcript_consequences.filter(t => t.canonical == 1).flatMap(t => t.consequence_terms).toSet() in
                            if (isDefined(canonical_consequences))
                                orElse(global.csq_terms.find(c => canonical_consequences.contains(c)), 
                                       va.vep.most_severe_consequence)
                            else
                                va.vep.most_severe_consequence
                        """
                    )
                    .annotate_variants_expr(
                        """
                        va.gene.transcript = let tc = va.vep.transcript_consequences.filter(t => t.consequence_terms.toSet.contains(va.gene.most_severe_consequence)) in 
                                             orElse(tc.find(t => t.canonical == 1), tc[0])
                        """
                    )
                )

            # drop VEP annotations if not specifically requested
            if not veps:
                self = self.annotate_variants_expr('va = drop(va, vep)')

            # subset VEP annotations if needed
            subset = ','.join([x.rsplit('.')[-1] for x in annotations if x.startswith('va.vep.')])
            if subset:
                self = self.annotate_variants_expr('va.vep = select(va.vep, {})'.format(subset))

        # iterate through files, selected annotations from each file
        for db_file, expr in file_exprs.iteritems():

            # if database file is a VDS
            if db_file.endswith('.vds'):

                # annotate analysis VDS with database VDS
                self = self.annotate_variants_vds(self.hc.read(db_file), expr=expr)

            # if database file is a keytable
            elif db_file.endswith('.kt'):

                # join on gene symbol for gene annotations
                if db_file == 'gs://annotationdb/gene/gene.kt':
                    if gene_key:
                        vds_key = gene_key
                    else:
                        vds_key = 'va.gene.transcript.gene_symbol'
                else:
                    vds_key = None

                # annotate analysis VDS with database keytable
                self = self.annotate_variants_table(self.hc.read_table(db_file), expr=expr, vds_key=vds_key)

            else:
                continue

        return self

    @handle_py4j
    def cache(self):
        """Mark this variant dataset to be cached in memory.

        :py:meth:`~hail.VariantDataset.cache` is the same as :func:`persist("MEMORY_ONLY") <hail.VariantDataset.persist>`.
        
        :rtype: :class:`.VariantDataset`
        """

        return VariantDataset(self.hc, self._jvdf.cache())

    @handle_py4j
    @requireTGenotype
    @typecheck_method(right=vds_type)
    def concordance(self, right):
        """Calculate call concordance with another variant dataset.

        .. include:: requireTGenotype.rst

        **Example**
        
        >>> comparison_vds = hc.read('data/example2.vds')
        >>> summary, samples, variants = vds.concordance(comparison_vds)

        **Notes**

        This method computes the genotype call concordance between two bialellic variant datasets. 
        It performs an inner join on samples (only samples in both datasets will be considered), and an outer join
        on variants. If a variant is only in one dataset, then each genotype is treated as "no data" in the other.
        This method returns a tuple of three objects: a nested list of list of int with global concordance
        summary statistics, a key table with sample concordance statistics, and a key table with variant concordance 
        statistics.
        
        **Using the global summary result**
        
        The global summary is a list of list of int (conceptually a 5 by 5 matrix), 
        where the indices have special meaning:

        0. No Data (missing variant)
        1. No Call (missing genotype call)
        2. Hom Ref
        3. Heterozygous
        4. Hom Var
        
        The first index is the state in the left dataset (the one on which concordance was called), and the second
        index is the state in the right dataset (the argument to the concordance method call). Typical uses of 
        the summary list are shown below.
          
        >>> summary, samples, variants = vds.concordance(hc.read('data/example2.vds'))
        >>> left_homref_right_homvar = summary[2][4]
        >>> left_het_right_missing = summary[3][1]
        >>> left_het_right_something_else = sum(summary[3][:]) - summary[3][3]
        >>> total_concordant = summary[2][2] + summary[3][3] + summary[4][4]
        >>> total_discordant = sum([sum(s[2:]) for s in summary[2:]]) - total_concordant
        
        **Using the key table results**
        
        Columns of the sample key table:
        
           - **s** (*String*) -- Sample ID, key column.
           - **nDiscordant** (*Long*) -- Count of discordant calls (see below for full definition).
           - **concordance** (*Array[Array[Long]]*) -- Array of concordance per state on left and right,
             matching the structure of the global summary defined above.
             
        Columns of the variant key table:
        
           - **v** (*Variant*) -- Key column.
           - **nDiscordant** (*Long*) -- Count of discordant calls (see below for full definition).
           - **concordance** (*Array[Array[Long]]*) -- Array of concordance per state on left and right,
             matches the structure of the global summary defined above.
             
        The two key tables produced by the concordance method can be queried with :py:meth:`.KeyTable.query`, 
        exported to text with :py:meth:`.KeyTable.export`, and used to annotate a variant dataset with
        :py:meth:`.VariantDataset.annotate_variants_table`, among other things.
        
        In these tables, the column **nDiscordant** is provided as a convenience, because this is often one
        of the most useful concordance statistics. This value is the number of genotypes 
        which were called (homozygous reference, heterozygous, or homozygous variant) in both datasets, 
        but where the call did not match between the two.
        
        The column **concordance** matches the structure of the global summmary, which is detailed above. Once again,
        the first index into this array is the state on the left, and the second index is the state on the right.
        For example, ``concordance[1][4]`` is the number of "no call" genotypes on the left that were called 
        homozygous variant on the right. 
        
        :param right: right hand variant dataset for concordance
        :type right: :class:`.VariantDataset`

        :return: The global concordance statistics, a key table with sample concordance
            statistics, and a key table with variant concordance statistics.
        :rtype: (list of list of int, :py:class:`.KeyTable`, :py:class:`.KeyTable`)
        """

        r = self._jvdf.concordance(right._jvds)
        j_global_concordance = r._1()
        sample_kt = KeyTable(self.hc, r._2())
        variant_kt = KeyTable(self.hc, r._3())
        global_concordance = [[j_global_concordance.apply(j).apply(i) for i in xrange(5)] for j in xrange(5)]

        return global_concordance, sample_kt, variant_kt

    @handle_py4j
    def count(self):
        """Returns number of samples and variants in the dataset.
        
        **Examples**
        
        >>> samples, variants = vds.count()
        
        **Notes**
        
        This is also the fastest way to force evaluation of a Hail pipeline.
        
        :returns: The sample and variant counts.
        :rtype: (int, int)
        """

        r = self._jvds.count()

        return r._1(), r._2()

    @handle_py4j
    def deduplicate(self):
        """Remove duplicate variants.

        :return: Deduplicated variant dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        return VariantDataset(self.hc, self._jvds.deduplicate())

    @handle_py4j
    @typecheck_method(fraction=numeric,
                      seed=integral)
    def sample_variants(self, fraction, seed=1):
        """Downsample variants to a given fraction of the dataset.
        
        **Examples**
        
        >>> small_vds = vds.sample_variants(0.01)
        
        **Notes**
        
        This method may not sample exactly ``(fraction * n_variants)``
        variants from the dataset.

        :param float fraction: (Expected) fraction of variants to keep.

        :param int seed: Random seed.

        :return: Downsampled variant dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        return VariantDataset(self.hc, self._jvds.sampleVariants(fraction, seed))

    @handle_py4j
    @requireTGenotype
    @typecheck_method(output=strlike,
                      precision=integral)
    def export_gen(self, output, precision=4):
        """Export variant dataset as GEN and SAMPLE file.

        .. include:: requireTGenotype.rst

        **Examples**

        Import genotype probability data, filter variants based on INFO score, and export data to a GEN and SAMPLE file:

        >>> vds3 = hc.import_bgen("data/example3.bgen", sample_file="data/example3.sample")

        >>> (vds3.filter_variants_expr("gs.infoScore().score >= 0.9")
        ...      .export_gen("output/infoscore_filtered"))

        **Notes**

        Writes out the internal VDS to a GEN and SAMPLE fileset in the `Oxford spec <http://www.stats.ox.ac.uk/%7Emarchini/software/gwas/file_format.html>`__.

        The first 6 columns of the resulting GEN file are the following:

        - Chromosome (``v.contig``)
        - Variant ID (``va.varid`` if defined, else Chromosome:Position:Ref:Alt)
        - rsID (``va.rsid`` if defined, else ".")
        - position (``v.start``)
        - reference allele (``v.ref``)
        - alternate allele (``v.alt``)

        Genotype probabilities:

        - 3 probabilities per sample ``(pHomRef, pHet, pHomVar)``.
        - Any filtered genotypes will be output as ``(0.0, 0.0, 0.0)``.
        - If the input data contained Phred-scaled likelihoods, the probabilities in the GEN file will be the normalized genotype probabilities assuming a uniform prior.
        - If the input data did not have genotype probabilities such as data imported using :py:meth:`~hail.HailContext.import_plink`, all genotype probabilities will be ``(0.0, 0.0, 0.0)``.

        The sample file has 3 columns:

        - ID_1 and ID_2 are identical and set to the sample ID (``s``).
        - The third column ("missing") is set to 0 for all samples.

        :param str output: Output file base.  Will write GEN and SAMPLE files.
        :param int precision: Number of digits after the decimal point each probability is truncated to.
        """

        self._jvdf.exportGen(output, precision)

    @handle_py4j
    @typecheck_method(output=strlike,
                      expr=strlike,
                      types=bool,
                      export_ref=bool,
                      export_missing=bool,
                      parallel=bool)
    def export_genotypes(self, output, expr, types=False, export_ref=False, export_missing=False, parallel=False):
        """Export genotype-level information to delimited text file.

        **Examples**

        Export genotype information with identifiers that form the header:

        >>> vds.export_genotypes('output/genotypes.tsv', 'SAMPLE=s, VARIANT=v, GQ=g.gq, DP=g.dp, ANNO1=va.anno1, ANNO2=va.anno2')

        Export the same information without identifiers, resulting in a file with no header:

        >>> vds.export_genotypes('output/genotypes.tsv', 's, v, g.gq, g.dp, va.anno1, va.anno2')

        **Notes**

        :py:meth:`~hail.VariantDataset.export_genotypes` outputs one line per cell (genotype) in the data set, though HomRef and missing genotypes are not output by default if the genotype schema is equal to :py:class:`~hail.expr.TGenotype`. Use the ``export_ref`` and ``export_missing`` parameters to force export of HomRef and missing genotypes, respectively.

        The ``expr`` argument is a comma-separated list of fields or expressions, all of which must be of the form ``IDENTIFIER = <expression>``, or else of the form ``<expression>``.  If some fields have identifiers and some do not, Hail will throw an exception. The accessible namespace includes ``g``, ``s``, ``sa``, ``v``, ``va``, and ``global``.

        .. warning::

            If the genotype schema does not have the type :py:class:`~hail.expr.TGenotype`, all genotypes will be exported unless the value of ``g`` is missing.
            Use :py:meth:`~hail.VariantDataset.filter_genotypes` to filter out genotypes based on an expression before exporting.

        :param str output: Output path.

        :param str expr: Export expression for values to export.

        :param bool types: Write types of exported columns to a file at (output + ".types")

        :param bool export_ref: If true, export reference genotypes. Only applicable if the genotype schema is :py:class:`~hail.expr.TGenotype`.

        :param bool export_missing: If true, export missing genotypes.

        :param bool parallel: If true, writes a set of files (one per partition) rather than serially concatenating these files.
        """

        if self._is_generic_genotype:
            self._jvdf.exportGenotypes(output, expr, types, export_missing, parallel)
        else:
            self._jvdf.exportGenotypes(output, expr, types, export_ref, export_missing, parallel)

    @handle_py4j
    @requireTGenotype
    @typecheck_method(output=strlike,
                      fam_expr=strlike)
    def export_plink(self, output, fam_expr='id = s'):
        """Export variant dataset as `PLINK2 <https://www.cog-genomics.org/plink2/formats>`__ BED, BIM and FAM.

        .. include:: requireTGenotype.rst

        **Examples**

        Import data from a VCF file, split multi-allelic variants, and export to a PLINK binary file:

        >>> vds.split_multi().export_plink('output/plink')

        **Notes**

        ``fam_expr`` can be used to set the fields in the FAM file.
        The following fields can be assigned:

        - ``famID: String``
        - ``id: String``
        - ``matID: String``
        - ``patID: String``
        - ``isFemale: Boolean``
        - ``isCase: Boolean`` or ``qPheno: Double``

        If no assignment is given, the value is missing and the
        missing value is used: ``0`` for IDs and sex and ``-9`` for
        phenotype.  Only one of ``isCase`` or ``qPheno`` can be
        assigned.

        ``fam_expr`` is in sample context only and the following
        symbols are in scope:

        - ``s`` (*Sample*): sample
        - ``sa``: sample annotations
        - ``global``: global annotations

        The BIM file ID field is set to ``CHR:POS:REF:ALT``.

        This code:

        >>> vds.split_multi().export_plink('output/plink')

        will behave similarly to the PLINK VCF conversion command

        .. code-block:: text

            plink --vcf /path/to/file.vcf --make-bed --out sample --const-fid --keep-allele-order

        except:

        - The order among split multi-allelic alternatives in the BED
          file may disagree.
        - PLINK uses the rsID for the BIM file ID.

        :param str output: Output file base.  Will write BED, BIM, and FAM files.

        :param str fam_expr: Expression for FAM file fields.
        """

        self._jvdf.exportPlink(output, fam_expr)

    @handle_py4j
    @typecheck_method(output=strlike,
                      expr=strlike,
                      types=bool)
    def export_samples(self, output, expr, types=False):
        """Export sample information to delimited text file.

        **Examples**

        Export some sample QC metrics:

        >>> (vds.sample_qc()
        ...     .export_samples('output/samples.tsv', 'SAMPLE = s, CALL_RATE = sa.qc.callRate, NHET = sa.qc.nHet'))

        This will produce a file with a header and three columns.  To
        produce a file with no header, just leave off the assignment
        to the column identifier:

        >>> (vds.sample_qc()
        ...     .export_samples('output/samples.tsv', 's, sa.qc.rTiTv'))

        **Notes**

        One line per sample will be exported.  As :py:meth:`~hail.VariantDataset.export_samples` runs in sample context, the following symbols are in scope:

        - ``s`` (*Sample*): sample
        - ``sa``: sample annotations
        - ``global``: global annotations
        - ``gs`` (*Aggregable[Genotype]*): aggregable of :ref:`genotype` for sample ``s``

        :param str output: Output file.

        :param str expr: Export expression for values to export.

        :param bool types: Write types of exported columns to a file at (output + ".types").
        """

        self._jvds.exportSamples(output, expr, types)

    @handle_py4j
    @typecheck_method(output=strlike,
                      expr=strlike,
                      types=bool,
                      parallel=bool)
    def export_variants(self, output, expr, types=False, parallel=False):
        """Export variant information to delimited text file.

        **Examples**

        Export a four column TSV with ``v``, ``va.pass``, ``va.filters``, and
        one computed field: ``1 - va.qc.callRate``.

        >>> vds.export_variants('output/file.tsv',
        ...        'VARIANT = v, PASS = va.pass, FILTERS = va.filters, MISSINGNESS = 1 - va.qc.callRate')

        It is also possible to export without identifiers, which will result in
        a file with no header. In this case, the expressions should look like
        the examples below:

        >>> vds.export_variants('output/file.tsv', 'v, va.pass, va.qc.AF')

        .. note::

            If any field is named, all fields must be named.

        In the common case that a group of annotations needs to be exported (for
        example, the annotations produced by ``variantqc``), one can use the
        ``struct.*`` syntax.  This syntax produces one column per field in the
        struct, and names them according to the struct field name.

        For example, the following invocation (assuming ``va.qc`` was generated
        by :py:meth:`.variant_qc`):

        >>> vds.export_variants('output/file.tsv', 'variant = v, va.qc.*')

        will produce the following set of columns:

        .. code-block:: text

            variant  callRate  AC  AF  nCalled  ...

        Note that using the ``.*`` syntax always results in named arguments, so it
        is not possible to export header-less files in this manner.  However,
        naming the "splatted" struct will apply the name in front of each column
        like so:

        >>> vds.export_variants('output/file.tsv', 'variant = v, QC = va.qc.*')

        which produces these columns:

        .. code-block:: text

            variant  QC.callRate  QC.AC  QC.AF  QC.nCalled  ...


        **Notes**

        This module takes a comma-delimited list of fields or expressions to
        print. These fields will be printed in the order they appear in the
        expression in the header and on each line.

        One line per variant in the VDS will be printed.  The accessible namespace includes:

        - ``v`` (*Variant*): :ref:`variant`
        - ``va``: variant annotations
        - ``global``: global annotations
        - ``gs`` (*Aggregable[Genotype]*): aggregable of :ref:`genotype` for variant ``v``

        **Designating output with an expression**

        Much like the filtering methods, this method uses the Hail expression language.
        While the filtering methods expect an
        expression that evaluates to true or false, this method expects a
        comma-separated list of fields to print. These fields take the
        form ``IDENTIFIER = <expression>``.


        :param str output: Output file.

        :param str expr: Export expression for values to export.

        :param bool types: Write types of exported columns to a file at (output + ".types")

        :param bool parallel: If true, writes a set of files (one per partition) rather than serially concatenating these files.
        """

        self._jvds.exportVariants(output, expr, types, parallel)

    @handle_py4j
    @typecheck_method(output=strlike,
                      append_to_header=nullable(strlike),
                      export_pp=bool,
                      parallel=bool)
    def export_vcf(self, output, append_to_header=None, export_pp=False, parallel=False):
        """Export variant dataset as a .vcf or .vcf.bgz file.

        **Examples**

        Export to VCF as a block-compressed file:

        >>> vds.export_vcf('output/example.vcf.bgz')

        **Notes**

        :py:meth:`~hail.VariantDataset.export_vcf` writes the VDS to disk in VCF format as described in the `VCF 4.2 spec <https://samtools.github.io/hts-specs/VCFv4.2.pdf>`__.

        Use the ``.vcf.bgz`` extension rather than ``.vcf`` in the output file name for `blocked GZIP <http://www.htslib.org/doc/tabix.html>`__ compression.

        .. note::

            We strongly recommended compressed (``.bgz`` extension) and parallel output (``parallel=True``) when exporting large VCFs.

        Consider the workflow of importing VCF to VDS and immediately exporting VDS to VCF:

        >>> vds.export_vcf('output/example_out.vcf')

        The *example_out.vcf* header will contain the FORMAT, FILTER, and INFO lines present in *example.vcf*. However, it will *not* contain CONTIG lines or lines added by external tools (such as bcftools and GATK) unless they are explicitly inserted using the ``append_to_header`` option.

        Hail only exports the contents of ``va.info`` to the INFO field. No other annotations besides ``va.info`` are exported.

        The genotype schema must have the type :py:class:`~hail.expr.TGenotype` or :py:class:`~hail.expr.TStruct`. If the type is
        :py:class:`~hail.expr.TGenotype`, then the FORMAT fields will be GT, AD, DP, GQ, and PL (or PP if ``export_pp`` is True).
        If the type is :py:class:`~hail.expr.TStruct`, then the exported FORMAT fields will be the names of each field of the Struct.
        Each field must have a type of String, Char, Int, Double, or Call. Arrays and Sets are also allowed as long as they are not nested.
        For example, a field with type ``Array[Int]`` can be exported but not a field with type ``Array[Array[Int]]``.
        Nested Structs are also not allowed.

        .. caution::

            If samples or genotypes are filtered after import, the value stored in ``va.info.AC`` value may no longer reflect the number of called alternate alleles in the filtered VDS. If the filtered VDS is then exported to VCF, downstream tools may produce erroneous results. The solution is to create new annotations in ``va.info`` or overwrite existing annotations. For example, in order to produce an accurate ``AC`` field, one can run :py:meth:`~hail.VariantDataset.variant_qc` and copy the ``va.qc.AC`` field to ``va.info.AC``:

            >>> (vds.filter_genotypes('g.gq >= 20')
            ...     .variant_qc()
            ...     .annotate_variants_expr('va.info.AC = va.qc.AC')
            ...     .export_vcf('output/example.vcf.bgz'))

        :param str output: Path of .vcf file to write.

        :param append_to_header: Path of file to append to VCF header.
        :type append_to_header: str or None

        :param bool export_pp: If true, export linear-scaled probabilities (Hail's `pp` field on genotype) as the VCF PP FORMAT field.

        :param bool parallel: If true, return a set of VCF files (one per partition) rather than serially concatenating these files.
        """

        self._jvdf.exportVCF(output, joption(append_to_header), export_pp, parallel)

    @handle_py4j
    @convertVDS
    @typecheck_method(output=strlike,
                      overwrite=bool,
                      parquet_genotypes=bool)
    def write(self, output, overwrite=False, parquet_genotypes=False):
        """Write variant dataset as VDS file.

        **Examples**

        Import data from a VCF file and then write the data to a VDS file:

        >>> vds.write("output/sample.vds")

        :param str output: Path of VDS file to write.

        :param bool overwrite: If true, overwrite any existing VDS file. Cannot be used to read from and write to the same path.

        :param bool parquet_genotypes: If true, store genotypes as Parquet rather than Hail's serialization.  The resulting VDS will be larger and slower in Hail but the genotypes will be accessible from other tools that support Parquet.

        """

        if self._is_generic_genotype:
            self._jvdf.write(output, overwrite)
        else:
            self._jvdf.write(output, overwrite, parquet_genotypes)

    @handle_py4j
    @requireTGenotype
    @typecheck_method(expr=strlike,
                      annotation=strlike,
                      subset=bool,
                      keep=bool,
                      filter_altered_genotypes=bool,
                      max_shift=integral,
                      keep_star=bool)
    def filter_alleles(self, expr, annotation='va = va', subset=True, keep=True,
                       filter_altered_genotypes=False, max_shift=100, keep_star=False):
        """Filter a user-defined set of alternate alleles for each variant.
        If all alternate alleles of a variant are filtered, the
        variant itself is filtered.  The expr expression is
        evaluated for each alternate allele, but not for
        the reference allele (i.e. ``aIndex`` will never be zero).

        .. include:: requireTGenotype.rst

        **Examples**

        To remove alternate alleles with zero allele count and
        update the alternate allele count annotation with the new
        indices:

        >>> vds_result = vds.filter_alleles('va.info.AC[aIndex - 1] == 0',
        ...     annotation='va.info.AC = aIndices[1:].map(i => va.info.AC[i - 1])',
        ...     keep=False)

        Note that we skip the first element of ``aIndices`` because
        we are mapping between the old and new *allele* indices, not
        the *alternate allele* indices.

        **Notes**

        If ``filter_altered_genotypes`` is true, genotypes that contain filtered-out alleles are set to missing.

        :py:meth:`~hail.VariantDataset.filter_alleles` implements two algorithms for filtering alleles: subset and downcode. We will illustrate their
        behavior on the example genotype below when filtering the first alternate allele (allele 1) at a site with 1 reference
        allele and 2 alternate alleles.

        .. code-block:: text

          GT: 1/2
          GQ: 10
          AD: 0,50,35

          0 | 1000
          1 | 1000   10
          2 | 1000   0     20
            +-----------------
               0     1     2

        **Subset algorithm**

        The subset algorithm (the default, ``subset=True``) subsets the
        AD and PL arrays (i.e. removes entries corresponding to filtered alleles)
        and then sets GT to the genotype with the minimum PL.  Note
        that if the genotype changes (as in the example), the PLs
        are re-normalized (shifted) so that the most likely genotype has a PL of
        0.  Qualitatively, subsetting corresponds to the belief
        that the filtered alleles are not real so we should discard any
        probability mass associated with them.

        The subset algorithm would produce the following:

        .. code-block:: text

          GT: 1/1
          GQ: 980
          AD: 0,50

          0 | 980
          1 | 980    0
            +-----------
               0      1

        In summary:

        - GT: Set to most likely genotype based on the PLs ignoring the filtered allele(s).
        - AD: The filtered alleles' columns are eliminated, e.g., filtering alleles 1 and 2 transforms ``25,5,10,20`` to ``25,20``.
        - DP: No change.
        - PL: The filtered alleles' columns are eliminated and the remaining columns shifted so the minimum value is 0.
        - GQ: The second-lowest PL (after shifting).

        **Downcode algorithm**

        The downcode algorithm (``subset=False``) recodes occurances of filtered alleles
        to occurances of the reference allele (e.g. 1 -> 0 in our example). So the depths of filtered alleles in the AD field
        are added to the depth of the reference allele. Where downcodeing filtered alleles merges distinct genotypes, the minimum PL is used (since PL is on a log scale, this roughly corresponds to adding probabilities). The PLs
        are then re-normalized (shifted) so that the most likely genotype has a PL of 0, and GT is set to this genotype.
        If an allele is filtered, this algorithm acts similarly to :py:meth:`~hail.VariantDataset.split_multi`.

        The downcoding algorithm would produce the following:

        .. code-block:: text

          GT: 0/1
          GQ: 10
          AD: 35,50

          0 | 20
          1 | 0    10
            +-----------
              0    1

        In summary:

        - GT: Downcode filtered alleles to reference.
        - AD: The filtered alleles' columns are eliminated and their value is added to the reference, e.g., filtering alleles 1 and 2 transforms ``25,5,10,20`` to ``40,20``.
        - DP: No change.
        - PL: Downcode filtered alleles to reference, combine PLs using minimum for each overloaded genotype, and shift so the overall minimum PL is 0.
        - GQ: The second-lowest PL (after shifting).

        **Expression Variables**

        The following symbols are in scope for ``expr``:

        - ``v`` (*Variant*): :ref:`variant`
        - ``va``: variant annotations
        - ``aIndex`` (*Int*): the index of the allele being tested

        The following symbols are in scope for ``annotation``:

        - ``v`` (*Variant*): :ref:`variant`
        - ``va``: variant annotations
        - ``aIndices`` (*Array[Int]*): the array of old indices (such that ``aIndices[newIndex] = oldIndex`` and ``aIndices[0] = 0``)

        :param str expr: Boolean filter expression involving v (variant), va (variant annotations), 
            and aIndex (allele index)

        :param str annotation: Annotation modifying expression involving v (new variant), va (old variant annotations),
            and aIndices (maps from new to old indices)

        :param bool subset: If true, subsets PL and AD, otherwise downcodes the PL and AD.
            Genotype and GQ are set based on the resulting PLs.

        :param bool keep: If true, keep variants matching expr

        :param bool filter_altered_genotypes: If true, genotypes that contain filtered-out alleles are set to missing.

        :param int max_shift: maximum number of base pairs by which
            a split variant can move.  Affects memory usage, and will
            cause Hail to throw an error if a variant that moves further
            is encountered.

        :param bool keepStar: If true, keep variants where the only allele left is a ``*`` allele.

        :return: Filtered variant dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        jvds = self._jvdf.filterAlleles(expr, annotation, filter_altered_genotypes, keep, subset, max_shift,
                                        keep_star)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @typecheck_method(expr=strlike,
                      keep=bool)
    def filter_genotypes(self, expr, keep=True):
        """Filter genotypes based on expression.

        **Examples**

        Filter genotypes by allele balance dependent on genotype call:

        >>> vds_result = vds.filter_genotypes('let ab = g.ad[1] / g.ad.sum() in ' +
        ...                      '((g.isHomRef() && ab <= 0.1) || ' +
        ...                      '(g.isHet() && ab >= 0.25 && ab <= 0.75) || ' +
        ...                      '(g.isHomVar() && ab >= 0.9))')

        **Notes**

        ``expr`` is in genotype context so the following symbols are in scope:

        - ``s`` (*Sample*): sample
        - ``v`` (*Variant*): :ref:`variant`
        - ``sa``: sample annotations
        - ``va``: variant annotations
        - ``global``: global annotations

        For more information, see the documentation on `data representation, annotations <overview.html#>`__, and
        the `expression language <exprlang.html>`__.

        .. caution::
            When ``expr`` evaluates to missing, the genotype will be removed regardless of whether ``keep=True`` or ``keep=False``.

        :param str expr: Boolean filter expression.
        
        :param bool keep: Keep genotypes where ``expr`` evaluates to true.

        :return: Filtered variant dataset.
        :rtype: :class:`.VariantDataset`
        """

        jvds = self._jvdf.filterGenotypes(expr, keep)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @requireTGenotype
    def filter_multi(self):
        """Filter out multi-allelic sites.

        .. include:: requireTGenotype.rst

        This method is much less computationally expensive than
        :py:meth:`.split_multi`, and can also be used to produce
        a variant dataset that can be used with methods that do not
        support multiallelic variants.

        :return: Dataset with no multiallelic sites, which can
            be used for biallelic-only methods.
        :rtype: :class:`.VariantDataset`
        """

        return VariantDataset(self.hc, self._jvdf.filterMulti())

    @handle_py4j
    def drop_samples(self):
        """Removes all samples from variant dataset.

        The variants, variant annotations, and global annnotations will remain,
        producing a sites-only variant dataset.

        :return: Sites-only variant dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        return VariantDataset(self.hc, self._jvds.dropSamples())

    @handle_py4j
    @typecheck_method(expr=strlike,
                      keep=bool)
    def filter_samples_expr(self, expr, keep=True):
        """Filter samples with the expression language.

        **Examples**

        Filter samples by phenotype (assumes sample annotation *sa.isCase* exists and is a Boolean variable):

        >>> vds_result = vds.filter_samples_expr("sa.isCase")

        Remove samples with an ID that matches a regular expression:

        >>> vds_result = vds.filter_samples_expr('"^NA" ~ s' , keep=False)

        Filter samples from sample QC metrics and write output to a new variant dataset:

        >>> (vds.sample_qc()
        ...     .filter_samples_expr('sa.qc.callRate >= 0.99 && sa.qc.dpMean >= 10')
        ...     .write("output/filter_samples.vds"))

        **Notes**

        ``expr`` is in sample context so the following symbols are in scope:

        - ``s`` (*Sample*): sample
        - ``sa``: sample annotations
        - ``global``: global annotations
        - ``gs`` (*Aggregable[Genotype]*): aggregable of :ref:`genotype` for sample ``s``

        For more information, see the documentation on `data representation, annotations <overview.html#>`__, and
        the `expression language <exprlang.html>`__.

        .. caution::
            When ``expr`` evaluates to missing, the sample will be removed regardless of whether ``keep=True`` or ``keep=False``.


        :param str expr: Boolean filter expression.
        
        :param bool keep: Keep samples where ``expr`` evaluates to true.

        :return: Filtered variant dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        jvds = self._jvds.filterSamplesExpr(expr, keep)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @typecheck_method(samples=listof(strlike),
                      keep=bool)
    def filter_samples_list(self, samples, keep=True):
        """Filter samples with a list of samples.
    
        **Examples**
    
        >>> to_remove = ['NA12878', 'NA12891', 'NA12892']
        >>> vds_result = vds.filter_samples_list(to_remove, keep=False)
        
        Read list from a file:
        
        >>> to_remove = [s.strip() for s in open('data/exclude_samples.txt')]
        >>> vds_result = vds.filter_samples_list(to_remove, keep=False)
    
        :param samples: List of samples to keep or remove.
        :type samples: list of str

        :param bool keep: If true, keep samples in ``samples``, otherwise remove them.
    
        :return: Filtered variant dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        return VariantDataset(self.hc, self._jvds.filterSamplesList(samples, keep))

    @handle_py4j
    @typecheck_method(table=KeyTable,
                      keep=bool)
    def filter_samples_table(self, table, keep=True):
        """Filter samples with a table keyed by sample ID.
        
        **Examples**
        
        Keep samples in a text file:
        
        >>> table = hc.import_table('data/samples1.tsv').key_by('Sample')
        >>> vds_filtered = vds.filter_samples_table(table, keep=True)
        
        Remove samples in a text file with 1 field, and no header:
        
        >>> to_remove = hc.import_table('data/exclude_samples.txt', no_header=True).key_by('f0')
        >>> vds_filtered = vds.filter_samples_table(to_remove, keep=False)
        
        **Notes**
        
        This method filters out or filters to the keys of a table. The table must have a key of 
        type ``String``. 
        
        :param table: Key table.
        :type table: :class:`.KeyTable`
        
        :param bool keep: If true, keep only the keys in ``table``, otherwise remove them.
        
        :return: Filtered dataset.
        :rtype: :class:`.VariantDataset`
        """

        return VariantDataset(self.hc, self._jvds.filterSamplesTable(table._jkt, keep))

    @handle_py4j
    def drop_variants(self):
        """Discard all variants, variant annotations and genotypes.

        Samples, sample annotations and global annotations are retained. This
        is the same as :func:`filter_variants_expr('false') <hail.VariantDataset.filter_variants_expr>`, but much faster.

        **Examples**

        >>> vds_result = vds.drop_variants()

        :return: Samples-only variant dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        return VariantDataset(self.hc, self._jvds.dropVariants())

    @handle_py4j
    @typecheck_method(expr=strlike,
                      keep=bool)
    def filter_variants_expr(self, expr, keep=True):
        """Filter variants with the expression language.

        **Examples**

        Keep variants in the gene CHD8 (assumes the variant annotation ``va.gene`` exists):

        >>> vds_result = vds.filter_variants_expr('va.gene == "CHD8"')


        Remove all variants on chromosome 1:

        >>> vds_result = vds.filter_variants_expr('v.contig == "1"', keep=False)

        .. caution::

           The double quotes on ``"1"`` are necessary because ``v.contig`` is of type String.

        **Notes**

        The following symbols are in scope for ``expr``:

        - ``v`` (*Variant*): :ref:`variant`
        - ``va``: variant annotations
        - ``global``: global annotations
        - ``gs`` (*Aggregable[Genotype]*): aggregable of :ref:`genotype` for variant ``v``

        For more information, see the `Overview <overview.html#>`__ and the `Expression Language <exprlang.html>`__.

        .. caution::
           When ``expr`` evaluates to missing, the variant will be removed regardless of whether ``keep=True`` or ``keep=False``.

        :param str expr: Boolean filter expression.

        :param bool keep: Keep variants where ``expr`` evaluates to true.
        
        :return: Filtered variant dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        jvds = self._jvds.filterVariantsExpr(expr, keep)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @typecheck_method(intervals=oneof(Interval, listof(Interval)),
                      keep=bool)
    def filter_intervals(self, intervals, keep=True):
        """Filter variants with an interval or list of intervals.

        **Examples**

        Filter to one interval:
        
        >>> vds_result = vds.filter_intervals(Interval.parse('17:38449840-38530994'))
        
        Another way of writing this same query:
        
        >>> vds_result = vds.filter_intervals(Interval(Locus('17', 38449840), Locus('17', 38530994)))
        
        Two identical ways of parsing a list of intervals:
        
        >>> intervals = map(Interval.parse, ['1:50M-75M', '2:START-400000', '3-22'])
        >>> intervals = [Interval.parse(x) for x in ['1:50M-75M', '2:START-400000', '3-22']]
        
        Use this interval list to filter:
        
        >>> vds_result = vds.filter_intervals(intervals)
        
        **Notes**
        
        This method takes an argument of :class:`.Interval` or list of :class:`.Interval`.

        Based on the ``keep`` argument, this method will either restrict to variants in the
        supplied interval ranges, or remove all variants in those ranges.  Note that intervals
        are left-inclusive, and right-exclusive.  The below interval includes the locus
        ``15:100000`` but not ``15:101000``.

        >>> interval = Interval.parse('15:100000-101000')

        This method performs predicate pushdown when ``keep=True``, meaning that data shards
        that don't overlap any supplied interval will not be loaded at all.  This property
        enables ``filter_intervals`` to be used for reasonably low-latency queries of small ranges
        of the genome, even on large datasets. Suppose we are interested in variants on 
        chromosome 15 between 100000 and 200000. This implementation with :py:meth:`.filter_variants_expr`
        may come to mind first:
        
        >>> vds_filtered = vds.filter_variants_expr('v.contig == "15" && v.start >= 100000 && v.start < 200000')
        
        However, it is **much** faster (and easier!) to use this method:
        
        >>> vds_filtered = vds.filter_intervals(Interval.parse('15:100000-200000'))

        .. note::

            A :py:class:`.KeyTable` keyed by interval can be used to filter a dataset efficiently as well.
            See the documentation for :py:meth:`.filter_variants_table` for an example. This is useful for
            using interval files to filter a dataset.

        :param intervals: Interval(s) to keep or remove.
        :type intervals: :class:`.Interval` or list of :class:`.Interval`

        :param bool keep: Keep variants overlapping an interval if ``True``, remove variants overlapping
                          an interval if ``False``.

        :return: Filtered variant dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        intervals = wrap_to_list(intervals)

        jvds = self._jvds.filterIntervals([x._jrep for x in intervals], keep)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @typecheck_method(variants=listof(Variant),
                      keep=bool)
    def filter_variants_list(self, variants, keep=True):
        """Filter variants with a list of variants.

        **Examples**

        Filter VDS down to a list of variants:

        >>> vds_filtered = vds.filter_variants_list([Variant.parse('20:10626633:G:GC'), 
        ...                                          Variant.parse('20:10019093:A:G')], keep=True)
        
        **Notes**


        This method performs predicate pushdown when ``keep=True``, meaning that data shards
        that don't overlap with any supplied variant will not be loaded at all.  This property
        enables ``filter_variants_list`` to be used for reasonably low-latency queries of one
        or more variants, even on large datasets. 
        
        :param variants: List of variants to keep or remove.
        :type variants: list of :py:class:`~hail.representation.Variant`

        :param bool keep: If true, keep variants in ``variants``, otherwise remove them.

        :return: Filtered variant dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        return VariantDataset(
            self.hc, self._jvds.filterVariantsList(
                [TVariant()._convert_to_j(v) for v in variants], keep))

    @handle_py4j
    @typecheck_method(table=KeyTable,
                      keep=bool)
    def filter_variants_table(self, table, keep=True):
        """Filter variants with a Variant keyed key table.

        **Example**

        Filter variants of a VDS to those appearing in a text file:

        >>> kt = hc.import_table('data/sample_variants.txt', key='Variant', impute=True)
        >>> filtered_vds = vds.filter_variants_table(kt, keep=True)
        
        Keep all variants whose chromosome and position (locus) appear in a file with 
        a chromosome:position column:
        
        >>> kt = hc.import_table('data/locus-table.tsv', impute=True).key_by('Locus')
        >>> filtered_vds = vds.filter_variants_table(kt, keep=True)
        
        Remove all variants which overlap an interval in a UCSC BED file:
        
        >>> kt = KeyTable.import_bed('data/file2.bed')
        >>> filtered_vds = vds.filter_variants_table(kt, keep=False)
        
        **Notes**
        
        This method takes a key table as an argument, which must be keyed by one of the following:
        
            - ``Interval``
            - ``Locus``
            - ``Variant``
            
        If the key is a ``Variant``, then a variant in the dataset will be kept or removed based
        on finding a complete match in the table. Be careful, however: ``1:1:A:T`` does not match 
        ``1:1:A:T,C``, and vice versa. 
        
        If the key is a ``Locus``, then a variant in the dataset will be kept or removed based on 
        finding a locus in the table that matches by chromosome and position.
        
        If the key is an ``Interval``, then a variant in the dataset will be kept or removed based 
        on finding an interval in the table that contains the variant's chromosome and position.

        :param table: Key table object.
        :type table: :py:class:`.KeyTable`

        :param bool keep: If true, keep only matches in ``table``, otherwise remove them.

        :return: Filtered variant dataset.
        :rtype: :py:class:`.VariantDataset`

        """

        return VariantDataset(
            self.hc, self._jvds.filterVariantsTable(table._jkt, keep))

    @property
    @handle_py4j
    def globals(self):

        """Return global annotations as a Python object.

        :return: Dataset global annotations.
        :rtype: :py:class:`~hail.representation.Struct`
        """
        if self._globals is None:
            self._globals = self.global_schema._convert_to_py(self._jvds.globalAnnotation())
        return self._globals

    @handle_py4j
    @requireTGenotype
    def grm(self):
        """Compute the Genetic Relatedness Matrix (GRM).

        .. include:: requireTGenotype.rst

        **Examples**
        
        >>> km = vds.grm()
        
        **Notes**
        
        The genetic relationship matrix (GRM) :math:`G` encodes genetic correlation between each pair of samples. It is defined by :math:`G = MM^T` where :math:`M` is a standardized version of the genotype matrix, computed as follows. Let :math:`C` be the :math:`n \\times m` matrix of raw genotypes in the variant dataset, with rows indexed by :math:`n` samples and columns indexed by :math:`m` bialellic autosomal variants; :math:`C_{ij}` is the number of alternate alleles of variant :math:`j` carried by sample :math:`i`, which can be 0, 1, 2, or missing. For each variant :math:`j`, the sample alternate allele frequency :math:`p_j` is computed as half the mean of the non-missing entries of column :math:`j`. Entries of :math:`M` are then mean-centered and variance-normalized as

        .. math::

          M_{ij} = \\frac{C_{ij}-2p_j}{\sqrt{2p_j(1-p_j)m}},

        with :math:`M_{ij} = 0` for :math:`C_{ij}` missing (i.e. mean genotype imputation). This scaling normalizes genotype variances to a common value :math:`1/m` for variants in Hardy-Weinberg equilibrium and is further motivated in the paper `Patterson, Price and Reich, 2006 <http://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.0020190>`__. (The resulting amplification of signal from the low end of the allele frequency spectrum will also introduce noise for rare variants; common practice is to filter out variants with minor allele frequency below some cutoff.)  The factor :math:`1/m` gives each sample row approximately unit total variance (assuming linkage equilibrium) so that the diagonal entries of the GRM are approximately 1. Equivalently,
        
        .. math::

          G_{ik} = \\frac{1}{m} \\sum_{j=1}^m \\frac{(C_{ij}-2p_j)(C_{kj}-2p_j)}{2 p_j (1-p_j)}  
                
        :return: Genetic Relatedness Matrix for all samples.
        :rtype: :py:class:`KinshipMatrix`
        """

        jkm = self._jvdf.grm()
        return KinshipMatrix(jkm)

    @handle_py4j
    @requireTGenotype
    def hardcalls(self):
        """Drop all genotype fields except the GT field.

        .. include:: requireTGenotype.rst

        A hard-called variant dataset is about two orders of magnitude
        smaller than a standard sequencing dataset. Use this
        method to create a smaller, faster
        representation for downstream processing that only
        requires the GT field.

        :return: Variant dataset with no genotype metadata.
        :rtype: :py:class:`.VariantDataset`
        """

        return VariantDataset(self.hc, self._jvdf.hardCalls())

    @handle_py4j
    @requireTGenotype
    @typecheck_method(maf=nullable(strlike),
                      bounded=bool,
                      min=nullable(numeric),
                      max=nullable(numeric))
    def ibd(self, maf=None, bounded=True, min=None, max=None):
        """Compute matrix of identity-by-descent estimations.

        .. include:: requireTGenotype.rst

        **Examples**

        To calculate a full IBD matrix, using minor allele frequencies computed
        from the variant dataset itself:

        >>> vds.ibd()

        To calculate an IBD matrix containing only pairs of samples with
        ``PI_HAT`` in [0.2, 0.9], using minor allele frequencies stored in
        ``va.panel_maf``:

        >>> vds.ibd(maf='va.panel_maf', min=0.2, max=0.9)

        **Notes**

        The implementation is based on the IBD algorithm described in the `PLINK
        paper <http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1950838>`__.

        :py:meth:`~hail.VariantDataset.ibd` requires the dataset to be
        bi-allelic (otherwise run :py:meth:`~hail.VariantDataset.split_multi` or otherwise run :py:meth:`~hail.VariantDataset.filter_multi`)
        and does not perform LD pruning. Linkage disequilibrium may bias the
        result so consider filtering variants first.

        The resulting :py:class:`.KeyTable` entries have the type: *{ i: String,
        j: String, ibd: { Z0: Double, Z1: Double, Z2: Double, PI_HAT: Double },
        ibs0: Long, ibs1: Long, ibs2: Long }*. The key list is: `*i: String, j:
        String*`.

        Conceptually, the output is a symmetric, sample-by-sample matrix. The
        output key table has the following form

        .. code-block:: text

            i		j	ibd.Z0	ibd.Z1	ibd.Z2	ibd.PI_HAT ibs0	ibs1	ibs2
            sample1	sample2	1.0000	0.0000	0.0000	0.0000 ...
            sample1	sample3	1.0000	0.0000	0.0000	0.0000 ...
            sample1	sample4	0.6807	0.0000	0.3193	0.3193 ...
            sample1	sample5	0.1966	0.0000	0.8034	0.8034 ...

        :param maf: Expression for the minor allele frequency.
        :type maf: str or None

        :param bool bounded: Forces the estimations for Z0, Z1, Z2,
            and PI_HAT to take on biologically meaningful values
            (in the range [0,1]).

        :param min: Sample pairs with a PI_HAT below this value will
            not be included in the output. Must be in [0,1].
        :type min: float or None

        :param max: Sample pairs with a PI_HAT above this value will
            not be included in the output. Must be in [0,1].
        :type max: float or None

        :return: A :py:class:`.KeyTable` mapping pairs of samples to their IBD
            statistics

        :rtype: :py:class:`.KeyTable`

        """

        return KeyTable(self.hc, self._jvdf.ibd(joption(maf), bounded, joption(min), joption(max)))

    @handle_py4j
    @requireTGenotype
    @typecheck_method(threshold=numeric,
                      tiebreaking_expr=nullable(strlike),
                      maf=nullable(strlike),
                      bounded=bool)
    def ibd_prune(self, threshold, tiebreaking_expr=None, maf=None, bounded=True):
        """
        Prune samples from the :py:class:`.VariantDataset` based on :py:meth:`~hail.VariantDataset.ibd` PI_HAT measures of relatedness.

        .. include:: requireTGenotype.rst

        **Examples**
        
        Prune samples so that no two have a PI_HAT value greater than or equal to 0.6.
        
        >>> pruned_vds = vds.ibd_prune(0.6)

        Prune samples so that no two have a PI_HAT value greater than or equal to 0.5, with a tiebreaking expression that 
        selects cases over controls:

        >>> pruned_vds = vds.ibd_prune(0.5, tiebreaking_expr="if (sa1.isCase) 1 else 0")

        **Notes**

        The variant dataset returned may change in near future as a result of algorithmic improvements. The current algorithm is very efficient on datasets with many small
        families, less so on datasets with large families. Currently, the algorithm works by deleting the person from each family who has the highest number of relatives,
        and iterating until no two people have a PI_HAT value greater than that specified. If two people within a family have the same number of relatives, the tiebreaking_expr
        given will be used to determine which sample gets deleted. 
        
        The tiebreaking_expr namespace has the following variables available:
        
        - ``s1``: The first sample id.
        - ``sa1``: The annotations associated with s1.
        - ``s2``: The second sample id. 
        - ``sa2``: The annotations associated with s2. 
        
        The tiebreaking_expr returns an integer expressing the preference for one sample over the other. Any negative integer expresses a preference for keeping ``s1``. Any positive integer expresses a preference for keeping ``s2``. A zero expresses no preference. This function must induce a `preorder <https://en.wikipedia.org/wiki/Preorder>`__ on the samples, in particular:

        - ``tiebreaking_expr(sample1, sample2)`` must equal ``-1 * tie breaking_expr(sample2, sample1)``, which evokes the common sense understanding that if ``x < y`` then `y > x``.
        - ``tiebreaking_expr(sample1, sample1)`` must equal 0, i.e. ``x = x``
        - if sample1 is preferred to sample2 and sample2 is preferred to sample3, then sample1 must also be preferred to sample3

        The last requirement is only important if you have three related samples with the same number of relatives and all three are related to one another. In cases like this one, it is important that either:

        - one of the three is preferred to **both** other ones, or
        - there is no preference among the three samples 

        :param threshold: The desired maximum PI_HAT value between any pair of samples.
        :param tiebreaking_expr: Expression used to choose between two samples with the same number of relatives. 
        :param maf: Expression for the minor allele frequency.
        :param bounded: Forces the estimations for Z0, Z1, Z2, and PI_HAT to take on biologically meaningful values (in the range [0,1]).

        :return: A :py:class:`.VariantDataset` containing no samples with a PI_HAT greater than threshold.
        :rtype: :py:class:`.VariantDataset`
        """
        return VariantDataset(self.hc, self._jvdf.ibdPrune(threshold, joption(tiebreaking_expr), joption(maf), bounded))

    @handle_py4j
    @requireTGenotype
    @typecheck_method(maf_threshold=numeric,
                      include_par=bool,
                      female_threshold=numeric,
                      male_threshold=numeric,
                      pop_freq=nullable(strlike))
    def impute_sex(self, maf_threshold=0.0, include_par=False, female_threshold=0.2, male_threshold=0.8, pop_freq=None):
        """Impute sex of samples by calculating inbreeding coefficient on the
        X chromosome.

        .. include:: requireTGenotype.rst

        **Examples**

        Remove samples where imputed sex does not equal reported sex:

        >>> imputed_sex_vds = (vds.impute_sex()
        ...     .annotate_samples_expr('sa.sexcheck = sa.pheno.isFemale == sa.imputesex.isFemale')
        ...     .filter_samples_expr('sa.sexcheck || isMissing(sa.sexcheck)'))

        **Notes**

        We have used the same implementation as `PLINK v1.7 <http://pngu.mgh.harvard.edu/~purcell/plink/summary.shtml#sexcheck>`__.

        1. X chromosome variants are selected from the VDS: ``v.contig == "X" || v.contig == "23"``
        2. Variants with a minor allele frequency less than the threshold given by ``maf-threshold`` are removed
        3. Variants in the pseudoautosomal region `(X:60001-2699520) || (X:154931044-155260560)` are included if the ``include_par`` optional parameter is set to true.
        4. The minor allele frequency (maf) per variant is calculated.
        5. For each variant and sample with a non-missing genotype call, :math:`E`, the expected number of homozygotes (from population MAF), is computed as :math:`1.0 - (2.0*maf*(1.0-maf))`.
        6. For each variant and sample with a non-missing genotype call, :math:`O`, the observed number of homozygotes, is computed as `0 = heterozygote; 1 = homozygote`
        7. For each variant and sample with a non-missing genotype call, :math:`N` is incremented by 1
        8. For each sample, :math:`E`, :math:`O`, and :math:`N` are combined across variants
        9. :math:`F` is calculated by :math:`(O - E) / (N - E)`
        10. A sex is assigned to each sample with the following criteria: `F < 0.2 => Female; F > 0.8 => Male`. Use ``female-threshold`` and ``male-threshold`` to change this behavior.

        **Annotations**

        The below annotations can be accessed with ``sa.imputesex``.

        - **isFemale** (*Boolean*) -- True if the imputed sex is female, false if male, missing if undetermined
        - **Fstat** (*Double*) -- Inbreeding coefficient
        - **nTotal** (*Long*) -- Total number of variants considered
        - **nCalled**  (*Long*) -- Number of variants with a genotype call
        - **expectedHoms** (*Double*) -- Expected number of homozygotes
        - **observedHoms** (*Long*) -- Observed number of homozygotes


        :param float maf_threshold: Minimum minor allele frequency threshold.

        :param bool include_par: Include pseudoautosomal regions.

        :param float female_threshold: Samples are called females if F < femaleThreshold

        :param float male_threshold: Samples are called males if F > maleThreshold

        :param str pop_freq: Variant annotation for estimate of MAF.
            If None, MAF will be computed.

        :return: Annotated dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        jvds = self._jvdf.imputeSex(maf_threshold, include_par, female_threshold, male_threshold, joption(pop_freq))
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @typecheck_method(right=vds_type)
    def join(self, right):
        """Join two variant datasets.

        **Notes**

        This method performs an inner join on variants,
        concatenates samples, and takes variant and
        global annotations from the left dataset (self).

        The datasets must have distinct samples, the same sample schema, and the same split status (both split or both multi-allelic).

        :param right: right-hand variant dataset
        :type right: :py:class:`.VariantDataset`

        :return: Joined variant dataset
        :rtype: :py:class:`.VariantDataset`
        """

        return VariantDataset(self.hc, self._jvds.join(right._jvds))

    @handle_py4j
    @requireTGenotype
    @typecheck_method(r2=numeric,
                      window=integral,
                      memory_per_core=integral,
                      num_cores=integral)
    def ld_prune(self, r2=0.2, window=1000000, memory_per_core=256, num_cores=1):
        """Prune variants in linkage disequilibrium (LD).

        .. include:: requireTGenotype.rst

        Requires :py:class:`~hail.VariantDataset.was_split` equals True.

        **Examples**

        Export the set of common LD pruned variants to a file:

        >>> vds_result = (vds.variant_qc()
        ...                  .filter_variants_expr("va.qc.AF >= 0.05 && va.qc.AF <= 0.95")
        ...                  .ld_prune()
        ...                  .export_variants("output/ldpruned.variants", "v"))

        **Notes**

        Variants are pruned in each contig from smallest to largest start position. The LD pruning algorithm is as follows:

        .. code-block:: python

            pruned_set = []
            for v1 in contig:
                keep = True
                for v2 in pruned_set:
                    if ((v1.position - v2.position) <= window and correlation(v1, v2) >= r2):
                        keep = False
                if keep:
                    pruned_set.append(v1)

        The parameter ``window`` defines the maximum distance in base pairs between two variants to check whether
        the variants are independent (:math:`R^2` < ``r2``) where ``r2`` is the maximum :math:`R^2` allowed.
        :math:`R^2` is defined as the square of `Pearson's correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`__
        :math:`{\\rho}_{x,y}` between the two genotype vectors :math:`{\\mathbf{x}}` and :math:`{\\mathbf{y}}`.

        .. math::

            {\\rho}_{x,y} = \\frac{\\mathrm{Cov}(X,Y)}{\\sigma_X \\sigma_Y}


        :py:meth:`.ld_prune` with default arguments is equivalent to ``plink --indep-pairwise 1000kb 1 0.2``.
        The list of pruned variants returned by Hail and PLINK will differ because Hail mean-imputes missing values and tests pairs of variants in a different order than PLINK.

        Be sure to provide enough disk space per worker because :py:meth:`.ld_prune` `persists <http://spark.apache.org/docs/latest/programming-guide.html#rdd-persistence>`__ up to 3 copies of the data to both memory and disk.
        The amount of disk space required will depend on the size and minor allele frequency of the input data and the prune parameters ``r2`` and ``window``. The number of bytes stored in memory per variant is about ``nSamples / 4 + 50``.

        .. warning::

            The variants in the pruned set are not guaranteed to be identical each time :py:meth:`.ld_prune` is run. We recommend running :py:meth:`.ld_prune` once and exporting the list of LD pruned variants using
            :py:meth:`.export_variants` for future use.


        :param float r2: Maximum :math:`R^2` threshold between two variants in the pruned set within a given window.

        :param int window: Width of window in base-pairs for computing pair-wise :math:`R^2` values.

        :param int memory_per_core: Total amount of memory available for each core in MB. If unsure, use the default value.

        :param int num_cores: The number of cores available. Equivalent to the total number of workers times the number of cores per worker.

        :return: Variant dataset filtered to those variants which remain after LD pruning.
        :rtype: :py:class:`.VariantDataset`
        """

        jvds = self._jvdf.ldPrune(r2, window, num_cores, memory_per_core)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @requireTGenotype
    @typecheck_method(force_local=bool)
    def ld_matrix(self, force_local=False):
        """Computes the linkage disequilibrium (correlation) matrix for the variants in this VDS.

        **Examples**

        >>> ld_mat = vds.ld_matrix()

        **Notes**

        Each entry (i, j) in the LD matrix gives the :math:`r` value between variants i and j, defined as
        `Pearson's correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`__
        :math:`\\rho_{x_i,x_j}` between the two genotype vectors :math:`x_i` and :math:`x_j`.

        .. math::

            \\rho_{x_i,x_j} = \\frac{\\mathrm{Cov}(X_i,X_j)}{\\sigma_{X_i} \\sigma_{X_j}}

        Also note that variants with zero variance (:math:`\\sigma = 0`) will be dropped from the matrix.

        .. caution::

            The matrix returned by this function can easily be very large with most entries near zero
            (for example, entries between variants on different chromosomes in a homogenous population).
            Most likely you'll want to reduce the number of variants with methods like
            :py:meth:`.sample_variants`, :py:meth:`.filter_variants_expr`, or :py:meth:`.ld_prune` before
            calling this unless your dataset is very small.

        :param bool force_local: If true, the LD matrix is computed using local matrix multiplication on the Spark driver. This may improve performance when the genotype matrix is small enough to easily fit in local memory. If false, the LD matrix is computed using distributed matrix multiplication if the number of genotypes exceeds :math:`5000^2` and locally otherwise.

        :return: Matrix of r values between pairs of variants.
        :rtype: :py:class:`LDMatrix`
        """

        jldm = self._jvdf.ldMatrix(force_local)
        return LDMatrix(jldm)

    @handle_py4j
    @requireTGenotype
    @typecheck_method(y=strlike,
                      covariates=listof(strlike),
                      root=strlike,
                      use_dosages=bool,
                      min_ac=integral,
                      min_af=numeric)
    def linreg(self, y, covariates=[], root='va.linreg', use_dosages=False, min_ac=1, min_af=0.0):
        r"""Test each variant for association using linear regression.

        .. include:: requireTGenotype.rst

        **Examples**

        Run linear regression per variant using a phenotype and two covariates stored in sample annotations:

        >>> vds_result = vds.linreg('sa.pheno.height', covariates=['sa.pheno.age', 'sa.pheno.isFemale'])

        **Notes**

        The :py:meth:`.linreg` method computes, for each variant, statistics of
        the :math:`t`-test for the genotype coefficient of the linear function
        of best fit from sample genotype and covariates to quantitative
        phenotype or case-control status. Hail only includes samples for which
        phenotype and all covariates are defined. For each variant, missing genotypes
        as the mean of called genotypes.

        By default, genotypes values are given by hard call genotypes (``g.gt``).
        If ``use_dosages=True``, then genotype values are defined by the dosage
        :math:`\mathrm{P}(\mathrm{Het}) + 2 \cdot \mathrm{P}(\mathrm{HomVar})`. For Phred-scaled values,
        :math:`\mathrm{P}(\mathrm{Het})` and :math:`\mathrm{P}(\mathrm{HomVar})` are
        calculated by normalizing the PL likelihoods (converted from the Phred-scale) to sum to 1.

        Assuming there are sample annotations ``sa.pheno.height``,
        ``sa.pheno.age``, ``sa.pheno.isFemale``, and ``sa.cov.PC1``, the code:

        >>> vds_result = vds.linreg('sa.pheno.height', covariates=['sa.pheno.age', 'sa.pheno.isFemale', 'sa.cov.PC1'])

        considers a model of the form

        .. math::

            \mathrm{height} = \beta_0 + \beta_1 \, \mathrm{gt} + \beta_2 \, \mathrm{age} + \beta_3 \, \mathrm{isFemale} + \beta_4 \, \mathrm{PC1} + \varepsilon, \quad \varepsilon \sim \mathrm{N}(0, \sigma^2)

        where the genotype :math:`\mathrm{gt}` is coded as :math:`0` for HomRef, :math:`1` for
        Het, and :math:`2` for HomVar, and the Boolean covariate :math:`\mathrm{isFemale}`
        is coded as :math:`1` for true (female) and :math:`0` for false (male). The null
        model sets :math:`\beta_1 = 0`.

        Those variants that don't vary across the included samples (e.g., all genotypes
        are HomRef) will have missing annotations. One can further
        restrict computation to those variants with at least :math:`k` observed
        alternate alleles (AC) or alternate allele frequency (AF) at least
        :math:`p` in the included samples using the options ``min_ac=k`` or
        ``min_af=p``, respectively. Unlike the :py:meth:`.filter_variants_expr`
        method, these filters do not remove variants from the underlying
        variant dataset; rather the linear regression annotations for variants with
        low AC or AF are set to missing. Adding both filters is equivalent to applying
        the more stringent of the two.

        Phenotype and covariate sample annotations may also be specified using `programmatic expressions <exprlang.html>`__ without identifiers, such as:

        >>> vds_result = vds.linreg('if (sa.pheno.isFemale) sa.pheno.age else (2 * sa.pheno.age + 10)')

        For Boolean covariate types, true is coded as 1 and false as 0. In particular, for the sample annotation ``sa.fam.isCase`` added by importing a FAM file with case-control phenotype, case is 1 and control is 0.

        The standard least-squares linear regression model is derived in Section
        3.2 of `The Elements of Statistical Learning, 2nd Edition
        <http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf>`__. See
        equation 3.12 for the t-statistic which follows the t-distribution with
        :math:`n - k - 2` degrees of freedom, under the null hypothesis of no
        effect, with :math:`n` samples and :math:`k` covariates in addition to
        genotype and intercept.

        **Annotations**

        With the default root, the following four variant annotations are added.

        - **va.linreg.beta** (*Double*) -- fit genotype coefficient, :math:`\hat\beta_1`
        - **va.linreg.se** (*Double*) -- estimated standard error, :math:`\widehat{\mathrm{se}}`
        - **va.linreg.tstat** (*Double*) -- :math:`t`-statistic, equal to :math:`\hat\beta_1 / \widehat{\mathrm{se}}`
        - **va.linreg.pval** (*Double*) -- :math:`p`-value

        :param str y: Response expression

        :param covariates: list of covariate expressions
        :type covariates: list of str

        :param str root: Variant annotation path to store result of linear regression.

        :param bool use_dosages: If true, use dosages genotypes rather than hard call genotypes.

        :param int min_ac: Minimum alternate allele count.

        :param float min_af: Minimum alternate allele frequency.

        :return: Variant dataset with linear regression variant annotations.
        :rtype: :py:class:`.VariantDataset`
        """

        jvds = self._jvdf.linreg(y, jarray(Env.jvm().java.lang.String, covariates), root, use_dosages, min_ac, min_af)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @typecheck_method(key_name=strlike,
                      variant_keys=strlike,
                      single_key=bool,
                      agg_expr=strlike,
                      y=strlike,
                      covariates=listof(strlike))
    def linreg_burden(self, key_name, variant_keys, single_key, agg_expr, y, covariates=[]):
        r"""Test each keyed group of variants for association by aggregating (collapsing) genotypes and applying the
        linear regression model.

        .. include:: requireTGenotype.rst

        **Examples**

        Run a gene burden test using linear regression on the maximum genotype per gene. Here ``va.genes`` is a variant
        annotation of type Set[String] giving the set of genes containing the variant (see **Extended example** below
        for a deep dive):

        >>> linreg_kt, sample_kt = (hc.read('data/example_burden.vds')
        ...     .linreg_burden(key_name='gene',
        ...                    variant_keys='va.genes',
        ...                    single_key=False,
        ...                    agg_expr='gs.map(g => g.gt).max()',
        ...                    y='sa.burden.pheno',
        ...                    covariates=['sa.burden.cov1', 'sa.burden.cov2']))

        Run a gene burden test using linear regression on the weighted sum of genotypes per gene. Here ``va.gene`` is
        a variant annotation of type String giving a single gene per variant (or no gene if missing), and ``va.weight``
        is a numeric variant annotation:

        >>> linreg_kt, sample_kt = (hc.read('data/example_burden.vds')
        ...     .linreg_burden(key_name='gene',
        ...                    variant_keys='va.gene',
        ...                    single_key=True,
        ...                    agg_expr='gs.map(g => va.weight * g.gt).sum()',
        ...                    y='sa.burden.pheno',
        ...                    covariates=['sa.burden.cov1', 'sa.burden.cov2']))

        To use a weighted sum of genotypes with missing genotypes mean-imputed rather than ignored, set
        ``agg_expr='gs.map(g => va.weight * orElse(g.gt.toDouble, 2 * va.qc.AF)).sum()'`` where ``va.qc.AF``
        is the allele frequency over those samples that have no missing phenotype or covariates.

        .. caution::

          With ``single_key=False``, ``variant_keys`` expects a variant annotation of Set or Array type, in order to
          allow each variant to have zero, one, or more keys (for example, the same variant may appear in multiple
          genes). Unlike with type Set, if the same key appears twice in a variant annotation of type Array, then that
          variant will be counted twice in that key's group. With ``single_key=True``, ``variant_keys`` expects a
          variant annotation whose value is itself the key of interest. In bose cases, variants with missing keys are
          ignored.

        **Notes**

        This method modifies :py:meth:`.linreg` by replacing the genotype covariate per variant and sample with
        an aggregated (i.e., collapsed) score per key and sample. This numeric score is computed from the sample's
        genotypes and annotations over all variants with that key. Conceptually, the method proceeds as follows:

        1) Filter to the set of samples for which all phenotype and covariates are defined.

        2) For each key and sample, aggregate genotypes across variants with that key to produce a numeric score.
           ``agg_expr`` must be of numeric type and has the following symbols are in scope:

           - ``s`` (*Sample*): sample
           - ``sa``: sample annotations
           - ``global``: global annotations
           - ``gs`` (*Aggregable[Genotype]*): aggregable of :ref:`genotype` for sample ``s``

           Note that ``v``, ``va``, and ``g`` are accessible through
           `Aggregable methods <https://hail.is/hail/types.html#aggregable>`_ on ``gs``.

           The resulting **sample key table** has key column ``key_name`` and a numeric column of scores for each sample
           named by the sample ID.

        3) For each key, fit the linear regression model using the supplied phenotype and covariates.
           The model is that of :py:meth:`.linreg` with sample genotype ``gt`` replaced by the score in the sample
           key table. For each key, missing scores are mean-imputed across all samples.

           The resulting **linear regression key table** has the following columns:

           - value of ``key_name`` (*String*) -- descriptor of variant group key (key column)
           - **beta** (*Double*) -- fit coefficient, :math:`\hat\beta_1`
           - **se** (*Double*) -- estimated standard error, :math:`\widehat{\mathrm{se}}`
           - **tstat** (*Double*) -- :math:`t`-statistic, equal to :math:`\hat\beta_1 / \widehat{\mathrm{se}}`
           - **pval** (*Double*) -- :math:`p`-value

        :py:meth:`.linreg_burden` returns both the linear regression key table and the sample key table.

        **Extended example**

        Let's walk through these steps in the ``max()`` toy example above.
        There are six samples with the following annotations:

        +--------+-------+------+------+
        | Sample | pheno | cov1 | cov2 |
        +========+=======+======+======+
        |      A |     0 |    0 |   -1 |
        +--------+-------+------+------+
        |      B |     0 |    2 |    3 |
        +--------+-------+------+------+
        |      C |     1 |    1 |    5 |
        +--------+-------+------+------+
        |      D |     1 |   -2 |    0 |
        +--------+-------+------+------+
        |      E |     1 |   -2 |   -4 |
        +--------+-------+------+------+
        |      F |     1 |    4 |    3 |
        +--------+-------+------+------+

        There are three variants with the following ``gt`` values:

        +---------+---+---+---+---+---+---+
        | Variant | A | B | C | D | E | F |
        +=========+===+===+===+===+===+===+
        | 1:1:A:C | 0 | 1 | 0 | 0 | 0 | 1 |
        +---------+---+---+---+---+---+---+
        | 1:2:C:T | . | 2 | . | 2 | 0 | 0 |
        +---------+---+---+---+---+---+---+
        | 1:3:G:C | 0 | . | 1 | 1 | 1 | . |
        +---------+---+---+---+---+---+---+

        The ``va.genes`` annotation of type Set[String] on ``example_burden.vds`` was created
        using :py:meth:`.annotate_variants_table` with ``product=True`` on the interval list:

        .. literalinclude:: data/genes.interval_list

        So there are three overlapping genes: gene A contains two variants,
        gene B contains one variant, and gene C contains all three variants.

        +--------+---------+---------+---------+
        |  gene  | 1:1:A:C | 1:2:C:T | 1:3:G:C |
        +========+=========+=========+=========+
        |  geneA |    X    |    X    |         |
        +--------+---------+---------+---------+
        |  geneB |         |    X    |         |
        +--------+---------+---------+---------+
        |  geneC |    X    |    X    |    X    |
        +--------+---------+---------+---------+

        Therefore :py:meth:`.annotate_variants_table` with ``product=True`` creates
        a variant annotation of type Set[String] with values ``Set('geneA', 'geneB')``,
        ``Set('geneB')``, and ``Set('geneA', 'geneB', 'geneC')``.

        So the sample aggregation key table is:

        +-----+---+---+---+---+---+---+
        | gene| A | B | C | D | E | F |
        +=====+===+===+===+===+===+===+
        |geneA|  0|  2|  0|  2|  0|  1|
        +-----+---+---+---+---+---+---+
        |geneB| NA|  2| NA|  2|  0|  0|
        +-----+---+---+---+---+---+---+
        |geneC|  0|  2|  1|  2|  1|  1|
        +-----+---+---+---+---+---+---+

        Linear regression is done for each row using the supplied phenotype, covariates, and implicit intercept.
        The resulting linear regression key table is:

        +------+-------+------+-------+------+
        | gene |  beta |   se | tstat | pval |
        +======+=======+======+=======+======+
        | geneA| -0.084| 0.368| -0.227| 0.841|
        +------+-------+------+-------+------+
        | geneB| -0.542| 0.335| -1.617| 0.247|
        +------+-------+------+-------+------+
        | geneC|  0.075| 0.515|  0.145| 0.898|
        +------+-------+------+-------+------+

        :param str key_name: Name to assign to key column of returned key tables.

        :param str variant_keys: Variant annotation path for the TArray or TSet of keys associated to each variant.

        :param bool single_key: if true, ``variant_keys`` is interpreted as a single (or missing) key per variant,
                                rather than as a collection of keys.

        :param str agg_expr: Sample aggregation expression (per key).

        :param str y: Response expression.

        :param covariates: list of covariate expressions.
        :type covariates: list of str

        :return: Tuple of linear regression key table and sample aggregation key table.
        :rtype: (:py:class:`.KeyTable`, :py:class:`.KeyTable`)
        """

        r = self._jvdf.linregBurden(key_name, variant_keys, single_key, agg_expr, y,
                                    jarray(Env.jvm().java.lang.String, covariates))
        linreg_kt = KeyTable(self.hc, r._1())
        sample_kt = KeyTable(self.hc, r._2())

        return linreg_kt, sample_kt

    @handle_py4j
    @requireTGenotype
    @typecheck_method(ys=listof(strlike),
                      covariates=listof(strlike),
                      root=strlike,
                      use_dosages=bool,
                      min_ac=integral,
                      min_af=numeric)
    def linreg_multi_pheno(self, ys, covariates=[], root='va.linreg', use_dosages=False, min_ac=1, min_af=0.0):
        r"""Test each variant for association with multiple phenotypes using linear regression.

        This method runs linear regression for multiple phenotypes more efficiently
        than looping over :py:meth:`.linreg`.

        .. warning::

            :py:meth:`.linreg_multi_pheno` uses the same set of samples for each phenotype,
            namely the set of samples for which **all** phenotypes and covariates are defined.

        **Annotations**

        With the default root, the following four variant annotations are added.
        The indexing of these annotations corresponds to that of ``y``.

        - **va.linreg.beta** (*Array[Double]*) -- array of fit genotype coefficients, :math:`\hat\beta_1`
        - **va.linreg.se** (*Array[Double]*) -- array of estimated standard errors, :math:`\widehat{\mathrm{se}}`
        - **va.linreg.tstat** (*Array[Double]*) -- array of :math:`t`-statistics, equal to :math:`\hat\beta_1 / \widehat{\mathrm{se}}`
        - **va.linreg.pval** (*Array[Double]*) -- array of :math:`p`-values

        :param ys: list of one or more response expressions.
        :type covariates: list of str

        :param covariates: list of covariate expressions.
        :type covariates: list of str

        :param str root: Variant annotation path to store result of linear regression.

        :param bool use_dosages: If true, use dosage genotypes rather than hard call genotypes.

        :param int min_ac: Minimum alternate allele count.

        :param float min_af: Minimum alternate allele frequency.

        :return: Variant dataset with linear regression variant annotations.
        :rtype: :py:class:`.VariantDataset`
        """

        jvds = self._jvdf.linregMultiPheno(jarray(Env.jvm().java.lang.String, ys),
                                           jarray(Env.jvm().java.lang.String, covariates), root, use_dosages, min_ac,
                                           min_af)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @requireTGenotype
    @typecheck_method(ys=listof(strlike),
                      covariates=listof(strlike),
                      root=strlike,
                      use_dosages=bool,
                      variant_block_size=integral)
    def linreg3(self, ys, covariates=[], root='va.linreg', use_dosages=False, variant_block_size=16):
        r"""Test each variant for association with multiple phenotypes using linear regression.

        This method runs linear regression for multiple phenotypes
        more efficiently than looping over :py:meth:`.linreg`.  This
        method is more efficient than :py:meth:`.linreg_multi_pheno`
        but doesn't implicitly filter on allele count or allele
        frequency.

        .. warning::

            :py:meth:`.linreg3` uses the same set of samples for each phenotype,
            namely the set of samples for which **all** phenotypes and covariates are defined.

        **Annotations**

        With the default root, the following four variant annotations are added.
        The indexing of the array annotations corresponds to that of ``y``.

        - **va.linreg.nCompleteSamples** (*Int*) -- number of samples used
        - **va.linreg.AC** (*Double*) -- sum of the genotype values ``x``
        - **va.linreg.ytx** (*Array[Double]*) -- array of dot products of each phenotype vector ``y`` with the genotype vector ``x``
        - **va.linreg.beta** (*Array[Double]*) -- array of fit genotype coefficients, :math:`\hat\beta_1`
        - **va.linreg.se** (*Array[Double]*) -- array of estimated standard errors, :math:`\widehat{\mathrm{se}}`
        - **va.linreg.tstat** (*Array[Double]*) -- array of :math:`t`-statistics, equal to :math:`\hat\beta_1 / \widehat{\mathrm{se}}`
        - **va.linreg.pval** (*Array[Double]*) -- array of :math:`p`-values

        :param ys: list of one or more response expressions.
        :type covariates: list of str

        :param covariates: list of covariate expressions.
        :type covariates: list of str

        :param str root: Variant annotation path to store result of linear regression.

        :param bool use_dosages: If true, use dosage genotypes rather than hard call genotypes.

        :param int variant_block_size: Number of variant regressions to perform simultaneously.  Larger block size requires more memmory.

        :return: Variant dataset with linear regression variant annotations.
        :rtype: :py:class:`.VariantDataset`

        """

        jvds = self._jvdf.linreg3(jarray(Env.jvm().java.lang.String, ys),
                                  jarray(Env.jvm().java.lang.String, covariates), root, use_dosages, variant_block_size)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @requireTGenotype
    @typecheck_method(kinshipMatrix=KinshipMatrix,
                      y=strlike,
                      covariates=listof(strlike),
                      global_root=strlike,
                      va_root=strlike,
                      run_assoc=bool,
                      use_ml=bool,
                      delta=nullable(numeric),
                      sparsity_threshold=numeric,
                      use_dosages=bool,
                      n_eigs=nullable(integral),
                      dropped_variance_fraction=(nullable(float)))
    def lmmreg(self, kinshipMatrix, y, covariates=[], global_root="global.lmmreg", va_root="va.lmmreg",
               run_assoc=True, use_ml=False, delta=None, sparsity_threshold=1.0, use_dosages=False,
               n_eigs=None, dropped_variance_fraction=None):
        """Use a kinship-based linear mixed model to estimate the genetic component of phenotypic variance (narrow-sense heritability) and optionally test each variant for association.

        .. include:: requireTGenotype.rst

        **Examples**

        Suppose the variant dataset saved at *data/example_lmmreg.vds* has a Boolean variant annotation ``va.useInKinship`` and numeric or Boolean sample annotations ``sa.pheno``, ``sa.cov1``, ``sa.cov2``. Then the :py:meth:`.lmmreg` function in

        >>> assoc_vds = hc.read("data/example_lmmreg.vds")
        >>> kinship_matrix = assoc_vds.filter_variants_expr('va.useInKinship').rrm()
        >>> lmm_vds = assoc_vds.lmmreg(kinship_matrix, 'sa.pheno', ['sa.cov1', 'sa.cov2'])

        will execute the following four steps in order:

        1) filter to samples in given kinship matrix to those for which ``sa.pheno``, ``sa.cov``, and ``sa.cov2`` are all defined
        2) compute the eigendecomposition :math:`K = USU^T` of the kinship matrix
        3) fit covariate coefficients and variance parameters in the sample-covariates-only (global) model using restricted maximum likelihood (`REML <https://en.wikipedia.org/wiki/Restricted_maximum_likelihood>`__), storing results in global annotations under ``global.lmmreg``
        4) test each variant for association, storing results under ``va.lmmreg`` in variant annotations

        This plan can be modified as follows:

        - Set ``run_assoc=False`` to not test any variants for association, i.e. skip Step 5.
        - Set ``use_ml=True`` to use maximum likelihood instead of REML in Steps 4 and 5.
        - Set the ``delta`` argument to manually set the value of :math:`\delta` rather that fitting :math:`\delta` in Step 4.
        - Set the ``global_root`` argument to change the global annotation root in Step 4.
        - Set the ``va_root`` argument to change the variant annotation root in Step 5.

        :py:meth:`.lmmreg` adds 9 or 13 global annotations in Step 4, depending on whether :math:`\delta` is set or fit.

        +----------------------------------------------+----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
        | Annotation                                   | Type                 | Value                                                                                                                                                |
        +==============================================+======================+======================================================================================================================================================+
        | ``global.lmmreg.useML``                      | Boolean              | true if fit by ML, false if fit by REML                                                                                                              |
        +----------------------------------------------+----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
        | ``global.lmmreg.beta``                       | Dict[String, Double] | map from *intercept* and the given ``covariates`` expressions to the corresponding fit :math:`\\beta` coefficients                                    |
        +----------------------------------------------+----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
        | ``global.lmmreg.sigmaG2``                    | Double               | fit coefficient of genetic variance, :math:`\\hat{\sigma}_g^2`                                                                                        |
        +----------------------------------------------+----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
        | ``global.lmmreg.sigmaE2``                    | Double               | fit coefficient of environmental variance :math:`\\hat{\sigma}_e^2`                                                                                   |
        +----------------------------------------------+----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
        | ``global.lmmreg.delta``                      | Double               | fit ratio of variance component coefficients, :math:`\\hat{\delta}`                                                                                   |
        +----------------------------------------------+----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
        | ``global.lmmreg.h2``                         | Double               | fit narrow-sense heritability, :math:`\\hat{h}^2`                                                                                                     |
        +----------------------------------------------+----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
        | ``global.lmmreg.nEigs``                      | Int                  | number of eigenvectors of kinship matrix used to fit model                                                                                           |
        +----------------------------------------------+----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
        | ``global.lmmreg.dropped_variance_fraction``  | Double               | specified value of ``dropped_variance_fraction``                                                                                                     |
        +----------------------------------------------+----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
        | ``global.lmmreg.evals``                      | Array[Double]        | all eigenvalues of the kinship matrix in descending order                                                                                            |
        +----------------------------------------------+----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
        | ``global.lmmreg.fit.seH2``                   | Double               | standard error of :math:`\\hat{h}^2` under asymptotic normal approximation                                                                            |
        +----------------------------------------------+----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
        | ``global.lmmreg.fit.normLkhdH2``             | Array[Double]        | likelihood function of :math:`h^2` normalized on the discrete grid ``0.01, 0.02, ..., 0.99``. Index ``i`` is the likelihood for percentage ``i``.    |
        +----------------------------------------------+----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
        | ``global.lmmreg.fit.maxLogLkhd``             | Double               | (restricted) maximum log likelihood corresponding to :math:`\\hat{\delta}`                                                                            |
        +----------------------------------------------+----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
        | ``global.lmmreg.fit.logDeltaGrid``           | Array[Double]        | values of :math:`\\mathrm{ln}(\delta)` used in the grid search                                                                                        |
        +----------------------------------------------+----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
        | ``global.lmmreg.fit.logLkhdVals``            | Array[Double]        | (restricted) log likelihood of :math:`y` given :math:`X` and :math:`\\mathrm{ln}(\delta)` at the (RE)ML fit of :math:`\\beta` and :math:`\sigma_g^2`   |
        +----------------------------------------------+----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+

        These global annotations are also added to ``hail.log``, with the ranked evals and :math:`\delta` grid with values in .tsv tabular form.  Use ``grep 'lmmreg:' hail.log`` to find the lines just above each table.

        If Step 5 is performed, :py:meth:`.lmmreg` also adds four linear regression variant annotations.

        +------------------------+--------+-------------------------------------------------------------------------+
        | Annotation             | Type   | Value                                                                   |
        +========================+========+=========================================================================+
        | ``va.lmmreg.beta``     | Double | fit genotype coefficient, :math:`\hat\\beta_0`                           |
        +------------------------+--------+-------------------------------------------------------------------------+
        | ``va.lmmreg.sigmaG2``  | Double | fit coefficient of genetic variance component, :math:`\hat{\sigma}_g^2` |
        +------------------------+--------+-------------------------------------------------------------------------+
        | ``va.lmmreg.chi2``     | Double | :math:`\chi^2` statistic of the likelihood ratio test                   |
        +------------------------+--------+-------------------------------------------------------------------------+
        | ``va.lmmreg.pval``     | Double | :math:`p`-value                                                         |
        +------------------------+--------+-------------------------------------------------------------------------+

        Those variants that don't vary across the included samples (e.g., all genotypes
        are HomRef) will have missing annotations.

        The simplest way to export all resulting annotations is:

        >>> lmm_vds.export_variants('output/lmmreg.tsv.bgz', 'variant = v, va.lmmreg.*')
        >>> lmmreg_results = lmm_vds.globals['lmmreg']
        
        By default, genotypes values are given by hard call genotypes (``g.gt``).
        If ``use_dosages=True``, then genotype values for per-variant association are defined by the dosage
        :math:`\mathrm{P}(\mathrm{Het}) + 2 \cdot \mathrm{P}(\mathrm{HomVar})`. For Phred-scaled values,
        :math:`\mathrm{P}(\mathrm{Het})` and :math:`\mathrm{P}(\mathrm{HomVar})` are
        calculated by normalizing the PL likelihoods (converted from the Phred-scale) to sum to 1.

        **Performance**

        Hail's initial version of :py:meth:`.lmmreg` scales beyond 15k samples and to an essentially unbounded number of variants, making it particularly well-suited to modern sequencing studies and complementary to tools designed for SNP arrays. Analysts have used :py:meth:`.lmmreg` in research to compute kinship from 100k common variants and test 32 million non-rare variants on 8k whole genomes in about 10 minutes on `Google cloud <http://discuss.hail.is/t/using-hail-on-the-google-cloud-platform/80>`__.

        While :py:meth:`.lmmreg` computes the kinship matrix :math:`K` using distributed matrix multiplication (Step 2), the full `eigendecomposition <https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix>`__ (Step 3) is currently run on a single core of master using the `LAPACK routine DSYEVD <http://www.netlib.org/lapack/explore-html/d2/d8a/group__double_s_yeigen_ga694ddc6e5527b6223748e3462013d867.html>`__, which we empirically find to be the most performant of the four available routines; laptop performance plots showing cubic complexity in :math:`n` are available `here <https://github.com/hail-is/hail/pull/906>`__. On Google cloud, eigendecomposition takes about 2 seconds for 2535 sampes and 1 minute for 8185 samples. If you see worse performance, check that LAPACK natives are being properly loaded (see "BLAS and LAPACK" in Getting Started).

        Given the eigendecomposition, fitting the global model (Step 4) takes on the order of a few seconds on master. Association testing (Step 5) is fully distributed by variant with per-variant time complexity that is completely independent of the number of sample covariates and dominated by multiplication of the genotype vector :math:`v` by the matrix of eigenvectors :math:`U^T` as described below, which we accelerate with a sparse representation of :math:`v`.  The matrix :math:`U^T` has size about :math:`8n^2` bytes and is currently broadcast to each Spark executor. For example, with 15k samples, storing :math:`U^T` consumes about 3.6GB of memory on a 16-core worker node with two 8-core executors. So for large :math:`n`, we recommend using a high-memory configuration such as ``highmem`` workers.

        **Linear mixed model**

        :py:meth:`.lmmreg` estimates the genetic proportion of residual phenotypic variance (narrow-sense heritability) under a kinship-based linear mixed model, and then optionally tests each variant for association using the likelihood ratio test. Inference is exact.

        We first describe the sample-covariates-only model used to estimate heritability, which we simply refer to as the *global model*. With :math:`n` samples and :math:`c` sample covariates, we define:

        - :math:`y = n \\times 1` vector of phenotypes
        - :math:`X = n \\times c` matrix of sample covariates and intercept column of ones
        - :math:`K = n \\times n` kinship matrix
        - :math:`I = n \\times n` identity matrix
        - :math:`\\beta = c \\times 1` vector of covariate coefficients
        - :math:`\sigma_g^2 =` coefficient of genetic variance component :math:`K`
        - :math:`\sigma_e^2 =` coefficient of environmental variance component :math:`I`
        - :math:`\delta = \\frac{\sigma_e^2}{\sigma_g^2} =` ratio of environmental and genetic variance component coefficients
        - :math:`h^2 = \\frac{\sigma_g^2}{\sigma_g^2 + \sigma_e^2} = \\frac{1}{1 + \delta} =` genetic proportion of residual phenotypic variance

        Under a linear mixed model, :math:`y` is sampled from the :math:`n`-dimensional `multivariate normal distribution <https://en.wikipedia.org/wiki/Multivariate_normal_distribution>`__ with mean :math:`X \\beta` and variance components that are scalar multiples of :math:`K` and :math:`I`:

        .. math::

          y \sim \mathrm{N}\\left(X\\beta, \sigma_g^2 K + \sigma_e^2 I\\right)

        Thus the model posits that the residuals :math:`y_i - X_{i,:}\\beta` and :math:`y_j - X_{j,:}\\beta` have covariance :math:`\sigma_g^2 K_{ij}` and approximate correlation :math:`h^2 K_{ij}`. Informally: phenotype residuals are correlated as the product of overall heritability and pairwise kinship. By contrast, standard (unmixed) linear regression is equivalent to fixing :math:`\sigma_2` (equivalently, :math:`h^2`) at 0 above, so that all phenotype residuals are independent.

        **Caution:** while it is tempting to interpret :math:`h^2` as the `narrow-sense heritability <https://en.wikipedia.org/wiki/Heritability#Definition>`__ of the phenotype alone, note that its value depends not only the phenotype and genetic data, but also on the choice of sample covariates.

        **Fitting the global model**

        The core algorithm is essentially a distributed implementation of the spectral approach taken in `FastLMM <https://www.microsoft.com/en-us/research/project/fastlmm/>`__. Let :math:`K = USU^T` be the `eigendecomposition <https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix#Real_symmetric_matrices>`__ of the real symmetric matrix :math:`K`. That is:

        - :math:`U = n \\times n` orthonormal matrix whose columns are the eigenvectors of :math:`K`
        - :math:`S = n \\times n` diagonal matrix of eigenvalues of :math:`K` in descending order. :math:`S_{ii}` is the eigenvalue of eigenvector :math:`U_{:,i}`
        - :math:`U^T = n \\times n` orthonormal matrix, the transpose (and inverse) of :math:`U`

        A bit of matrix algebra on the multivariate normal density shows that the linear mixed model above is mathematically equivalent to the model

        .. math::

          U^Ty \\sim \mathrm{N}\\left(U^TX\\beta, \sigma_g^2 (S + \delta I)\\right)

        for which the covariance is diagonal (e.g., unmixed). That is, rotating the phenotype vector (:math:`y`) and covariate vectors (columns of :math:`X`) in :math:`\mathbb{R}^n` by :math:`U^T` transforms the model to one with independent residuals. For any particular value of :math:`\delta`, the restricted maximum likelihood (REML) solution for the latter model can be solved exactly in time complexity that is linear rather than cubic in :math:`n`.  In particular, having rotated, we can run a very efficient 1-dimensional optimization procedure over :math:`\delta` to find the REML estimate :math:`(\hat{\delta}, \\hat{\\beta}, \\hat{\sigma}_g^2)` of the triple :math:`(\delta, \\beta, \sigma_g^2)`, which in turn determines :math:`\\hat{\sigma}_e^2` and :math:`\\hat{h}^2`.

        We first compute the maximum log likelihood on a :math:`\delta`-grid that is uniform on the log scale, with :math:`\\mathrm{ln}(\delta)` running from -8 to 8 by 0.01, corresponding to :math:`h^2` decreasing from 0.9995 to 0.0005. If :math:`h^2` is maximized at the lower boundary then standard linear regression would be more appropriate and Hail will exit; more generally, consider using standard linear regression when :math:`\\hat{h}^2` is very small. A maximum at the upper boundary is highly suspicious and will also cause Hail to exit, with the ``hail.log`` recording all values over the grid for further inspection.

        If the optimal grid point falls in the interior of the grid as expected, we then use `Brent's method <https://en.wikipedia.org/wiki/Brent%27s_method>`__ to find the precise location of the maximum over the same range, with initial guess given by the optimal grid point and a tolerance on :math:`\\mathrm{ln}(\delta)` of 1e-6. If this location differs from the optimal grid point by more than 0.01, a warning will be displayed and logged, and one would be wise to investigate by plotting the values over the grid.

        Note that :math:`h^2` is related to :math:`\\mathrm{ln}(\delta)` through the `sigmoid function <https://en.wikipedia.org/wiki/Sigmoid_function>`_. More precisely,

        .. math::

          h^2 = 1 - \mathrm{sigmoid}(\\mathrm{ln}(\delta)) = \mathrm{sigmoid}(-\\mathrm{ln}(\delta))

        Hence one can change variables to extract a high-resolution discretization of the likelihood function of :math:`h^2` over :math:`[0,1]` at the corresponding REML estimators for :math:`\\beta` and :math:`\sigma_g^2`, as well as integrate over the normalized likelihood function using `change of variables <https://en.wikipedia.org/wiki/Integration_by_substitution>`_ and the `sigmoid differential equation <https://en.wikipedia.org/wiki/Sigmoid_function#Properties>`_.

        For convenience, ``global.lmmreg.fit.normLkhdH2`` records the the likelihood function of :math:`h^2` normalized over the discrete grid ``0.01, 0.02, ..., 0.98, 0.99``. The length of the array is 101 so that index ``i`` contains the likelihood at percentage ``i``. The values at indices 0 and 100 are left undefined.

        By the theory of maximum likelihood estimation, this normalized likelihood function is approximately normally distributed near the maximum likelihood estimate. So we estimate the standard error of the estimator of :math:`h^2` as follows. Let :math:`x_2` be the maximum likelihood estimate of :math:`h^2` and let :math:`x_ 1` and :math:`x_3` be just to the left and right of :math:`x_2`. Let :math:`y_1`, :math:`y_2`, and :math:`y_3` be the corresponding values of the (unnormalized) log likelihood function. Setting equal the leading coefficient of the unique parabola through these points (as given by Lagrange interpolation) and the leading coefficient of the log of the normal distribution, we have:

        .. math::

          \\frac{x_3 (y_2 - y_1) + x_2 (y_1 - y_3) + x_1 (y_3 - y_2))}{(x_2 - x_1)(x_1 - x_3)(x_3 - x_2)} = -\\frac{1}{2 \sigma^2}

        The standard error :math:`\\hat{\sigma}` is then estimated by solving for :math:`\sigma`.

        Note that the mean and standard deviation of the (discretized or continuous) distribution held in ``global.lmmreg.fit.normLkhdH2`` will not coincide with :math:`\\hat{h}^2` and :math:`\\hat{\sigma}`, since this distribution only becomes normal in the infinite sample limit. One can visually assess normality by plotting this distribution against a normal distribution with the same mean and standard deviation, or use this distribution to approximate credible intervals under a flat prior on :math:`h^2`.

        **Testing each variant for association**

        Fixing a single variant, we define:

        - :math:`v = n \\times 1` vector of genotypes, with missing genotypes imputed as the mean of called genotypes
        - :math:`X_v = \\left[v | X \\right] = n \\times (1 + c)` matrix concatenating :math:`v` and :math:`X`
        - :math:`\\beta_v = (\\beta^0_v, \\beta^1_v, \\ldots, \\beta^c_v) = (1 + c) \\times 1` vector of covariate coefficients

        Fixing :math:`\delta` at the global REML estimate :math:`\\hat{\delta}`, we find the REML estimate :math:`(\\hat{\\beta}_v, \\hat{\sigma}_{g,v}^2)` via rotation of the model

        .. math::

          y \\sim \\mathrm{N}\\left(X_v\\beta_v, \sigma_{g,v}^2 (K + \\hat{\delta} I)\\right)

        Note that the only new rotation to compute here is :math:`U^T v`.

        To test the null hypothesis that the genotype coefficient :math:`\\beta^0_v` is zero, we consider the restricted model with parameters :math:`((0, \\beta^1_v, \ldots, \\beta^c_v), \sigma_{g,v}^2)` within the full model with parameters :math:`(\\beta^0_v, \\beta^1_v, \\ldots, \\beta^c_v), \sigma_{g_v}^2)`, with :math:`\delta` fixed at :math:`\\hat\delta` in both. The latter fit is simply that of the global model, :math:`((0, \\hat{\\beta}^1, \\ldots, \\hat{\\beta}^c), \\hat{\sigma}_g^2)`. The likelihood ratio test statistic is given by

        .. math::

          \chi^2 = n \\, \\mathrm{ln}\left(\\frac{\hat{\sigma}^2_g}{\\hat{\sigma}_{g,v}^2}\\right)

        and follows a chi-squared distribution with one degree of freedom. Here the ratio :math:`\\hat{\sigma}^2_g / \\hat{\sigma}_{g,v}^2` captures the degree to which adding the variant :math:`v` to the global model reduces the residual phenotypic variance.

        **Kinship Matrix**

        FastLMM uses the Realized Relationship Matrix (RRM) for kinship. This can be computed with :py:meth:`~hail.VariantDataset.rrm`. However, any instance of :py:class:`KinshipMatrix` may be used, so long as ``sample_list`` contains the complete samples of the caller variant dataset in the same order.

        **Low-rank approximation of kinship for improved performance**

        :py:meth:`.lmmreg` can implicitly use a low-rank approximation of the kinship matrix to more rapidly fit delta and the statistics for each variant. The computational complexity per variant is proportional to the number of eigenvectors used. This number can be specified in two ways. Specify the parameter ``n_eigs`` to use only the top ``n_eigs`` eigenvectors. Alternatively, specify ``dropped_variance_fraction`` to use as many eigenvectors as necessary to capture all but at most this fraction of the sample variance (also known as the trace, or the sum of the eigenvalues). For example, ``dropped_variance_fraction=0.01`` will use the minimal number of eigenvectors to account for 99% of the sample variance. Specifying both parameters will apply the more stringent (fewest eigenvectors) of the two.

        **Further background**

        For the history and mathematics of linear mixed models in genetics, including `FastLMM <https://www.microsoft.com/en-us/research/project/fastlmm/>`__, see `Christoph Lippert's PhD thesis <https://publikationen.uni-tuebingen.de/xmlui/bitstream/handle/10900/50003/pdf/thesis_komplett.pdf>`__. For an investigation of various approaches to defining kinship, see `Comparison of Methods to Account for Relatedness in Genome-Wide Association Studies with Family-Based Data <http://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1004445>`__.

        :param kinshipMatrix: Kinship matrix to be used.
        :type kinshipMatrix: :class:`KinshipMatrix`

        :param str y: Response sample annotation.

        :param covariates: List of covariate sample annotations.
        :type covariates: list of str

        :param str global_root: Global annotation root, a period-delimited path starting with `global`.

        :param str va_root: Variant annotation root, a period-delimited path starting with `va`.

        :param bool run_assoc: If true, run association testing in addition to fitting the global model.

        :param bool use_ml: Use ML instead of REML throughout.

        :param delta: Fixed delta value to use in the global model, overrides fitting delta.
        :type delta: float or None

        :param float sparsity_threshold: Genotype vector sparsity at or below which to use sparse genotype vector in rotation (advanced).

        :param bool use_dosages: If true, use dosages rather than hard call genotypes.

        :param int n_eigs: Number of eigenvectors of the kinship matrix used to fit the model.

        :param float dropped_variance_fraction: Upper bound on fraction of sample variance lost by dropping eigenvectors with small eigenvalues.

        :return: Variant dataset with linear mixed regression annotations.
        :rtype: :py:class:`.VariantDataset`
        """

        jvds = self._jvdf.lmmreg(kinshipMatrix._jkm, y, jarray(Env.jvm().java.lang.String, covariates),
                                 use_ml, global_root, va_root, run_assoc, joption(delta), sparsity_threshold,
                                 use_dosages, joption(n_eigs), joption(dropped_variance_fraction))
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @requireTGenotype
    @typecheck_method(test=strlike,
                      y=strlike,
                      covariates=listof(strlike),
                      root=strlike,
                      use_dosages=bool)
    def logreg(self, test, y, covariates=[], root='va.logreg', use_dosages=False):
        """Test each variant for association using logistic regression.

        .. include:: requireTGenotype.rst

        **Examples**

        Run the logistic regression Wald test per variant using a Boolean phenotype and two covariates stored
        in sample annotations:

        >>> vds_result = vds.logreg('wald', 'sa.pheno.isCase', covariates=['sa.pheno.age', 'sa.pheno.isFemale'])

        **Notes**

        The :py:meth:`~hail.VariantDataset.logreg` method performs,
        for each variant, a significance test of the genotype in
        predicting a binary (case-control) phenotype based on the
        logistic regression model. The phenotype type must either be numeric (with all
        present values 0 or 1) or Boolean, in which case true and false are coded as 1 and 0, respectively.

        Hail supports the Wald test ('wald'), likelihood ratio test ('lrt'), Rao score test ('score'),
        and Firth test ('firth'). Hail only includes samples for which the phenotype and all covariates are
        defined. For each variant, Hail imputes missing genotypes as the mean of called genotypes.

        By default, genotypes values are given by hard call genotypes (``g.gt``).
        If ``use_dosages=True``, then genotype values are defined by the dosage
        :math:`\mathrm{P}(\mathrm{Het}) + 2 \cdot \mathrm{P}(\mathrm{HomVar})`. For Phred-scaled values,
        :math:`\mathrm{P}(\mathrm{Het})` and :math:`\mathrm{P}(\mathrm{HomVar})` are
        calculated by normalizing the PL likelihoods (converted from the Phred-scale) to sum to 1.

        The example above considers a model of the form

        .. math::

          \mathrm{Prob}(\mathrm{isCase}) = \mathrm{sigmoid}(\\beta_0 + \\beta_1 \, \mathrm{gt} + \\beta_2 \, \mathrm{age} + \\beta_3 \, \mathrm{isFemale} + \\varepsilon), \quad \\varepsilon \sim \mathrm{N}(0, \sigma^2)

        where :math:`\mathrm{sigmoid}` is the `sigmoid
        function <https://en.wikipedia.org/wiki/Sigmoid_function>`__, the
        genotype :math:`\mathrm{gt}` is coded as 0 for HomRef, 1 for
        Het, and 2 for HomVar, and the Boolean covariate
        :math:`\mathrm{isFemale}` is coded as 1 for true (female) and
        0 for false (male). The null model sets :math:`\\beta_1 = 0`.

        The resulting variant annotations depend on the test statistic
        as shown in the tables below.

        ========== =================== ====== =====
        Test       Annotation          Type   Value
        ========== =================== ====== =====
        Wald       ``va.logreg.beta``  Double fit genotype coefficient, :math:`\hat\\beta_1`
        Wald       ``va.logreg.se``    Double estimated standard error, :math:`\widehat{\mathrm{se}}`
        Wald       ``va.logreg.zstat`` Double Wald :math:`z`-statistic, equal to :math:`\hat\\beta_1 / \widehat{\mathrm{se}}`
        Wald       ``va.logreg.pval``  Double Wald p-value testing :math:`\\beta_1 = 0`
        LRT, Firth ``va.logreg.beta``  Double fit genotype coefficient, :math:`\hat\\beta_1`
        LRT, Firth ``va.logreg.chi2``  Double deviance statistic
        LRT, Firth ``va.logreg.pval``  Double LRT / Firth p-value testing :math:`\\beta_1 = 0`
        Score      ``va.logreg.chi2``  Double score statistic
        Score      ``va.logreg.pval``  Double score p-value testing :math:`\\beta_1 = 0`
        ========== =================== ====== =====

        For the Wald and likelihood ratio tests, Hail fits the logistic model for each variant using Newton iteration and only emits the above annotations when the maximum likelihood estimate of the coefficients converges. The Firth test uses a modified form of Newton iteration. To help diagnose convergence issues, Hail also emits three variant annotations which summarize the iterative fitting process:

        ================ =========================== ======= =====
        Test             Annotation                  Type    Value
        ================ =========================== ======= =====
        Wald, LRT, Firth ``va.logreg.fit.nIter``     Int     number of iterations until convergence, explosion, or reaching the max (25 for Wald, LRT; 100 for Firth)
        Wald, LRT, Firth ``va.logreg.fit.converged`` Boolean true if iteration converged
        Wald, LRT, Firth ``va.logreg.fit.exploded``  Boolean true if iteration exploded
        ================ =========================== ======= =====

        We consider iteration to have converged when every coordinate of :math:`\\beta` changes by less than :math:`10^{-6}`. For Wald and LRT, up to 25 iterations are attempted; in testing we find 4 or 5 iterations nearly always suffice. Convergence may also fail due to explosion, which refers to low-level numerical linear algebra exceptions caused by manipulating ill-conditioned matrices. Explosion may result from (nearly) linearly dependent covariates or complete `separation <https://en.wikipedia.org/wiki/Separation_(statistics)>`__.

        A more common situation in genetics is quasi-complete seperation, e.g. variants that are observed only in cases (or controls). Such variants inevitably arise when testing millions of variants with very low minor allele count. The maximum likelihood estimate of :math:`\\beta` under logistic regression is then undefined but convergence may still occur after a large number of iterations due to a very flat likelihood surface. In testing, we find that such variants produce a secondary bump from 10 to 15 iterations in the histogram of number of iterations per variant. We also find that this faux convergence produces large standard errors and large (insignificant) p-values. To not miss such variants, consider using Firth logistic regression, linear regression, or group-based tests.

        Here's a concrete illustration of quasi-complete seperation in R. Suppose we have 2010 samples distributed as follows for a particular variant:

        ======= ====== === ======
        Status  HomRef Het HomVar
        ======= ====== === ======
        Case    1000   10  0
        Control 1000   0   0
        ======= ====== === ======

        The following R code fits the (standard) logistic, Firth logistic, and linear regression models to this data, where ``x`` is genotype, ``y`` is phenotype, and ``logistf`` is from the logistf package:

        .. code-block:: R

            x <- c(rep(0,1000), rep(1,1000), rep(1,10)
            y <- c(rep(0,1000), rep(0,1000), rep(1,10))
            logfit <- glm(y ~ x, family=binomial())
            firthfit <- logistf(y ~ x)
            linfit <- lm(y ~ x)

        The resulting p-values for the genotype coefficient are 0.991, 0.00085, and 0.0016, respectively. The erroneous value 0.991 is due to quasi-complete separation. Moving one of the 10 hets from case to control eliminates this quasi-complete separation; the p-values from R are then 0.0373, 0.0111, and 0.0116, respectively, as expected for a less significant association.

        The Firth test reduces bias from small counts and resolves the issue of separation by penalizing maximum likelihood estimation by the `Jeffrey's invariant prior <https://en.wikipedia.org/wiki/Jeffreys_prior>`__. This test is slower, as both the null and full model must be fit per variant, and convergence of the modified Newton method is linear rather than quadratic. For Firth, 100 iterations are attempted for the null model and, if that is successful, for the full model as well. In testing we find 20 iterations nearly always suffices. If the null model fails to converge, then the ``sa.lmmreg.fit`` annotations reflect the null model; otherwise, they reflect the full model.

        See `Recommended joint and meta-analysis strategies for case-control association testing of single low-count variants <http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4049324/>`__ for an empirical comparison of the logistic Wald, LRT, score, and Firth tests. The theoretical foundations of the Wald, likelihood ratio, and score tests may be found in Chapter 3 of Gesine Reinert's notes `Statistical Theory <http://www.stats.ox.ac.uk/~reinert/stattheory/theoryshort09.pdf>`__.  Firth introduced his approach in `Bias reduction of maximum likelihood estimates, 1993 <http://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/GibbsFieldEst/BiasReductionMLE.pdf>`__. Heinze and Schemper further analyze Firth's approach in `A solution to the problem of separation in logistic regression, 2002 <https://cemsiis.meduniwien.ac.at/fileadmin/msi_akim/CeMSIIS/KB/volltexte/Heinze_Schemper_2002_Statistics_in_Medicine.pdf>`__.

        Those variants that don't vary across the included samples (e.g., all genotypes
        are HomRef) will have missing annotations.

        Phenotype and covariate sample annotations may also be specified using `programmatic expressions <exprlang.html>`__ without identifiers, such as:

        .. code-block:: text

            if (sa.isFemale) sa.cov.age else (2 * sa.cov.age + 10)

        For Boolean covariate types, true is coded as 1 and false as 0. In particular, for the sample annotation ``sa.fam.isCase`` added by importing a FAM file with case-control phenotype, case is 1 and control is 0.

        Hail's logistic regression tests correspond to the ``b.wald``, ``b.lrt``, and ``b.score`` tests in `EPACTS <http://genome.sph.umich.edu/wiki/EPACTS#Single_Variant_Tests>`__. For each variant, Hail imputes missing genotypes as the mean of called genotypes, whereas EPACTS subsets to those samples with called genotypes. Hence, Hail and EPACTS results will currently only agree for variants with no missing genotypes.

        :param str test: Statistical test, one of: 'wald', 'lrt', 'score', or 'firth'.

        :param str y: Response expression.  Must evaluate to Boolean or
            numeric with all values 0 or 1.

        :param covariates: list of covariate expressions
        :type covariates: list of str

        :param str root: Variant annotation path to store result of logistic regression.

        :param bool use_dosages: If true, use genotype dosage rather than hard call.

        :return: Variant dataset with logistic regression variant annotations.
        :rtype: :py:class:`.VariantDataset`
        """

        jvds = self._jvdf.logreg(test, y, jarray(Env.jvm().java.lang.String, covariates), root, use_dosages)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @typecheck_method(key_name=strlike,
                      variant_keys=strlike,
                      single_key=bool,
                      agg_expr=strlike,
                      test=strlike,
                      y=strlike,
                      covariates=listof(strlike))
    def logreg_burden(self, key_name, variant_keys, single_key, agg_expr, test, y, covariates=[]):
        r"""Test each keyed group of variants for association by aggregating (collapsing) genotypes and applying the
        logistic regression model.

        .. include:: requireTGenotype.rst

        **Examples**

        Run a gene burden test using the logistic Wald test on the maximum genotype per gene. Here ``va.genes`` is
        a variant annotation of type Set[String] giving the set of genes containing the variant
        (see **Extended example** in :py:meth:`.linreg_burden` for a deeper  dive in the context of linear regression):

        >>> logreg_kt, sample_kt = (hc.read('data/example_burden.vds')
        ...     .logreg_burden(key_name='gene',
        ...                    variant_keys='va.genes',
        ...                    single_key=False,
        ...                    agg_expr='gs.map(g => g.gt).max()',
        ...                    test='wald',
        ...                    y='sa.burden.pheno',
        ...                    covariates=['sa.burden.cov1', 'sa.burden.cov2']))

        Run a gene burden test using the logistic score test on the weighted sum of genotypes per gene.
        Here ``va.gene`` is a variant annotation of type String giving a single gene per variant (or no gene if
        missing), and ``va.weight`` is a numeric variant annotation:

        >>> logreg_kt, sample_kt = (hc.read('data/example_burden.vds')
        ...     .logreg_burden(key_name='gene',
        ...                    variant_keys='va.gene',
        ...                    single_key=True,
        ...                    agg_expr='gs.map(g => va.weight * g.gt).sum()',
        ...                    test='score',
        ...                    y='sa.burden.pheno',
        ...                    covariates=['sa.burden.cov1', 'sa.burden.cov2']))

        To use a weighted sum of genotypes with missing genotypes mean-imputed rather than ignored, set
        ``agg_expr='gs.map(g => va.weight * orElse(g.gt.toDouble, 2 * va.qc.AF)).sum()'`` where ``va.qc.AF``
        is the allele frequency over those samples that have no missing phenotype or covariates.

        .. caution::

          With ``single_key=False``, ``variant_keys`` expects a variant annotation of Set or Array type, in order to
          allow each variant to have zero, one, or more keys (for example, the same variant may appear in multiple
          genes). Unlike with type Set, if the same key appears twice in a variant annotation of type Array, then that
          variant will be counted twice in that key's group. With ``single_key=True``, ``variant_keys`` expects a
          variant annotation whose value is itself the key of interest. In bose cases, variants with missing keys are
          ignored.

        **Notes**

        This method modifies :py:meth:`.logreg` by replacing the genotype covariate per variant and sample with
        an aggregated (i.e., collapsed) score per key and sample. This numeric score is computed from the sample's
        genotypes and annotations over all variants with that key. The phenotype type must either be numeric
        (with all present values 0 or 1) or Boolean, in which case true and false are coded as 1 and 0, respectively.

        Hail supports the Wald test ('wald'), likelihood ratio test ('lrt'), Rao score test ('score'),
        and Firth test ('firth') as the ``test`` parameter. Conceptually, the method proceeds as follows:

        1) Filter to the set of samples for which all phenotype and covariates are defined.

        2) For each key and sample, aggregate genotypes across variants with that key to produce a numeric score.
           ``agg_expr`` must be of numeric type and has the following symbols are in scope:

           - ``s`` (*Sample*): sample
           - ``sa``: sample annotations
           - ``global``: global annotations
           - ``gs`` (*Aggregable[Genotype]*): aggregable of :ref:`genotype` for sample ``s``

           Note that ``v``, ``va``, and ``g`` are accessible through
           `Aggregable methods <https://hail.is/hail/types.html#aggregable>`_ on ``gs``.

           The resulting **sample key table** has key column ``key_name`` and a numeric column of scores for each sample
           named by the sample ID.

        3) For each key, fit the logistic regression model using the supplied phenotype, covariates, and test.
           The model and tests are those of :py:meth:`.logreg` with sample genotype ``gt`` replaced by the
           score in the sample key table. For each key, missing scores are mean-imputed across all samples.

           The resulting **logistic regression key table** has key column of type String given by the ``key_name``
           parameter and additional columns corresponding to the fields of the ``va.logreg`` schema given for ``test``
           in :py:meth:`.logreg`.

        :py:meth:`.logreg_burden` returns both the logistic regression key table and the sample key table.

        :param str key_name: Name to assign to key column of returned key tables.

        :param str variant_keys: Variant annotation path for the TArray or TSet of keys associated to each variant.

        :param bool single_key: if true, ``variant_keys`` is interpreted as a single (or missing) key per variant,
                                rather than as a collection of keys.

        :param str agg_expr: Sample aggregation expression (per key).

        :param str test: Statistical test, one of: 'wald', 'lrt', 'score', or 'firth'.

        :param str y: Response expression.

        :param covariates: list of covariate expressions.
        :type covariates: list of str

        :return: Tuple of logistic regression key table and sample aggregation key table.
        :rtype: (:py:class:`.KeyTable`, :py:class:`.KeyTable`)
        """

        r = self._jvdf.logregBurden(key_name, variant_keys, single_key, agg_expr, test, y, jarray(Env.jvm().java.lang.String, covariates))
        logreg_kt = KeyTable(self.hc, r._1())
        sample_kt = KeyTable(self.hc, r._2())

        return logreg_kt, sample_kt

    @handle_py4j
    @requireTGenotype
    @typecheck_method(pedigree=Pedigree)
    def mendel_errors(self, pedigree):
        """Find Mendel errors; count per variant, individual and nuclear
        family.

        .. include:: requireTGenotype.rst

        **Examples**

        Find all violations of Mendelian inheritance in each (dad,
        mom, kid) trio in a pedigree and return four tables:

        >>> ped = Pedigree.read('data/trios.fam')
        >>> all, per_fam, per_sample, per_variant = vds.mendel_errors(ped)
        
        Export all mendel errors to a text file:
        
        >>> all.export('output/all_mendel_errors.tsv')

        Annotate samples with the number of Mendel errors:
        
        >>> annotated_vds = vds.annotate_samples_table(per_sample, root="sa.mendel")
        
        Annotate variants with the number of Mendel errors:
        
        >>> annotated_vds = vds.annotate_variants_table(per_variant, root="va.mendel")
        
        **Notes**
        
        This method assumes all contigs apart from X and Y are fully autosomal;
        mitochondria, decoys, etc. are not given special treatment.

        The example above returns four tables, which contain Mendelian violations grouped in
        various ways. These tables are modeled after the 
        `PLINK mendel formats <https://www.cog-genomics.org/plink2/formats#mendel>`_. The four
        tables contain the following columns:
        
        **First table:** all Mendel errors. This table contains one row per Mendel error in the dataset;
        it is possible that a variant or sample may be found on more than one row. This table closely
        reflects the structure of the ".mendel" PLINK format detailed below.
        
        Columns:
        
            - **fid** (*String*) -- Family ID.
            - **s** (*String*) -- Proband ID.
            - **v** (*Variant*) -- Variant in which the error was found.
            - **code** (*Int*) -- Mendel error code, see below. 
            - **error** (*String*) -- Readable representation of Mendel error.
        
        **Second table:** errors per nuclear family. This table contains one row per nuclear family in the dataset.
        This table closely reflects the structure of the ".fmendel" PLINK format detailed below. 
        
        Columns:
        
            - **fid** (*String*) -- Family ID.
            - **father** (*String*) -- Paternal ID.
            - **mother** (*String*) -- Maternal ID.
            - **nChildren** (*Int*) -- Number of children in this nuclear family.
            - **nErrors** (*Int*) -- Number of Mendel errors in this nuclear family.
            - **nSNP** (*Int*) -- Number of Mendel errors at SNPs in this nuclear family.
        
        **Third table:** errors per individual. This table contains one row per individual in the dataset, 
        including founders. This table closely reflects the structure of the ".imendel" PLINK format detailed 
        below.
        
        Columns:
        
            - **s** (*String*) -- Sample ID (key column).
            - **fid** (*String*) -- Family ID.
            - **nErrors** (*Int*) -- Number of Mendel errors found involving this individual.
            - **nSNP** (*Int*) -- Number of Mendel errors found involving this individual at SNPs.
            - **error** (*String*) -- Readable representation of Mendel error.
        
        **Fourth table:** errors per variant. This table contains one row per variant in the dataset.
        
        Columns:
        
            - **v** (*Variant*) -- Variant (key column).
            - **nErrors** (*Int*) -- Number of Mendel errors in this variant.
        
        **PLINK Mendel error formats:**

            - ``*.mendel`` -- all mendel errors: FID KID CHR SNP CODE ERROR
            - ``*.fmendel`` -- error count per nuclear family: FID PAT MAT CHLD N
            - ``*.imendel`` -- error count per individual: FID IID N
            - ``*.lmendel`` -- error count per variant: CHR SNP N
        
        In the PLINK formats, **FID**, **KID**, **PAT**, **MAT**, and **IID** refer to family, kid,
        dad, mom, and individual ID, respectively, with missing values set to ``0``. SNP denotes 
        the variant identifier ``chr:pos:ref:alt``. N is the error count. CHLD is the number of 
        children in a nuclear family.

        The CODE of each Mendel error is determined by the table below,
        extending the `Plink
        classification <https://www.cog-genomics.org/plink2/basic_stats#mendel>`__.

        Those individuals implicated by each code are in bold.

        The copy state of a locus with respect to a trio is defined as follows,
        where PAR is the `pseudoautosomal region <https://en.wikipedia.org/wiki/Pseudoautosomal_region>`__ (PAR).

        - HemiX -- in non-PAR of X, male child
        - HemiY -- in non-PAR of Y, male child
        - Auto -- otherwise (in autosome or PAR, or female child)

        Any refers to :math:`\{ HomRef, Het, HomVar, NoCall \}` and ! denotes complement in this set.

        +--------+------------+------------+----------+------------------+
        |Code    | Dad        | Mom        |     Kid  |   Copy State     |
        +========+============+============+==========+==================+
        |    1   | HomVar     | HomVar     | Het      | Auto             |
        +--------+------------+------------+----------+------------------+
        |    2   | HomRef     | HomRef     | Het      | Auto             |
        +--------+------------+------------+----------+------------------+
        |    3   | HomRef     |  ! HomRef  |  HomVar  | Auto             |
        +--------+------------+------------+----------+------------------+
        |    4   |  ! HomRef  | HomRef     |  HomVar  | Auto             |
        +--------+------------+------------+----------+------------------+
        |    5   | HomRef     | HomRef     |  HomVar  | Auto             |
        +--------+------------+------------+----------+------------------+
        |    6   | HomVar     |  ! HomVar  |  HomRef  | Auto             |
        +--------+------------+------------+----------+------------------+
        |    7   |  ! HomVar  | HomVar     |  HomRef  | Auto             |
        +--------+------------+------------+----------+------------------+
        |    8   | HomVar     | HomVar     |  HomRef  | Auto             |
        +--------+------------+------------+----------+------------------+
        |    9   | Any        | HomVar     |  HomRef  | HemiX            |
        +--------+------------+------------+----------+------------------+
        |   10   | Any        | HomRef     |  HomVar  | HemiX            |
        +--------+------------+------------+----------+------------------+
        |   11   | HomVar     | Any        |  HomRef  | HemiY            |
        +--------+------------+------------+----------+------------------+
        |   12   | HomRef     | Any        |  HomVar  | HemiY            |
        +--------+------------+------------+----------+------------------+

        This method only considers children with two parents and a defined sex.

        PAR is currently defined with respect to reference
        `GRCh37 <http://www.ncbi.nlm.nih.gov/projects/genome/assembly/grc/human/>`__:

        - X: 60001 - 2699520, 154931044 - 155260560
        - Y: 10001 - 2649520, 59034050 - 59363566

        :param pedigree: Sample pedigree.
        :type pedigree: :class:`~hail.representation.Pedigree`

        :returns: Four tables with Mendel error statistics.
        :rtype: (:class:`.KeyTable`, :class:`.KeyTable`, :class:`.KeyTable`, :class:`.KeyTable`)
        """

        kts = self._jvdf.mendelErrors(pedigree._jrep)
        return KeyTable(self.hc, kts._1()), KeyTable(self.hc, kts._2()), \
               KeyTable(self.hc, kts._3()), KeyTable(self.hc, kts._4())

    @handle_py4j
    @typecheck_method(max_shift=integral)
    def min_rep(self, max_shift=100):
        """
        Gives minimal, left-aligned representation of alleles. Note that this can change the variant position.

        **Examples**

        1. Simple trimming of a multi-allelic site, no change in variant position
        `1:10000:TAA:TAA,AA` => `1:10000:TA:T,A`

        2. Trimming of a bi-allelic site leading to a change in position
        `1:10000:AATAA,AAGAA` => `1:10002:T:G`

        :param int max_shift: maximum number of base pairs by which
          a split variant can move.  Affects memory usage, and will
          cause Hail to throw an error if a variant that moves further
          is encountered.

        :rtype: :class:`.VariantDataset`
        """

        jvds = self._jvds.minRep(max_shift)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @requireTGenotype
    @typecheck_method(scores=strlike,
                      loadings=nullable(strlike),
                      eigenvalues=nullable(strlike),
                      k=integral,
                      as_array=bool)
    def pca(self, scores, loadings=None, eigenvalues=None, k=10, as_array=False):
        """Run Principal Component Analysis (PCA) on the matrix of genotypes.

        .. include:: requireTGenotype.rst

        **Examples**

        Compute the top 5 principal component scores, stored as sample annotations ``sa.scores.PC1``, ..., ``sa.scores.PC5`` of type Double:

        >>> vds_result = vds.pca('sa.scores', k=5)

        Compute the top 5 principal component scores, loadings, and eigenvalues, stored as annotations ``sa.scores``, ``va.loadings``, and ``global.evals`` of type Array[Double]:

        >>> vds_result = vds.pca('sa.scores', 'va.loadings', 'global.evals', 5, as_array=True)

        **Notes**

        Hail supports principal component analysis (PCA) of genotype data, a now-standard procedure `Patterson, Price and Reich, 2006 <http://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.0020190>`__. This method expects a variant dataset with biallelic autosomal variants. Scores are computed and stored as sample annotations of type Struct by default; variant loadings and eigenvalues can optionally be computed and stored in variant and global annotations, respectively.

        PCA is based on the singular value decomposition (SVD) of a standardized genotype matrix :math:`M`, computed as follows. An :math:`n \\times m` matrix :math:`C` records raw genotypes, with rows indexed by :math:`n` samples and columns indexed by :math:`m` bialellic autosomal variants; :math:`C_{ij}` is the number of alternate alleles of variant :math:`j` carried by sample :math:`i`, which can be 0, 1, 2, or missing. For each variant :math:`j`, the sample alternate allele frequency :math:`p_j` is computed as half the mean of the non-missing entries of column :math:`j`. Entries of :math:`M` are then mean-centered and variance-normalized as

        .. math::

          M_{ij} = \\frac{C_{ij}-2p_j}{\sqrt{2p_j(1-p_j)m}},

        with :math:`M_{ij} = 0` for :math:`C_{ij}` missing (i.e. mean genotype imputation). This scaling normalizes genotype variances to a common value :math:`1/m` for variants in Hardy-Weinberg equilibrium and is further motivated in the paper cited above. (The resulting amplification of signal from the low end of the allele frequency spectrum will also introduce noise for rare variants; common practice is to filter out variants with minor allele frequency below some cutoff.)  The factor :math:`1/m` gives each sample row approximately unit total variance (assuming linkage equilibrium) and yields the sample correlation or genetic relationship matrix (GRM) as simply :math:`MM^T`.

        PCA then computes the SVD

        .. math::

          M = USV^T

        where columns of :math:`U` are left singular vectors (orthonormal in :math:`\mathbb{R}^n`), columns of :math:`V` are right singular vectors (orthonormal in :math:`\mathbb{R}^m`), and :math:`S=\mathrm{diag}(s_1, s_2, \ldots)` with ordered singular values :math:`s_1 \ge s_2 \ge \cdots \ge 0`. Typically one computes only the first :math:`k` singular vectors and values, yielding the best rank :math:`k` approximation :math:`U_k S_k V_k^T` of :math:`M`; the truncations :math:`U_k`, :math:`S_k` and :math:`V_k` are :math:`n \\times k`, :math:`k \\times k` and :math:`m \\times k` respectively.

        From the perspective of the samples or rows of :math:`M` as data, :math:`V_k` contains the variant loadings for the first :math:`k` PCs while :math:`MV_k = U_k S_k` contains the first :math:`k` PC scores of each sample. The loadings represent a new basis of features while the scores represent the projected data on those features. The eigenvalues of the GRM :math:`MM^T` are the squares of the singular values :math:`s_1^2, s_2^2, \ldots`, which represent the variances carried by the respective PCs. By default, Hail only computes the loadings if the ``loadings`` parameter is specified.

        *Note:* In PLINK/GCTA the GRM is taken as the starting point and it is computed slightly differently with regard to missing data. Here the :math:`ij` entry of :math:`MM^T` is simply the dot product of rows :math:`i` and :math:`j` of :math:`M`; in terms of :math:`C` it is

        .. math::

          \\frac{1}{m}\sum_{l\in\mathcal{C}_i\cap\mathcal{C}_j}\\frac{(C_{il}-2p_l)(C_{jl} - 2p_l)}{2p_l(1-p_l)}

        where :math:`\mathcal{C}_i = \{l \mid C_{il} \\text{ is non-missing}\}`. In PLINK/GCTA the denominator :math:`m` is replaced with the number of terms in the sum :math:`\\lvert\mathcal{C}_i\cap\\mathcal{C}_j\\rvert`, i.e. the number of variants where both samples have non-missing genotypes. While this is arguably a better estimator of the true GRM (trading shrinkage for noise), it has the drawback that one loses the clean interpretation of the loadings and scores as features and projections.

        Separately, for the PCs PLINK/GCTA output the eigenvectors of the GRM; even ignoring the above discrepancy that means the left singular vectors :math:`U_k` instead of the component scores :math:`U_k S_k`. While this is just a matter of the scale on each PC, the scores have the advantage of representing true projections of the data onto features with the variance of a score reflecting the variance explained by the corresponding feature. (In PC bi-plots this amounts to a change in aspect ratio; for use of PCs as covariates in regression it is immaterial.)

        **Annotations**

        Given root ``scores='sa.scores'`` and ``as_array=False``, :py:meth:`~hail.VariantDataset.pca` adds a Struct to sample annotations:

         - **sa.scores** (*Struct*) -- Struct of sample scores

        With ``k=3``, the Struct has three field:

         - **sa.scores.PC1** (*Double*) -- Score from first PC

         - **sa.scores.PC2** (*Double*) -- Score from second PC

         - **sa.scores.PC3** (*Double*) -- Score from third PC

        Analogous variant and global annotations of type Struct are added by specifying the ``loadings`` and ``eigenvalues`` arguments, respectively.

        Given roots ``scores='sa.scores'``, ``loadings='va.loadings'``, and ``eigenvalues='global.evals'``, and ``as_array=True``, :py:meth:`~hail.VariantDataset.pca` adds the following annotations:

         - **sa.scores** (*Array[Double]*) -- Array of sample scores from the top k PCs

         - **va.loadings** (*Array[Double]*) -- Array of variant loadings in the top k PCs

         - **global.evals** (*Array[Double]*) -- Array of the top k eigenvalues

        :param str scores: Sample annotation path to store scores.

        :param loadings: Variant annotation path to store site loadings.
        :type loadings: str or None

        :param eigenvalues: Global annotation path to store eigenvalues.
        :type eigenvalues: str or None

        :param k: Number of principal components.
        :type k: int or None

        :param bool as_array: Store annotations as type Array rather than Struct
        :type k: bool or None

        :return: Dataset with new PCA annotations.
        :rtype: :class:`.VariantDataset`
        """

        jvds = self._jvdf.pca(scores, k, joption(loadings), joption(eigenvalues), as_array)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @typecheck_method(k=integral,
                      maf=numeric,
                      block_size=integral)
    def pc_relate(self, k, maf, block_size=512):
        """Compute relatedness estimates between individuals using a variant of the
        PC-Relate method.

        .. include:: experimental.rst

        **Examples**

        Estimate kinship, identity-by-descent two, identity-by-descent one, and
        identity-by-descent zero for every pair of samples, using 5 prinicpal
        components to correct for ancestral populations, and a minimum minor
        allele frequency filter of 0.01:

        >>> rel = vds.pc_relate(5, 0.01)

        Calculate values as above, but when performing distributed matrix
        multiplications use a matrix-block-size of 1024 by 1024.

        >>> rel = vds.pc_relate(5, 0.01, 1024)

        **Method**

        The traditional estimator for kinship between a pair of individuals
        :math:`i` and :math:`j`, sharing the set :math:`S_{ij}` of
        single-nucleotide variants, from a population with allele frequencies
        :math:`p_s`, is given by:

        .. math::

          \\widehat{\phi_{ij}} := \\frac{1}{|S_{ij}|}\\sum_{s \in S_{ij}}\\frac{(g_{is} - 2 p_s) (g_{js} - 2 p_s)}{4 * \sum_{s \in S_{ij} p_s (1 - p_s)}}

        This estimator is true under the model that the sharing of common
        (relative to the population) alleles is not very informative to
        relatedness (because they're common) and the sharing of rare alleles
        suggests a recent common ancestor from which the allele was inherited by
        descent.

        When multiple ancestry groups are mixed in a sample, this model breaks
        down. Alleles that are rare in all but one ancestry group are treated as
        very informative to relatedness. However, these alleles are simply
        markers of the ancestry group. The PC-Relate method corrects for this
        situation and the related situation of admixed individuals.

        PC-Relate slightly modifies the usual estimator for relatedness:
        occurences of population allele frequency are replaced with an
        "individual-specific allele frequency". This modification allows the
        method to correctly weight an allele according to an individual's unique
        ancestry profile.

        The "individual-specific allele frequency" at a given genetic locus is
        modeled by PC-Relate as a linear function of their first ``k`` principal
        component coordinates. As such, the efficacy of this method rests on two
        assumptions:

         - an individual's first ``k`` principal component coordinates fully
           describe their allele-frequency-relevant ancestry, and

         - the relationship between ancestry (as described by principal
           component coordinates) and population allele frequency is linear

        The estimators for kinship, and identity-by-descent zero, one, and two
        follow. Let:

         - :math:`S_{ij}` be the set of genetic loci at which both individuals
           :math:`i` and :math:`j` have a defined genotype

         - :math:`g_{is} \in {0, 1, 2}` be the number of alternate alleles that
           individual :math:`i` has at gentic locus :math:`s`

         - :math:`\\widehat{\\mu_{is}} \in [0, 1]` be the individual-specific allele
           frequency for individual :math:`i` at genetic locus :math:`s`

         - :math:`{\\widehat{\\sigma^2_{is}}} := \\widehat{\\mu_{is}} (1 -
           \\widehat{\\mu_{is}})`, the binomial variance of
           :math:`\\widehat{\\mu_{is}}`

         - :math:`\\widehat{\\sigma_{is}} := \sqrt{\\widehat{\\sigma^2_{is}}}`,
           the binomial standard deviation of :math:`\\widehat{\\mu_{is}}`

         - :math:`\\text{IBS}^{(0)}_{ij} := \\sum_{s \\in S_{ij}} \\mathbb{1}_{||g_{is} -
           g_{js} = 2||}`, the number of genetic loci at which individuals
           :math:`i` and :math:`j` share no alleles

         - :math:`\\widehat{f_i} := 2 \\widehat{\phi_{ii}} - 1`, the inbreeding
           coefficient for individual :math:`i`

         - :math:`g^D_{is}` be a dominance encoding of the genotype matrix, and
           :math:`X_{is}` be a normalized dominance-coded genotype matrix

        .. math::

          g^D_{is} :=
            \\begin{cases}
              \\widehat{\\mu_{is}}     & g_{is} = 0 \\\\
              0                        & g_{is} = 1 \\\\
              1 - \\widehat{\\mu_{is}} & g_{is} = 2
            \\end{cases}

          X_{is} := g^D_{is} - \\widehat{\\sigma^2_{is}} (1 - \\widehat{f_i})

        The estimator for kinship is given by:

        .. math::

          \\widehat{\phi_{ij}} := \\frac{\sum_{s \in S_{ij}}(g - 2 \\mu)_{is} (g - 2 \\mu)_{js}}{4 * \sum_{s \in S_{ij}}\\widehat{\\sigma_{is}} \\widehat{\\sigma_{js}}}

        The estimator for identity-by-descent two is given by:

        .. math::

          \\widehat{k^{(2)}_{ij}} := \\frac{\sum_{s \in S_{ij}}X_{is} X_{js}}{\sum_{s \in S_{ij}}\\widehat{\\sigma^2_{is}} \\widehat{\\sigma^2_{js}}}

        The estimator for identity-by-descent zero is given by:

        .. math::

          \\widehat{k^{(0)}_{ij}} :=
            \\begin{cases}
              \\frac{\\text{IBS}^{(0)}_{ij}}
                   {\sum_{s \in S_{ij}} \\widehat{\\mu_{is}}^2(1 - \\widehat{\\mu_{js}})^2 + (1 - \\widehat{\\mu_{is}})^2\\widehat{\\mu_{js}}^2}
                & \\widehat{\phi_{ij}} > 2^{-5/2} \\\\
              1 - 4 \\widehat{\phi_{ij}} + k^{(2)}_{ij}
                & \\widehat{\phi_{ij}} \le 2^{-5/2}
            \\end{cases}

        The estimator for identity-by-descent one is given by:

        .. math::

          \\widehat{k^{(1)}_{ij}} := 1 - \\widehat{k^{(2)}_{ij}} - \\widehat{k^{(0)}_{ij}}

        **Details**

        The PC-Relate method is described in "Model-free Estimation of Recent
        Genetic Relatedness". Conomos MP, Reiner AP, Weir BS, Thornton TA. in
        American Journal of Human Genetics. 2016 Jan 7. The reference
        implementation is available in the `GENESIS Bioconductor package
        <https://bioconductor.org/packages/release/bioc/html/GENESIS.html>`_ .

        :py:meth:`~hail.VariantDataset.pc_relate` differs from the reference
        implementation in a couple key ways:

         - the principal components analysis does not use an unrelated set of
           individuals

         - the estimators do not perform small sample correction

         - the algorithm does not provide an option to use population-wide
           allele frequency estimates

         - the algorithm does not provide an option to not use "overall
           standardization" (see R ``pcrelate`` documentation)

        **Notes**

        The ``block_size`` controls memory usage and parallelism. If it is large
        enough to hold an entire sample-by-sample matrix of 64-bit doubles in
        memory, then only one Spark worker node can be used to compute matrix
        operations. If it is too small, communication overhead will begin to
        dominate the computation's time. The author has found that on Google
        Dataproc (where each core has about 3.75GB of memory), setting
        ``block_size`` larger than 512 tends to cause memory exhaustion errors.

        The minimum allele frequency filter is applied per-pair: if either of
        the two individual's individual-specific minor allele frequency is below
        the threshold, then the variant's contribution to relatedness estimates
        is zero.

        The resulting :py:class:`.KeyTable` entries have the type: *{ i: String,
        j: String, kin: Double, k2: Double, k1: Double, k0: Double }*. The key
        list is: *i: String, j: String*.

        :param int k: The number of principal components to use to distinguish
                      ancestries.

        :param float maf: The minimum individual-specific allele frequency for
                          an allele used to measure relatedness.

        :param int block_size: the side length of the blocks of the block-
                               distributed matrices; this should be set such
                               that at least three of these matrices fit in
                               memory (in addition to all other objects
                               necessary for Spark and Hail)

        :return: A :py:class:`.KeyTable` mapping pairs of samples to estimations
                 of their kinship and identity-by-descent zero, one, and two
        :rtype: :py:class:`.KeyTable`

        """

        return KeyTable(self.hc, self._jvdf.pcRelate(k, maf, block_size))

    @handle_py4j
    @typecheck_method(storage_level=strlike)
    def persist(self, storage_level="MEMORY_AND_DISK"):
        """Persist this variant dataset to memory and/or disk.

        **Examples**

        Persist the variant dataset to both memory and disk:

        >>> vds_result = vds.persist()

        **Notes**

        The :py:meth:`~hail.VariantDataset.persist` and :py:meth:`~hail.VariantDataset.cache` methods 
        allow you to store the current dataset on disk or in memory to avoid redundant computation and 
        improve the performance of Hail pipelines.

        :py:meth:`~hail.VariantDataset.cache` is an alias for 
        :func:`persist("MEMORY_ONLY") <hail.VariantDataset.persist>`.  Most users will want "MEMORY_AND_DISK".
        See the `Spark documentation <http://spark.apache.org/docs/latest/programming-guide.html#rdd-persistence>`__ 
        for a more in-depth discussion of persisting data.
        
        .. warning ::
            
            Persist, like all other :class:`.VariantDataset` functions, is functional.
            Its output must be captured. This is wrong:
            
            >>> vds = vds.linreg('sa.phenotype') # doctest: +SKIP
            >>> vds.persist() # doctest: +SKIP
            
            The above code does NOT persist ``vds``. Instead, it copies ``vds`` and persists that result. 
            The proper usage is this:
            
            >>> vds = vds.pca().persist() # doctest: +SKIP

        :param storage_level: Storage level.  One of: NONE, DISK_ONLY,
            DISK_ONLY_2, MEMORY_ONLY, MEMORY_ONLY_2, MEMORY_ONLY_SER,
            MEMORY_ONLY_SER_2, MEMORY_AND_DISK, MEMORY_AND_DISK_2,
            MEMORY_AND_DISK_SER, MEMORY_AND_DISK_SER_2, OFF_HEAP
            
        :rtype: :class:`.VariantDataset`
        """

        return VariantDataset(self.hc, self._jvdf.persist(storage_level))

    def unpersist(self):
        """
        Unpersists this VDS from memory/disk.
        
        **Notes**
        This function will have no effect on a VDS that was not previously persisted.
        
        There's nothing stopping you from continuing to use a VDS that has been unpersisted, but doing so will result in
        all previous steps taken to compute the VDS being performed again since the VDS must be recomputed. Only unpersist
        a VDS when you are done with it.
         
        """
        self._jvds.unpersist()

    @property
    @handle_py4j
    def global_schema(self):
        """
        Returns the signature of the global annotations contained in this VDS.

        **Examples**

        >>> print(vds.global_schema)

        The ``pprint`` module can be used to print the schema in a more human-readable format:

        >>> from pprint import pprint
        >>> pprint(vds.global_schema)


        :rtype: :class:`.Type`
        """

        if self._global_schema is None:
            self._global_schema = Type._from_java(self._jvds.globalSignature())
        return self._global_schema

    @property
    @handle_py4j
    def colkey_schema(self):
        """
        Returns the signature of the column key (sample) contained in this VDS.

        **Examples**

        >>> print(vds.colkey_schema)

        The ``pprint`` module can be used to print the schema in a more human-readable format:

        >>> from pprint import pprint
        >>> pprint(vds.colkey_schema)

        :rtype: :class:`.Type`
        """

        if self._colkey_schema is None:
            self._colkey_schema = Type._from_java(self._jvds.sSignature())
        return self._colkey_schema

    @property
    @handle_py4j
    def sample_schema(self):
        """
        Returns the signature of the sample annotations contained in this VDS.

        **Examples**

        >>> print(vds.sample_schema)

        The ``pprint`` module can be used to print the schema in a more human-readable format:

        >>> from pprint import pprint
        >>> pprint(vds.sample_schema)

        :rtype: :class:`.Type`
        """

        if self._sa_schema is None:
            self._sa_schema = Type._from_java(self._jvds.saSignature())
        return self._sa_schema

    @property
    @handle_py4j
    def rowkey_schema(self):
        """
        Returns the signature of the row key (variant) contained in this VDS.

        **Examples**

        >>> print(vds.rowkey_schema)

        The ``pprint`` module can be used to print the schema in a more human-readable format:

        >>> from pprint import pprint
        >>> pprint(vds.rowkey_schema)

        :rtype: :class:`.Type`
        """

        if self._rowkey_schema is None:
            self._rowkey_schema = Type._from_java(self._jvds.vSignature())
        return self._rowkey_schema

    @property
    @handle_py4j
    def variant_schema(self):
        """
        Returns the signature of the variant annotations contained in this VDS.

        **Examples**

        >>> print(vds.variant_schema)

        The ``pprint`` module can be used to print the schema in a more human-readable format:

        >>> from pprint import pprint
        >>> pprint(vds.variant_schema)

        :rtype: :class:`.Type`
        """

        if self._va_schema is None:
            self._va_schema = Type._from_java(self._jvds.vaSignature())
        return self._va_schema

    @property
    @handle_py4j
    def genotype_schema(self):
        """
        Returns the signature of the genotypes contained in this VDS.

        **Examples**

        >>> print(vds.genotype_schema)

        The ``pprint`` module can be used to print the schema in a more human-readable format:

        >>> from pprint import pprint
        >>> pprint(vds.genotype_schema)

        :rtype: :class:`.Type`
        """

        if self._genotype_schema is None:
            self._genotype_schema = Type._from_java(self._jvds.genotypeSignature())
        return self._genotype_schema

    @handle_py4j
    @typecheck_method(exprs=oneof(strlike, listof(strlike)))
    def query_samples_typed(self, exprs):
        """Performs aggregation queries over samples and sample annotations, and returns Python object(s) and type(s).

        **Examples**

        >>> low_callrate_samples, t = vds.query_samples_typed(
        ...    'samples.filter(s => sa.qc.callRate < 0.95).collect()')

        See :py:meth:`.query_samples` for more information.

        :param exprs: query expressions
        :type exprs: str or list of str

        :rtype: (annotation or list of annotation,  :class:`.Type` or list of :class:`.Type`)
        """

        if isinstance(exprs, list):
            result_list = self._jvds.querySamples(jarray(Env.jvm().java.lang.String, exprs))
            ptypes = [Type._from_java(x._2()) for x in result_list]
            annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in xrange(len(ptypes))]
            return annotations, ptypes
        else:
            result = self._jvds.querySamples(exprs)
            t = Type._from_java(result._2())
            return t._convert_to_py(result._1()), t

    @handle_py4j
    @typecheck_method(exprs=oneof(strlike, listof(strlike)))
    def query_samples(self, exprs):
        """Performs aggregation queries over samples and sample annotations, and returns Python object(s).

        **Examples**

        >>> low_callrate_samples = vds.query_samples('samples.filter(s => sa.qc.callRate < 0.95).collect()')

        **Notes**

        This method evaluates Hail expressions over samples and sample
        annotations.  The ``exprs`` argument requires either a single string
        or a list of strings. If a single string was passed, then a single
        result is returned. If a list is passed, a list is returned.


        The namespace of the expressions includes:

        - ``global``: global annotations
        - ``samples`` (*Aggregable[Sample]*): aggregable of sample

        Map and filter expressions on this aggregable have the additional
        namespace:

        - ``global``: global annotations
        - ``s``: sample
        - ``sa``: sample annotations

        :param exprs: query expressions
        :type exprs: str or list of str

        :rtype: annotation or list of annotation
        """

        r, t = self.query_samples_typed(exprs)
        return r

    @handle_py4j
    @typecheck_method(exprs=oneof(strlike, listof(strlike)))
    def query_variants_typed(self, exprs):
        """Performs aggregation queries over variants and variant annotations, and returns Python object(s) and type(s).

        **Examples**

        >>> lof_variant_count, t = vds.query_variants_typed(
        ...     'variants.filter(v => va.consequence == "LOF").count()')

        >>> [lof_variant_count, missense_count], [t1, t2] = vds.query_variants_typed([
        ...     'variants.filter(v => va.consequence == "LOF").count()',
        ...     'variants.filter(v => va.consequence == "Missense").count()'])

        See :py:meth:`.query_variants` for more information.

        :param exprs: query expressions
        :type exprs: str or list of str

        :rtype: (annotation or list of annotation, :class:`.Type` or list of :class:`.Type`)
        """
        if isinstance(exprs, list):
            result_list = self._jvds.queryVariants(jarray(Env.jvm().java.lang.String, exprs))
            ptypes = [Type._from_java(x._2()) for x in result_list]
            annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in xrange(len(ptypes))]
            return annotations, ptypes

        else:
            result = self._jvds.queryVariants(exprs)
            t = Type._from_java(result._2())
            return t._convert_to_py(result._1()), t

    @handle_py4j
    @typecheck_method(exprs=oneof(strlike, listof(strlike)))
    def query_variants(self, exprs):
        """Performs aggregation queries over variants and variant annotations, and returns Python object(s).

        **Examples**

        >>> lof_variant_count = vds.query_variants('variants.filter(v => va.consequence == "LOF").count()')

        >>> [lof_variant_count, missense_count] = vds.query_variants([
        ...     'variants.filter(v => va.consequence == "LOF").count()',
        ...     'variants.filter(v => va.consequence == "Missense").count()'])

        **Notes**

        This method evaluates Hail expressions over variants and variant
        annotations.  The ``exprs`` argument requires either a single string
        or a list of strings. If a single string was passed, then a single
        result is returned. If a list is passed, a list is returned.


        The namespace of the expressions includes:

        - ``global``: global annotations
        - ``variants`` (*Aggregable[Variant]*): aggregable of :ref:`variant`

        Map and filter expressions on this aggregable have the additional
        namespace:

        - ``global``: global annotations
        - ``v``: :ref:`variant`
        - ``va``: variant annotations

        **Performance Note**
        It is far faster to execute multiple queries in one method than
        to execute multiple query methods.  The combined query:

        >>> exprs = ['variants.count()', 'variants.filter(v => v.altAllele.isSNP()).count()']
        >>> [num_variants, num_snps] = vds.query_variants(exprs)

        will be nearly twice as fast as the split query:

        >>> result1 = vds.query_variants('variants.count()')
        >>> result2 = vds.query_variants('variants.filter(v => v.altAllele.isSNP()).count()')

        :param exprs: query expressions
        :type exprs: str or list of str

        :rtype: annotation or list of annotation
        """

        r, t = self.query_variants_typed(exprs)
        return r

    @handle_py4j
    @typecheck_method(exprs=oneof(strlike, listof(strlike)))
    def query_genotypes_typed(self, exprs):
        """Performs aggregation queries over genotypes, and returns Python object(s) and type(s).

        **Examples**

        >>> gq_hist, t = vds.query_genotypes_typed('gs.map(g => g.gq).hist(0, 100, 100)')

        >>> [gq_hist, dp_hist], [t1, t2] = vds.query_genotypes_typed(['gs.map(g => g.gq).hist(0, 100, 100)',
        ...                                                           'gs.map(g => g.dp).hist(0, 60, 60)'])

        See :py:meth:`.query_genotypes` for more information.

        This method evaluates Hail expressions over genotypes, along with
        all variant and sample metadata for that genotype. The ``exprs``
        argument requires either a list of strings or a single string
        The method returns a list of results and a list of types (which
        each contain one element if the input parameter was a single str).

        The namespace of the expressions includes:

        - ``global``: global annotations
        - ``gs`` (*Aggregable[Genotype]*): aggregable of :ref:`genotype`

        Map and filter expressions on this aggregable have the following
        namespace:

        - ``global``: global annotations
        - ``g``: :ref:`genotype`
        - ``v``: :ref:`variant`
        - ``va``: variant annotations
        - ``s``: sample
        - ``sa``: sample annotations

        **Performance Note**
        It is far faster to execute multiple queries in one method than
        to execute multiple query methods.  This:

        >>> result1 = vds.query_genotypes('gs.count()')
        >>> result2 = vds.query_genotypes('gs.filter(g => v.altAllele.isSNP() && g.isHet).count()')

        will be nearly twice as slow as this:

        >>> exprs = ['gs.count()', 'gs.filter(g => v.altAllele.isSNP() && g.isHet).count()']
        >>> [geno_count, snp_hets] = vds.query_genotypes(exprs)

        :param exprs: query expressions
        :type exprs: str or list of str

        :rtype: (annotation or list of annotation, :class:`.Type` or list of :class:`.Type`)
        """

        if isinstance(exprs, list):
            result_list = self._jvds.queryGenotypes(jarray(Env.jvm().java.lang.String, exprs))
            ptypes = [Type._from_java(x._2()) for x in result_list]
            annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in xrange(len(ptypes))]
            return annotations, ptypes
        else:
            result = self._jvds.queryGenotypes(exprs)
            t = Type._from_java(result._2())
            return t._convert_to_py(result._1()), t

    @handle_py4j
    @typecheck_method(exprs=oneof(strlike, listof(strlike)))
    def query_genotypes(self, exprs):
        """Performs aggregation queries over genotypes, and returns Python object(s).

        **Examples**

        Compute global GQ histogram

        >>> gq_hist = vds.query_genotypes('gs.map(g => g.gq).hist(0, 100, 100)')

        Compute call rate

        >>> call_rate = vds.query_genotypes('gs.fraction(g => g.isCalled)')

        Compute GQ and DP histograms

        >>> [gq_hist, dp_hist] = vds.query_genotypes(['gs.map(g => g.gq).hist(0, 100, 100)',
        ...                                                     'gs.map(g => g.dp).hist(0, 60, 60)'])


        :param exprs: query expressions
        :type exprs: str or list of str

        :rtype: annotation or list of annotation
        """

        r, t = self.query_genotypes_typed(exprs)
        return r

    @handle_py4j
    @typecheck_method(mapping=dictof(strlike, strlike))
    def rename_samples(self, mapping):
        """Rename samples.

        **Examples**

        >>> vds_result = vds.rename_samples({'ID1': 'id1', 'ID2': 'id2'})

        Use a file with an "old_id" and "new_id" column to rename samples:

        >>> mapping_table = hc.import_table('data/sample_mapping.txt')
        >>> mapping_dict = {row.old_id: row.new_id for row in mapping_table.collect()}
        >>> vds_result = vds.rename_samples(mapping_dict)

        :param dict mapping: Mapping from old to new sample IDs.

        :return: Dataset with remapped sample IDs.
        :rtype: :class:`.VariantDataset`
        """

        jvds = self._jvds.renameSamples(mapping)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @typecheck_method(num_partitions=integral,
                      shuffle=bool)
    def repartition(self, num_partitions, shuffle=True):
        """Increase or decrease the number of variant dataset partitions.

        **Examples**

        Repartition the variant dataset to have 500 partitions:

        >>> vds_result = vds.repartition(500)

        **Notes**

        Check the current number of partitions with :py:meth:`.num_partitions`.

        The data in a variant dataset is divided into chunks called partitions, which may be stored together or across a network, so that each partition may be read and processed in parallel by available cores. When a variant dataset with :math:`M` variants is first imported, each of the :math:`k` partition will contain about :math:`M/k` of the variants. Since each partition has some computational overhead, decreasing the number of partitions can improve performance after significant filtering. Since it's recommended to have at least 2 - 4 partitions per core, increasing the number of partitions can allow one to take advantage of more cores.

        Partitions are a core concept of distributed computation in Spark, see `here <http://spark.apache.org/docs/latest/programming-guide.html#resilient-distributed-datasets-rdds>`__ for details. With ``shuffle=True``, Hail does a full shuffle of the data and creates equal sized partitions. With ``shuffle=False``, Hail combines existing partitions to avoid a full shuffle. These algorithms correspond to the ``repartition`` and ``coalesce`` commands in Spark, respectively. In particular, when ``shuffle=False``, ``num_partitions`` cannot exceed current number of partitions.

        :param int num_partitions: Desired number of partitions, must be less than the current number if ``shuffle=False``

        :param bool shuffle: If true, use full shuffle to repartition.

        :return: Variant dataset with the number of partitions equal to at most ``num_partitions``
        :rtype: :class:`.VariantDataset`
        """

        jvds = self._jvdf.coalesce(num_partitions, shuffle)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @typecheck_method(max_partitions=integral)
    def naive_coalesce(self, max_partitions):
        """Naively descrease the number of partitions.

        .. warning ::

          :py:meth:`~hail.VariantDataset.naive_coalesce` simply combines adjacent partitions to achieve the desired number.  It does not attempt to rebalance, unlike :py:meth:`~hail.VariantDataset.repartition`, so it can produce a heavily unbalanced dataset.  An unbalanced dataset can be inefficient to operate on because the work is not evenly distributed across partitions.

        :param int max_partitions: Desired number of partitions.  If the current number of partitions is less than ``max_partitions``, do nothing.

        :return: Variant dataset with the number of partitions equal to at most ``max_partitions``
        :rtype: :class:`.VariantDataset`
        """

        jvds = self._jvds.naiveCoalesce(max_partitions)
        return VariantDataset(self.hc, jvds)
    
    @handle_py4j
    @typecheck_method(force_block=bool,
                      force_gramian=bool)
    def rrm(self, force_block=False, force_gramian=False):
        """Computes the Realized Relationship Matrix (RRM).

        **Examples**

        >>> kinship_matrix = vds.rrm()

        **Notes**

        The Realized Relationship Matrix is defined as follows. Consider the :math:`n \\times m` matrix :math:`C` of raw genotypes, with rows indexed by :math:`n` samples and
        columns indexed by the :math:`m` bialellic autosomal variants; :math:`C_{ij}` is the number of alternate alleles of variant :math:`j` carried by sample :math:`i`, which
        can be 0, 1, 2, or missing. For each variant :math:`j`, the sample alternate allele frequency :math:`p_j` is computed as half the mean of the non-missing entries of column
        :math:`j`. Entries of :math:`M` are then mean-centered and variance-normalized as

        .. math::

          M_{ij} = \\frac{C_{ij}-2p_j}{\sqrt{\\frac{m}{n} \sum_{k=1}^n (C_{ij}-2p_j)^2}},

        with :math:`M_{ij} = 0` for :math:`C_{ij}` missing (i.e. mean genotype imputation). This scaling normalizes each variant column to have empirical variance :math:`1/m`, which gives each sample row approximately unit total variance (assuming linkage equilibrium) and yields the :math:`n \\times n` sample correlation or realized relationship matrix (RRM) :math:`K` as simply

        .. math::

          K = MM^T

        Note that the only difference between the Realized Relationship Matrix and the Genetic Relationship Matrix (GRM) used in :py:meth:`~hail.VariantDataset.grm` is the variant (column) normalization: where RRM uses empirical variance, GRM uses expected variance under Hardy-Weinberg Equilibrium.

        :param bool force_block: Force using Spark's BlockMatrix to compute kinship (advanced).

        :param bool force_gramian: Force using Spark's RowMatrix.computeGramian to compute kinship (advanced).

        :return: Realized Relationship Matrix for all samples.
        :rtype: :py:class:`KinshipMatrix`
        """
        return KinshipMatrix(self._jvdf.rrm(force_block, force_gramian))

    @handle_py4j
    @typecheck_method(other=vds_type,
                      tolerance=numeric)
    def same(self, other, tolerance=1e-6):
        """True if the two variant datasets have the same variants, samples, genotypes, and annotation schemata and values.

        **Examples**

        This will return True:

        >>> vds.same(vds)

        **Notes**

        The ``tolerance`` parameter sets the tolerance for equality when comparing floating-point fields. More precisely, :math:`x` and :math:`y` are equal if

        .. math::

            \abs{x - y} \leq tolerance * \max{\abs{x}, \abs{y}}

        :param other: variant dataset to compare against
        :type other: :class:`.VariantDataset`

        :param float tolerance: floating-point tolerance for equality

        :rtype: bool
        """

        return self._jvds.same(other._jvds, tolerance)

    @handle_py4j
    @requireTGenotype
    @typecheck_method(root=strlike,
                      keep_star=bool)
    def sample_qc(self, root='sa.qc', keep_star=False):
        """Compute per-sample QC metrics.

        .. include:: requireTGenotype.rst

        **Annotations**

        :py:meth:`~hail.VariantDataset.sample_qc` computes 20 sample statistics from the 
        genotype data and stores the results as sample annotations that can be accessed with
        ``sa.qc.<identifier>`` (or ``<root>.<identifier>`` if a non-default root was passed):

        +---------------------------+--------+----------------------------------------------------------+
        | Name                      | Type   | Description                                              |
        +===========================+========+==========================================================+
        | ``callRate``              | Double | Fraction of genotypes called                             |
        +---------------------------+--------+----------------------------------------------------------+
        | ``nHomRef``               | Int    | Number of homozygous reference genotypes                 |
        +---------------------------+--------+----------------------------------------------------------+
        | ``nHet``                  | Int    | Number of heterozygous genotypes                         |
        +---------------------------+--------+----------------------------------------------------------+
        | ``nHomVar``               | Int    | Number of homozygous alternate genotypes                 |
        +---------------------------+--------+----------------------------------------------------------+
        | ``nCalled``               | Int    | Sum of ``nHomRef`` + ``nHet`` + ``nHomVar``              |
        +---------------------------+--------+----------------------------------------------------------+
        | ``nNotCalled``            | Int    | Number of uncalled genotypes                             |
        +---------------------------+--------+----------------------------------------------------------+
        | ``nSNP``                  | Int    | Number of SNP alternate alleles                          |
        +---------------------------+--------+----------------------------------------------------------+
        | ``nInsertion``            | Int    | Number of insertion alternate alleles                    |
        +---------------------------+--------+----------------------------------------------------------+
        | ``nDeletion``             | Int    | Number of deletion alternate alleles                     |
        +---------------------------+--------+----------------------------------------------------------+
        | ``nSingleton``            | Int    | Number of private alleles                                |
        +---------------------------+--------+----------------------------------------------------------+
        | ``nTransition``           | Int    | Number of transition (A-G, C-T) alternate alleles        |
        +---------------------------+--------+----------------------------------------------------------+
        | ``nTransversion``         | Int    | Number of transversion alternate alleles                 |
        +---------------------------+--------+----------------------------------------------------------+
        | ``nNonRef``               | Int    | Sum of ``nHet`` and ``nHomVar``                          |
        +---------------------------+--------+----------------------------------------------------------+
        | ``rTiTv``                 | Double | Transition/Transversion ratio                            |
        +---------------------------+--------+----------------------------------------------------------+
        | ``rHetHomVar``            | Double | Het/HomVar genotype ratio                                |
        +---------------------------+--------+----------------------------------------------------------+
        | ``rInsertionDeletion``    | Double | Insertion/Deletion ratio                                 |
        +---------------------------+--------+----------------------------------------------------------+
        | ``dpMean``                | Double | Depth mean across all genotypes                          |
        +---------------------------+--------+----------------------------------------------------------+
        | ``dpStDev``               | Double | Depth standard deviation across all genotypes            |
        +---------------------------+--------+----------------------------------------------------------+
        | ``gqMean``                | Double | The average genotype quality across all genotypes        |
        +---------------------------+--------+----------------------------------------------------------+
        | ``gqStDev``               | Double | Genotype quality standard deviation across all genotypes |
        +---------------------------+--------+----------------------------------------------------------+

        Missing values ``NA`` may result (for example, due to division by zero) and are handled properly in filtering and written as "NA" in export modules. The empirical standard deviation is computed with zero degrees of freedom.

        :param str root: Sample annotation root for the computed struct.
        :param bool keep_star: Count star alleles as non-reference alleles
        
        :return: Annotated variant dataset with new sample qc annotations.
        :rtype: :class:`.VariantDataset`
        """

        return VariantDataset(self.hc, self._jvdf.sampleQC(root, keep_star))

    @handle_py4j
    def storage_level(self):
        """Returns the storage (persistence) level of the variant dataset.

        **Notes**

        See the `Spark documentation <http://spark.apache.org/docs/latest/programming-guide.html#rdd-persistence>`__ for details on persistence levels.

        :rtype: str
        """

        return self._jvds.storageLevel()

    @handle_py4j
    @requireTGenotype
    def summarize(self):
        """Returns a summary of useful information about the dataset.
        
        .. include:: requireTGenotype.rst

        
        **Examples**
        
        >>> s = vds.summarize()
        >>> print(s.contigs)
        >>> print('call rate is %.2f' % s.call_rate)
        >>> s.report()
        
        The following information is contained in the summary:
        
         - **samples** (*int*) - Number of samples.
         - **variants** (*int*) - Number of variants.
         - **call_rate** (*float*) - Fraction of all genotypes called.
         - **contigs** (*list of str*) - List of all unique contigs found in the dataset.
         - **multiallelics** (*int*) - Number of multiallelic variants.
         - **snps** (*int*) - Number of SNP alternate alleles.
         - **mnps** (*int*) - Number of MNP alternate alleles.
         - **insertions** (*int*) - Number of insertion alternate alleles.
         - **deletions** (*int*) - Number of deletions alternate alleles.
         - **complex** (*int*) - Number of complex alternate alleles.
         - **star** (*int*) - Number of star (upstream deletion) alternate alleles.
         - **max_alleles** (*int*) - The highest number of alleles at any variant.
         
        :return: Object containing summary information.
        :rtype: :class:`~hail.utils.Summary`
        """

        js = self._jvdf.summarize()
        return Summary._from_java(js)

    @handle_py4j
    @typecheck_method(ann_path=strlike,
                      attributes=dictof(strlike, strlike))
    def set_va_attributes(self, ann_path, attributes):
        """Sets attributes for a variant annotation.
        Attributes are key/value pairs that can be attached to a variant annotation field.

        The following attributes are read from the VCF header when importing a VCF and written
        to the VCF header when exporting a VCF:

        - INFO fields attributes (attached to (`va.info.*`)):

          - 'Number': The arity of the field. Can take values

            - `0` (Boolean flag),
            - `1` (single value),
            - `R` (one value per allele, including the reference),
            - `A` (one value per non-reference allele),
            - `G` (one value per genotype), and
            - `.` (any number of values)

              - When importing: The value in read from the VCF INFO field definition
              - When exporting: The default value is `0` for **Boolean**, `.` for **Arrays** and 1 for all other types

            - 'Description' (default is '')

        - FILTER entries in the VCF header are generated based on the attributes
          of `va.filters`.  Each key/value pair in the attributes will generate
          a FILTER entry in the VCF with ID = key and Description = value.

        **Examples**

        Consider the following command which adds a filter and an annotation to the VDS (we're assuming a split VDS for simplicity):

        1) an INFO field `AC_HC`, which stores the allele count of high
           confidence genotypes (DP >= 10, GQ >= 20) for each non-reference allele,

        2) a filter `HardFilter` that filters all sites with the `GATK suggested hard filters <http://gatkforums.broadinstitute.org/gatk/discussion/2806/howto-apply-hard-filters-to-a-call-set>`__:

           - For SNVs: QD < 2.0 || FS < 60 || MQ < 40 || MQRankSum < -12.5 || ReadPosRankSum < -8.0

           - For Indels (and other complex): QD < 2.0 || FS < 200.0 || ReadPosRankSum < 20.0

        >>> annotated_vds = vds.annotate_variants_expr([
        ... 'va.info.AC_HC = gs.filter(g => g.dp >= 10 && g.gq >= 20).callStats(g => v).AC[1:]',
        ... 'va.filters = if((v.altAllele.isSNP && (va.info.QD < 2.0 || va.info.FS < 60 || va.info.MQ < 40 || ' +
        ... 'va.info.MQRankSum < -12.5 || va.info.ReadPosRankSum < -8.0)) || ' +
        ... '(va.info.QD < 2.0 || va.info.FS < 200.0 || va.info.ReadPosRankSum < 20.0)) va.filters.add("HardFilter") else va.filters'])

        If we now export this VDS as VCF, it would produce the following header (for these new fields):

        .. code-block:: text

            ##INFO=<ID=AC_HC,Number=.,Type=String,Description=""

        This header doesn't contain all information that should be present in an optimal VCF header:
        1) There is no FILTER entry for `HardFilter`
        2) Since `AC_HC` has one entry per non-reference allele, its `Number` should be `A`
        3) `AC_HC` should have a Description

        We can fix this by setting the attributes of these fields:

        >>> annotated_vds = (annotated_vds
        ...     .set_va_attributes(
        ...         'va.info.AC_HC',
        ...         {'Description': 'Allele count for high quality genotypes (DP >= 10, GQ >= 20)',
        ...          'Number': 'A'})
        ...     .set_va_attributes(
        ...         'va.filters',
        ...         {'HardFilter': 'This site fails GATK suggested hard filters.'}))

        Exporting the VDS with the attributes now prints the following header lines:

        .. code-block:: text

            ##INFO=<ID=test,Number=A,Type=String,Description="Allele count for high quality genotypes (DP >= 10, GQ >= 20)"
            ##FILTER=<ID=HardFilter,Description="This site fails GATK suggested hard filters.">

        :param str ann_path: Path to variant annotation beginning with `va`.

        :param dict attributes: A str-str dict containing the attributes to set

        :return: Annotated dataset with the attribute added to the variant annotation.
        :rtype: :class:`.VariantDataset`

        """

        return VariantDataset(self.hc, self._jvds.setVaAttributes(ann_path, Env.jutils().javaMapToMap(attributes)))

    @handle_py4j
    @typecheck_method(ann_path=strlike,
                      attribute=strlike)
    def delete_va_attribute(self, ann_path, attribute):
        """Removes an attribute from a variant annotation field.
        Attributes are key/value pairs that can be attached to a variant annotation field.

        The following attributes are read from the VCF header when importing a VCF and written
        to the VCF header when exporting a VCF:

        - INFO fields attributes (attached to (`va.info.*`)):

          - 'Number': The arity of the field. Can take values

            - `0` (Boolean flag),
            - `1` (single value),
            - `R` (one value per allele, including the reference),
            - `A` (one value per non-reference allele),
            - `G` (one value per genotype), and
            - `.` (any number of values)

              - When importing: The value in read from the VCF INFO field definition
              - When exporting: The default value is `0` for **Boolean**, `.` for **Arrays** and 1 for all other types

            - 'Description' (default is '')

        - FILTER entries in the VCF header are generated based on the attributes
          of `va.filters`. Each key/value pair in the attributes will generate a
          FILTER entry in the VCF with ID = key and Description = value.

        :param str ann_path: Variant annotation path starting with 'va', period-delimited.

        :param str attribute: The attribute to remove (key).

        :return: Annotated dataset with the updated variant annotation without the attribute.
        :rtype: :class:`.VariantDataset`

        """

        return VariantDataset(self.hc, self._jvds.deleteVaAttribute(ann_path, attribute))

    @handle_py4j
    @requireTGenotype
    @typecheck_method(propagate_gq=bool,
                      keep_star_alleles=bool,
                      max_shift=integral)
    def split_multi(self, propagate_gq=False, keep_star_alleles=False, max_shift=100):
        """Split multiallelic variants.

        .. include:: requireTGenotype.rst

        **Examples**

        >>> vds.split_multi().write('output/split.vds')

        **Notes**

        We will explain by example. Consider a hypothetical 3-allelic
        variant:

        .. code-block:: text

          A   C,T 0/2:7,2,6:15:45:99,50,99,0,45,99

        split_multi will create two biallelic variants (one for each
        alternate allele) at the same position

        .. code-block:: text

          A   C   0/0:13,2:15:45:0,45,99
          A   T   0/1:9,6:15:50:50,0,99

        Each multiallelic GT field is downcoded once for each
        alternate allele. A call for an alternate allele maps to 1 in
        the biallelic variant corresponding to itself and 0
        otherwise. For example, in the example above, 0/2 maps to 0/0
        and 0/1. The genotype 1/2 maps to 0/1 and 0/1.

        The biallelic alt AD entry is just the multiallelic AD entry
        corresponding to the alternate allele. The ref AD entry is the
        sum of the other multiallelic entries.

        The biallelic DP is the same as the multiallelic DP.

        The biallelic PL entry for for a genotype g is the minimum
        over PL entries for multiallelic genotypes that downcode to
        g. For example, the PL for (A, T) at 0/1 is the minimum of the
        PLs for 0/1 (50) and 1/2 (45), and thus 45.

        Fixing an alternate allele and biallelic variant, downcoding
        gives a map from multiallelic to biallelic alleles and
        genotypes. The biallelic AD entry for an allele is just the
        sum of the multiallelic AD entries for alleles that map to
        that allele. Similarly, the biallelic PL entry for a genotype
        is the minimum over multiallelic PL entries for genotypes that
        map to that genotype.

        By default, GQ is recomputed from PL. If ``propagate_gq=True``
        is passed, the biallelic GQ field is simply the multiallelic
        GQ field, that is, genotype qualities are unchanged.

        Here is a second example for a het non-ref

        .. code-block:: text

          A   C,T 1/2:2,8,6:16:45:99,50,99,45,0,99

        splits as

        .. code-block:: text

          A   C   0/1:8,8:16:45:45,0,99
          A   T   0/1:10,6:16:50:50,0,99

        **VCF Info Fields**

        Hail does not split annotations in the info field. This means
        that if a multiallelic site with ``info.AC`` value ``[10, 2]`` is
        split, each split site will contain the same array ``[10,
        2]``. The provided allele index annotation ``va.aIndex`` can be used
        to select the value corresponding to the split allele's
        position:

        >>> vds_result = (vds.split_multi()
        ...     .filter_variants_expr('va.info.AC[va.aIndex - 1] < 10', keep = False))

        VCFs split by Hail and exported to new VCFs may be
        incompatible with other tools, if action is not taken
        first. Since the "Number" of the arrays in split multiallelic
        sites no longer matches the structure on import ("A" for 1 per
        allele, for example), Hail will export these fields with
        number ".".

        If the desired output is one value per site, then it is
        possible to use annotate_variants_expr to remap these
        values. Here is an example:

        >>> (vds.split_multi()
        ...     .annotate_variants_expr('va.info.AC = va.info.AC[va.aIndex - 1]')
        ...     .export_vcf('output/export.vcf'))

        The info field AC in *data/export.vcf* will have ``Number=1``.

        **Annotations**

        :py:meth:`~hail.VariantDataset.split_multi` adds the
        following annotations:

         - **va.wasSplit** (*Boolean*) -- true if this variant was
           originally multiallelic, otherwise false.
         - **va.aIndex** (*Int*) -- The original index of this
           alternate allele in the multiallelic representation (NB: 1
           is the first alternate allele or the only alternate allele
           in a biallelic variant). For example, 1:100:A:T,C splits
           into two variants: 1:100:A:T with ``aIndex = 1`` and
           1:100:A:C with ``aIndex = 2``.

        :param bool propagate_gq: Set the GQ of output (split)
          genotypes to be the GQ of the input (multi-allelic) variants
          instead of recompute GQ as the difference between the two
          smallest PL values.  Intended to be used in conjunction with
          ``import_vcf(store_gq=True)``.  This option will be obviated
          in the future by generic genotype schemas.  Experimental.
        :param bool keep_star_alleles: Do not filter out * alleles.
        :param int max_shift: maximum number of base pairs by which
          a split variant can move.  Affects memory usage, and will
          cause Hail to throw an error if a variant that moves further
          is encountered.

        :return: A biallelic variant dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        jvds = self._jvdf.splitMulti(propagate_gq, keep_star_alleles, max_shift)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @requireTGenotype
    @typecheck_method(pedigree=Pedigree,
                      root=strlike)
    def tdt(self, pedigree, root='va.tdt'):
        """Find transmitted and untransmitted variants; count per variant and
        nuclear family.

        .. include:: requireTGenotype.rst

        **Examples**

        Compute TDT association results:

        >>> pedigree = Pedigree.read('data/trios.fam')
        >>> (vds.tdt(pedigree)
        ...     .export_variants("output/tdt_results.tsv", "Variant = v, va.tdt.*"))

        **Notes**

        The transmission disequilibrium test tracks the number of times the alternate allele is transmitted (t) or not transmitted (u) from a heterozgyous parent to an affected child under the null that the rate of such transmissions is 0.5.  For variants where transmission is guaranteed (i.e., the Y chromosome, mitochondria, and paternal chromosome X variants outside of the PAR), the test cannot be used.

        The TDT statistic is given by

        .. math::

            (t-u)^2 \over (t+u)

        and follows a 1 degree of freedom chi-squared distribution under the null hypothesis.


        The number of transmissions and untransmissions for each possible set of genotypes is determined from the table below.  The copy state of a locus with respect to a trio is defined as follows, where PAR is the pseudoautosomal region (PAR).

        - HemiX -- in non-PAR of X and child is male
        - Auto -- otherwise (in autosome or PAR, or child is female)

        +--------+--------+--------+------------+---+---+
        |  Kid   | Dad    | Mom    | Copy State | T | U |
        +========+========+========+============+===+===+
        | HomRef | Het    | Het    | Auto       | 0 | 2 |
        +--------+--------+--------+------------+---+---+
        | HomRef | HomRef | Het    | Auto       | 0 | 1 |
        +--------+--------+--------+------------+---+---+
        | HomRef | Het    | HomRef | Auto       | 0 | 1 |
        +--------+--------+--------+------------+---+---+
        | Het    | Het    | Het    | Auto       | 1 | 1 |
        +--------+--------+--------+------------+---+---+
        | Het    | HomRef | Het    | Auto       | 1 | 0 |
        +--------+--------+--------+------------+---+---+
        | Het    | Het    | HomRef | Auto       | 1 | 0 |
        +--------+--------+--------+------------+---+---+
        | Het    | HomVar | Het    | Auto       | 0 | 1 |
        +--------+--------+--------+------------+---+---+
        | Het    | Het    | HomVar | Auto       | 0 | 1 |
        +--------+--------+--------+------------+---+---+
        | HomVar | Het    | Het    | Auto       | 2 | 0 |
        +--------+--------+--------+------------+---+---+
        | HomVar | Het    | HomVar | Auto       | 1 | 0 |
        +--------+--------+--------+------------+---+---+
        | HomVar | HomVar | Het    | Auto       | 1 | 0 |
        +--------+--------+--------+------------+---+---+
        | HomRef | HomRef | Het    | HemiX      | 0 | 1 |
        +--------+--------+--------+------------+---+---+
        | HomRef | HomVar | Het    | HemiX      | 0 | 1 |
        +--------+--------+--------+------------+---+---+
        | HomVar | HomRef | Het    | HemiX      | 1 | 0 |
        +--------+--------+--------+------------+---+---+
        | HomVar | HomVar | Het    | HemiX      | 1 | 0 |
        +--------+--------+--------+------------+---+---+


        :py:meth:`~hail.VariantDataset.tdt` only considers complete trios (two parents and a proband) with defined sex.

        PAR is currently defined with respect to reference `GRCh37 <http://www.ncbi.nlm.nih.gov/projects/genome/assembly/grc/human/>`__:

        - X: 60001-2699520
        - X: 154931044-155260560
        - Y: 10001-2649520
        - Y: 59034050-59363566

        :py:meth:`~hail.VariantDataset.tdt` assumes all contigs apart from X and Y are fully autosomal; decoys, etc. are not given special treatment.

        **Annotations**

        :py:meth:`~hail.VariantDataset.tdt` adds the following annotations:

         - **tdt.nTransmitted** (*Int*) -- Number of transmitted alternate alleles.

         - **va.tdt.nUntransmitted** (*Int*) -- Number of untransmitted alternate alleles.

         - **va.tdt.chi2** (*Double*) -- TDT statistic.

         - **va.tdt.pval** (*Double*) -- p-value.

        :param pedigree: Sample pedigree.
        :type pedigree: :class:`~hail.representation.Pedigree`

        :param root: Variant annotation root to store TDT result.

        :return: Variant dataset with TDT association results added to variant annotations.
        :rtype: :py:class:`.VariantDataset`
        """

        jvds = self._jvdf.tdt(pedigree._jrep, root)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    def _typecheck(self):
        """Check if all sample, variant and global annotations are consistent with the schema."""

        self._jvds.typecheck()

    @handle_py4j
    @requireTGenotype
    @typecheck_method(root=strlike)
    def variant_qc(self, root='va.qc'):
        """Compute common variant statistics (quality control metrics).

        .. include:: requireTGenotype.rst

        **Examples**

        >>> vds_result = vds.variant_qc()

        .. _variantqc_annotations:

        **Annotations**

        :py:meth:`~hail.VariantDataset.variant_qc` computes 18 variant statistics from the 
        genotype data and stores the results as variant annotations that can be accessed 
        with ``va.qc.<identifier>`` (or ``<root>.<identifier>`` if a non-default root was passed):

        +---------------------------+--------+--------------------------------------------------------+
        | Name                      | Type   | Description                                            |
        +===========================+========+========================================================+
        | ``callRate``              | Double | Fraction of samples with called genotypes              |
        +---------------------------+--------+--------------------------------------------------------+
        | ``AF``                    | Double | Calculated alternate allele frequency (q)              |
        +---------------------------+--------+--------------------------------------------------------+
        | ``AC``                    | Int    | Count of alternate alleles                             |
        +---------------------------+--------+--------------------------------------------------------+
        | ``rHeterozygosity``       | Double | Proportion of heterozygotes                            |
        +---------------------------+--------+--------------------------------------------------------+
        | ``rHetHomVar``            | Double | Ratio of heterozygotes to homozygous alternates        |
        +---------------------------+--------+--------------------------------------------------------+
        | ``rExpectedHetFrequency`` | Double | Expected rHeterozygosity based on HWE                  |
        +---------------------------+--------+--------------------------------------------------------+
        | ``pHWE``                  | Double | p-value from Hardy Weinberg Equilibrium null model     |
        +---------------------------+--------+--------------------------------------------------------+
        | ``nHomRef``               | Int    | Number of homozygous reference samples                 |
        +---------------------------+--------+--------------------------------------------------------+
        | ``nHet``                  | Int    | Number of heterozygous samples                         |
        +---------------------------+--------+--------------------------------------------------------+
        | ``nHomVar``               | Int    | Number of homozygous alternate samples                 |
        +---------------------------+--------+--------------------------------------------------------+
        | ``nCalled``               | Int    | Sum of ``nHomRef``, ``nHet``, and ``nHomVar``          |
        +---------------------------+--------+--------------------------------------------------------+
        | ``nNotCalled``            | Int    | Number of uncalled samples                             |
        +---------------------------+--------+--------------------------------------------------------+
        | ``nNonRef``               | Int    | Sum of ``nHet`` and ``nHomVar``                        |
        +---------------------------+--------+--------------------------------------------------------+
        | ``rHetHomVar``            | Double | Het/HomVar ratio across all samples                    |
        +---------------------------+--------+--------------------------------------------------------+
        | ``dpMean``                | Double | Depth mean across all samples                          |
        +---------------------------+--------+--------------------------------------------------------+
        | ``dpStDev``               | Double | Depth standard deviation across all samples            |
        +---------------------------+--------+--------------------------------------------------------+
        | ``gqMean``                | Double | The average genotype quality across all samples        |
        +---------------------------+--------+--------------------------------------------------------+
        | ``gqStDev``               | Double | Genotype quality standard deviation across all samples |
        +---------------------------+--------+--------------------------------------------------------+

        Missing values ``NA`` may result (for example, due to division by zero) and are handled properly 
        in filtering and written as "NA" in export modules. The empirical standard deviation is computed
        with zero degrees of freedom.

        :param str root: Variant annotation root for computed struct.

        :return: Annotated variant dataset with new variant QC annotations.
        :rtype: :py:class:`.VariantDataset`
        """

        jvds = self._jvdf.variantQC(root)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @typecheck_method(config=strlike,
                      block_size=integral,
                      root=strlike,
                      csq=bool)
    def vep(self, config, block_size=1000, root='va.vep', csq=False):
        """Annotate variants with VEP.

        :py:meth:`~hail.VariantDataset.vep` runs `Variant Effect Predictor <http://www.ensembl.org/info/docs/tools/vep/index.html>`__ with
        the `LOFTEE plugin <https://github.com/konradjk/loftee>`__
        on the current variant dataset and adds the result as a variant annotation.

        **Examples**

        Add VEP annotations to the dataset:

        >>> vds_result = vds.vep("data/vep.properties") # doctest: +SKIP

        **Configuration**

        :py:meth:`~hail.VariantDataset.vep` needs a configuration file to tell it how to run
        VEP. The format is a `.properties file <https://en.wikipedia.org/wiki/.properties>`__.
        Roughly, each line defines a property as a key-value pair of the form `key = value`. `vep` supports the following properties:

        - **hail.vep.perl** -- Location of Perl. Optional, default: perl.
        - **hail.vep.perl5lib** -- Value for the PERL5LIB environment variable when invoking VEP. Optional, by default PERL5LIB is not set.
        - **hail.vep.path** -- Value of the PATH environment variable when invoking VEP.  Optional, by default PATH is not set.
        - **hail.vep.location** -- Location of the VEP Perl script.  Required.
        - **hail.vep.cache_dir** -- Location of the VEP cache dir, passed to VEP with the `--dir` option.  Required.
        - **hail.vep.fasta** -- Location of the FASTA file to use to look up the reference sequence, passed to VEP with the `--fasta` option.  Required.
        - **hail.vep.assembly** -- Genome assembly version to use. Optional, default: GRCh37
        - **hail.vep.plugin** -- VEP plugin, passed to VEP with the `--plugin` option.  Optional. Overrides `hail.vep.lof.human_ancestor` and `hail.vep.lof.conservation_file`.
        - **hail.vep.lof.human_ancestor** -- Location of the human ancestor file for the LOFTEE plugin.  Ignored if `hail.vep.plugin` is set. Required otherwise.
        - **hail.vep.lof.conservation_file** -- Location of the conservation file for the LOFTEE plugin.  Ignored if `hail.vep.plugin` is set. Required otherwise.


        Here is an example `vep.properties` configuration file

        .. code-block:: text

            hail.vep.perl = /usr/bin/perl
            hail.vep.path = /usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
            hail.vep.location = /path/to/vep/ensembl-tools-release-81/scripts/variant_effect_predictor/variant_effect_predictor.pl
            hail.vep.cache_dir = /path/to/vep
            hail.vep.lof.human_ancestor = /path/to/loftee_data/human_ancestor.fa.gz
            hail.vep.lof.conservation_file = /path/to/loftee_data//phylocsf.sql

        **VEP Invocation**

        .. code-block:: text

            <hail.vep.perl>
            <hail.vep.location>
            --format vcf
            --json
            --everything
            --allele_number
            --no_stats
            --cache --offline
            --dir <hail.vep.cache_dir>
            --fasta <hail.vep.fasta>
            --minimal
            --assembly <hail.vep.assembly>
            --plugin LoF,human_ancestor_fa:$<hail.vep.lof.human_ancestor>,filter_position:0.05,min_intron_size:15,conservation_file:<hail.vep.lof.conservation_file>
            -o STDOUT

        **Annotations**

        Annotations with the following schema are placed in the location specified by ``root``.
        The full resulting dataset schema can be queried with :py:attr:`~hail.VariantDataset.variant_schema`.

        .. code-block:: text

            Struct{
              assembly_name: String,
              allele_string: String,
              colocated_variants: Array[Struct{
                aa_allele: String,
                aa_maf: Double,
                afr_allele: String,
                afr_maf: Double,
                allele_string: String,
                amr_allele: String,
                amr_maf: Double,
                clin_sig: Array[String],
                end: Int,
                eas_allele: String,
                eas_maf: Double,
                ea_allele: String,,
                ea_maf: Double,
                eur_allele: String,
                eur_maf: Double,
                exac_adj_allele: String,
                exac_adj_maf: Double,
                exac_allele: String,
                exac_afr_allele: String,
                exac_afr_maf: Double,
                exac_amr_allele: String,
                exac_amr_maf: Double,
                exac_eas_allele: String,
                exac_eas_maf: Double,
                exac_fin_allele: String,
                exac_fin_maf: Double,
                exac_maf: Double,
                exac_nfe_allele: String,
                exac_nfe_maf: Double,
                exac_oth_allele: String,
                exac_oth_maf: Double,
                exac_sas_allele: String,
                exac_sas_maf: Double,
                id: String,
                minor_allele: String,
                minor_allele_freq: Double,
                phenotype_or_disease: Int,
                pubmed: Array[Int],
                sas_allele: String,
                sas_maf: Double,
                somatic: Int,
                start: Int,
                strand: Int
              }],
              end: Int,
              id: String,
              input: String,
              intergenic_consequences: Array[Struct{
                allele_num: Int,
                consequence_terms: Array[String],
                impact: String,
                minimised: Int,
                variant_allele: String
              }],
              most_severe_consequence: String,
              motif_feature_consequences: Array[Struct{
                allele_num: Int,
                consequence_terms: Array[String],
                high_inf_pos: String,
                impact: String,
                minimised: Int,
                motif_feature_id: String,
                motif_name: String,
                motif_pos: Int,
                motif_score_change: Double,
                strand: Int,
                variant_allele: String
              }],
              regulatory_feature_consequences: Array[Struct{
                allele_num: Int,
                biotype: String,
                consequence_terms: Array[String],
                impact: String,
                minimised: Int,
                regulatory_feature_id: String,
                variant_allele: String
              }],
              seq_region_name: String,
              start: Int,
              strand: Int,
              transcript_consequences: Array[Struct{
                allele_num: Int,
                amino_acids: String,
                biotype: String,
                canonical: Int,
                ccds: String,
                cdna_start: Int,
                cdna_end: Int,
                cds_end: Int,
                cds_start: Int,
                codons: String,
                consequence_terms: Array[String],
                distance: Int,
                domains: Array[Struct{
                  db: String
                  name: String
                }],
                exon: String,
                gene_id: String,
                gene_pheno: Int,
                gene_symbol: String,
                gene_symbol_source: String,
                hgnc_id: String,
                hgvsc: String,
                hgvsp: String,
                hgvs_offset: Int,
                impact: String,
                intron: String,
                lof: String,
                lof_flags: String,
                lof_filter: String,
                lof_info: String,
                minimised: Int,
                polyphen_prediction: String,
                polyphen_score: Double,
                protein_end: Int,
                protein_start: Int,
                protein_id: String,
                sift_prediction: String,
                sift_score: Double,
                strand: Int,
                swissprot: String,
                transcript_id: String,
                trembl: String,
                uniparc: String,
                variant_allele: String
              }],
              variant_class: String
            }

        :param str config: Path to VEP configuration file.

        :param block_size: Number of variants to annotate per VEP invocation.
        :type block_size: int

        :param str root: Variant annotation path to store VEP output.

        :param bool csq: If ``True``, annotates VCF CSQ field as a String.
            If ``False``, annotates with the full nested struct schema

        :return: An annotated with variant annotations from VEP.
        :rtype: :py:class:`.VariantDataset`
        """

        jvds = self._jvds.vep(config, root, csq, block_size)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    def nirvana(self, config, block_size = 50000, root = 'va.nirvana'):
        """Annotate variants with `Nirvana <https://github.com/Illumina/Nirvana>`_.

        ***Configuration***

        :py:meth:`~hail.VariantDataset.nirvana` needs a configuration file to tell it how to run
        Nirvana. The format is a `.properties file <https://en.wikipedia.org/wiki/.properties>`_.
        Roughly, each line defines a property as a key-value pair of the form `key = value`. `nirvana` supports the following properties:

        - **hail.nirvana.dotnet** -- Location of dotnet. Optional, default: dotnet.
        - **hail.nirvana.path** -- Value of the PATH environment variable when invoking Nirvana.  Optional, by default PATH is not set.
        - **hail.nirvana.location** -- Location of Nirvana.dll. Required.
        - **hail.nirvana.cache** --Location of cache. Required.
        - **hail.nirvana.supplementaryAnnotationDirectory** -- Location of Supplementary Database. Optional, no supplementary database by default.
        
        **Annotations**

        Annotations with the following schema are placed in the location specified by ``root``.

        .. code-block:: text
        
            Struct{
                chromosome: String,
                refAllele: String,
                position: Int,
                altAlleles: Array[String],
                cytogeneticBand: String,
                filters: Array[String],
                variants: Array[Struct{
                  altAllele: String,
                  refAllele: String,
                  chromosome: String,
                  begin: Int,
                  end: Int,
                  phylopScore: Double,
                  isReferenceMinor: Boolean,
                  variantType: String,
                  vid: String,
                  isRecomposed: Boolean,
                  regulatoryRegions: Array[Struct{
                    id: String,
                    consequence: Set(String),
                    type: String
                  }],
                  clinvar: Array[Struct{
                    id: String,
                    reviewStatus: String,
                    isAlleleSpecific: Boolean,
                    alleleOrigins: Array[String],
                    refAllele: String,
                    altAllele: String,
                    phenotypes: Array[String],
                    medGenIds: Array[String],
                    omimIds: Array[String],
                    orphanetIds: Array[String],
                    geneReviewsId: String,
                    significance: String,
                    lastUpdatedDate: String,
                    pubMedIds: Array[String]
                  }],
                  cosmic: Array[Struct{
                    id: String,
                    isAlleleSpecific: Boolean,
                    refAllele: String,
                    altAllele: String,
                    gene: String,
                    sampleCount: Int,
                    studies: Array[Struct{
                      id: Int,
                      histology: String,
                      primarySite: String
                    }]
                  }],
                  dbsnp: Struct{"ids: Array[String]},
                  evs: Struct{
                    coverage: Int,
                    sampleCount: Int,
                    allAf: Double,
                    afrAf: Double,
                    eurAf: Double
                  },
                  exac: Struct{
                    coverage: Int,
                    allAf: Double,
                    allAc: Int,
                    allAn: Int,
                    afrAf: Double,
                    afrAc: Int,
                    afrAn: Int,
                    amrAf: Double,
                    amrAc: Int,
                    amrAn: Int,
                    easAf: Double,
                    easAc: Int,
                    easAn: Int,
                    finAf: Double,
                    finAc: Int,
                    finAn: Int,
                    nfeAf: Double,
                    nfeAc: Int,
                    nfeAn: Int,
                    othAf: Double,
                    othAc: Int,
                    othAn: Int,
                    sasAf: Double,
                    sasAc: Int,
                    sasAn: Int
                  },
                  globalAllele: Struct{
                    globalMinorAllele: String,
                    globalMinorAlleleFrequency: Double
                  },
                  oneKg: Struct{
                    ancestralAllele: String,
                    allAf: Double,
                    allAc: Int,
                    allAn: Int,
                    afrAf: Double,
                    afrAc: Int,
                    afrAn: Int,
                    amrAf: Double,
                    amrAc: Int,
                    amrAn: Int,
                    easAf: Double,
                    easAc: Int,
                    easAn: Int,
                    eurAf: Double,
                    eurAc: Int,
                    eurAn: Int,
                    sasAf: Double,
                    sasAc: Int,
                    sasAn: Int
                  },
                  transcripts: Struct{
                    refSeq: Array[Struct{
                      transcript: String,
                      bioType: String,
                      aminoAcids: String,
                      cDnaPos: String,
                      codons: String,
                      cdsPos: String,
                      exons: String,
                      introns: String,
                      geneId: String,
                      hgnc: String,
                      consequence: Array[String],
                      hgvsc: String,
                      hgvsp: String,
                      isCanonical: Boolean,
                      polyPhenScore: Double,
                      polyPhenPrediction: String,
                      proteinId: String,
                      proteinPos: String,
                      siftScore: Double,
                      siftPrediction: String
                    }],
                    ensembl: Array[Struct{
                      transcript: String,
                      bioType: String,
                      aminoAcids: String,
                      cDnaPos: String,
                      codons: String,
                      cdsPos: String,
                      exons: String,
                      introns: String,
                      geneId: String,
                      hgnc: String,
                      consequence: Array[String],
                      hgvsc: String,
                      hgvsp: String,
                      isCanonical: Boolean,
                      polyPhenScore: Double,
                      polyPhenPrediction: String,
                      proteinId: String,
                      proteinPos: String,
                      siftScore: Double,
                      siftPrediction: String
                    }]
                  },
                  genes: Array[Struct{
                    name: String,
                    omim: Array[Struct(
                      mimNumber: Int,
                      hgnc: String,
                      description: String,
                      phenotypes: Array[Struct{
                        mimNumber: Int,
                        phenotype: String,
                        mapping: String,
                        inheritance: Array[String],
                        comments: String
                      }]
                    )]
                  }]
                }]
            }

        :param str config: The path to the config file.

        :param int block_size: The number of variants processed in one Nirvana job within a partition. If block_size is greater than or equal to the number of variants in a partition, that whole partition will be processed in one job.

        :param str root: The root of the annotation path for variant annotations.

        :return: An annotated dataset with variant annotations from Nirvana.
        :rtype: :py:class:`.VariantDataset`

        """
        jvds = self._jvdf.nirvana(config, block_size, root)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    def variants_table(self):
        """Convert variants and variant annotations to a KeyTable.

        The resulting KeyTable has schema:

        .. code-block:: text

          Struct {
            v: Variant
            va: variant annotations
          }

        with a single key ``v``.

        :return: Key table with variants and variant annotations.
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(self.hc, self._jvds.variantsKT())

    @handle_py4j
    def samples_table(self):
        """Convert samples and sample annotations to KeyTable.

        The resulting KeyTable has schema:

        .. code-block:: text

          Struct {
            s: Sample
            sa: sample annotations
          }

        with a single key ``s``.

        :return: Key table with samples and sample annotations.
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(self.hc, self._jvds.samplesKT())

    @handle_py4j
    def genotypes_table(self):
        """Generate a fully expanded genotype table.

        **Examples**

        >>> gs = vds.genotypes_table()

        **Notes**

        This produces a (massive) flat table from all the
        genotypes in the dataset. The table has columns:

            - **v** (*Variant*) - Variant (key column).
            - **va** (*Variant annotation schema*) - Variant annotations.
            - **s** (*String*) - Sample ID (key column).
            - **sa** (*Sample annotation schema*) - Sample annotations.
            - **g** (*Genotype schema*) - Genotype or generic genotype.

        .. caution::

            This table has a row for each variant/sample pair. The genotype
            key table for a dataset with 10M variants and 10K samples will
            contain 100 billion rows. Writing or exporting this table will
            produce a file **much** larger than the equivalent VDS.

        :return: Key table with a row for each genotype.
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(self.hc, self._jvds.genotypeKT())

    @handle_py4j
    @typecheck_method(variant_expr=oneof(strlike, listof(strlike)),
                      genotype_expr=oneof(strlike, listof(strlike)),
                      key=oneof(strlike, listof(strlike)),
                      separator=strlike)
    def make_table(self, variant_expr, genotype_expr, key=[], separator='.'):
        """Produce a key with one row per variant and one or more columns per sample.

        **Examples**

        Consider a :py:class:`VariantDataset` ``vds`` with 2 variants and 3 samples:

        .. code-block:: text

          Variant	FORMAT	A	B	C
          1:1:A:T	GT:GQ	0/1:99	./.	0/0:99
          1:2:G:C	GT:GQ	0/1:89	0/1:99	1/1:93

        Then

        >>> kt = vds.make_table('v = v', ['gt = g.gt', 'gq = g.gq'])

        returns a :py:class:`KeyTable` with schema

        .. code-block:: text

            v: Variant
            A.gt: Int
            A.gq: Int
            B.gt: Int
            B.gq: Int
            C.gt: Int
            C.gq: Int

        and values

        .. code-block:: text

            v	A.gt	A.gq	B.gt	B.gq	C.gt	C.gq
            1:1:A:T	1	99	NA	NA	0	99
            1:2:G:C	1	89	1	99	2	93

        The above table can be generated and exported as a TSV using :class:`.KeyTable` :py:meth:`~hail.KeyTable.export`.

        **Notes**

        Per sample field names in the result are formed by
        concatenating the sample ID with the ``genotype_expr`` left
        hand side with ``separator``.  If the left hand side is empty::

          `` = expr

        then the dot (.) is omitted.

        :param variant_expr: Variant annotation expressions.
        :type variant_expr: str or list of str

        :param genotype_expr: Genotype annotation expressions.
        :type genotype_expr: str or list of str

        :param key: List of key columns.
        :type key: str or list of str

        :param str separator: Separator to use between sample IDs and genotype expression left-hand side identifiers.

        :rtype: :py:class:`.KeyTable`

        """

        if isinstance(variant_expr, list):
            variant_expr = ','.join(variant_expr)
        if isinstance(genotype_expr, list):
            genotype_expr = ','.join(genotype_expr)

        jkt = self._jvds.makeKT(variant_expr, genotype_expr,
                                jarray(Env.jvm().java.lang.String, wrap_to_list(key)), separator)
        return KeyTable(self.hc, jkt)

vds_type.set(VariantDataset)
