from __future__ import print_function  # Python 2 and 3 print compatibility

from hail.java import scala_package_object, jarray, raise_py4j_exception, env
from hail.keytable import KeyTable
from hail.type import Type
from hail.utils import TextTableConfig
from py4j.protocol import Py4JJavaError

import warnings

warnings.filterwarnings(module=__name__, action='once')


class VariantDataset(object):
    """Hail's primary representation of genomic data, a matrix keyed by sample and variant.

    :ivar hc: Hail Context
    :vartype hc: :class:`.HailContext`
    :ivar sample_ids: Sample IDs
    :vartype sample_ids: list of str
    :ivar sample_annotations: Sample annotations
    :vartype sample_annotations: dict
    :ivar globals: Global annotations
    :ivar variant_schema: type of variant annotations
    :vartype variant_schema: :class:`.Type`
    :ivar sample_schema: type of sample annotations
    :vartype sample_schema: :class:`.Type`
    :ivar global_schema: type of global annotations
    :vartype global_schema: :class:`.Type`
    :ivar int num_samples: number of samples in dataset
    """

    def __init__(self, hc, jvds):
        self.hc = hc
        self._jvds = jvds

        self._globals = None
        self._sample_annotations = None
        self._sa_schema = None
        self._va_schema = None
        self._global_schema = None
        self._sample_ids = None
        self._num_samples = None

    @property
    def sample_ids(self):
        """Return sampleIDs.

        :return: List of sample IDs.

        :rtype: list of str

        """
        if self._sample_ids is None:
            self._sample_ids = list(self._jvds.sampleIdsAsArray())
        return self._sample_ids

    @property
    def sample_annotations(self):
        """Return a dict of sample annotations.

        The keys of this dictionary are the sample IDs (strings).
        The values are sample annotations.

        :return: dict
        """

        if self._sample_annotations is None:
            zipped_annotations = env.jutils.iterableToArrayList(
                self._jvds.sampleIdsAndAnnotations()
            )
            r = {}
            for element in zipped_annotations:
                r[element._1()] = self.sample_schema._convert_to_py(element._2())
            self._sample_annotations = r
        return self._sample_annotations

    def num_partitions(self):
        """Number of RDD partitions.

        :rtype: int

        """
        try:
            return self._jvds.nPartitions()
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    @property
    def num_samples(self):
        """Number of samples.

        :rtype: int

        """
        if self._num_samples is None:
            self._num_samples = self._jvds.nSamples()
        return self._num_samples

    def count_variants(self):
        """Count the number of variants in the dataset.

        :rtype: long

        """
        try:
            return self._jvds.nVariants()
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def was_split(self):
        """Multiallelic variants have been split into multiple biallelic variants.

        Result is True if :py:meth:`~hail.VariantDataset.split_multi` has been called on this dataset
        or the dataset was imported with :py:meth:`~hail.HailContext.import_plink`, :py:meth:`~hail.HailContext.import_gen`,
        or :py:meth:`~hail.HailContext.import_bgen`.

        :rtype: bool

        """
        try:
            return self._jvds.wasSplit()
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def is_dosage(self):
        """Genotype probabilities are dosages.

        The result of ``is_dosage()`` will be True if the dataset was imported with :py:meth:`~hail.HailContext.import_gen` or
        :py:meth:`~hail.HailContext.import_bgen`.

        :rtype: bool

        """
        try:
            return self._jvds.isDosage()
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def file_version(self):
        """File version of dataset.

        :rtype: int

        """
        try:
            return self._jvds.fileVersion()
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def aggregate_by_key(self, key_code, agg_code):
        """Aggregate by user-defined key and aggregation expressions.
        Equivalent of a group-by operation in SQL.

        :param key_code: Named expression(s) for which fields are keys.
        :type key_code: str or list of str
        :param agg_code: Named aggregation expression(s).
        :type agg_code: str or list of str

        :rtype: :class:`.KeyTable`

        """
        try:
            return KeyTable(self.hc, self._jvds.aggregateByKey(key_code, agg_code))
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def aggregate_intervals(self, input, condition, output):
        '''Aggregate over intervals and export.

        **Examples**

        Calculate the total number of SNPs, indels, and variants contained in
        the intervals specified by *data/capture_intervals.txt*:

        >>> vds.aggregate_intervals('data/capture_intervals.txt',
        >>>   """n_SNP = variants.filter(v => v.altAllele.isSNP).count(),
        >>>      n_indel = variants.filter(v => v.altAllele.isIndel).count(),
        >>>      n_total = variants.count()""",
        >>>   'out.txt')

        If *data/capture_intervals.txt* contains:

        .. code-block:: text

            4:1500-123123
            5:1-1000000
            16:29500000-30200000

        then the previous expression writes something like the following to
        *out.txt*:

        .. code-block:: text

            Contig    Start       End         n_SNP   n_indel     n_total
            4         1500        123123      502     51          553
            5         1           1000000     25024   4500        29524
            16        29500000    30200000    17222   2021        19043

        The parameter ``condition`` defines the names of the column headers (in
        the previous case: ``n_SNP``, ``n_indel``, ``n_total``) and how to
        calculate the value of that column for each interval.

        Count the number of LOF, missense, and synonymous non-reference calls
        per interval:

        >>> (vds.annotate_variants_expr('va.n_calls = gs.filter(g => g.isCalledNonRef).count()')
        >>>     .aggregate_intervals('data/intervals.txt'
        >>>        """LOF_CALLS = variants.filter(v => va.consequence == "LOF").map(v => va.n_calls).sum(),
        >>>           MISSENSE_CALLS = variants.filter(v => va.consequence == "missense").map(v => va.n_calls).sum(),
        >>>           SYN_CALLS = variants.filter(v => va.consequence == "synonymous").map(v => va.n_calls).sum()""",
        >>>        'out.txt'))

        If *data/intervals.txt* contains:

        .. code-block:: text

            4:1500-123123
            5:1-1000000
            16:29500000-30200000

        then the previous expression writes something like the following to
        *out.txt*:

        .. code-block:: text

            Contig    Start       End         LOF_CALLS   MISSENSE_CALLS   SYN_CALLS
            4         1500        123123      42          122              553
            5         1           1000000     3           12               66
            16        29500000    30200000    17          22               202

        **Notes**

        Intervals are **left inclusive, right exclusive**.  This means that
        [chr1:1, chr1:3) contains chr1:1 and chr1:2.

        **Designating output with an expression**

        An export expression designates a list of computations to perform, and
        what these columns are named.  These expressions should take the form
        ``COL_NAME_1 = <expression>, COL_NAME_2 = <expression>, ...``.

        ``condition`` has the following symbols in scope:

        - ``interval`` (*Interval*): genomic interval
        - ``global``: global annotations
        - ``variants`` (*Aggregable[Variant]*): aggregable of :ref:`variant`s Aggregator namespace below.

        The ``variants`` aggregator has the following namespace:

        - ``v`` (*Variant*): :ref:`variant`
        - ``va``: variant annotations
        - ``global``: Global annotations

        :param str input: Input interval list file.
        :param str condition: Aggregation expression.
        :param str output: Output file.
        '''

        pargs = ['aggregateintervals', '-i', input, '-c', condition, '-o', output]
        self.hc._run_command(self, pargs)

    def annotate_alleles_expr(self, condition, propagate_gq=False):
        """Annotate alleles with expression.

        :param condition: Annotation expression.
        :type condition: str or list of str
        :param bool propagate_gq: Propagate GQ instead of computing from (split) PL.

        :return: Annotated dataset.
        :rtype :class:`.VariantDataset`
        """
        if isinstance(condition, list):
            condition = ','.join(condition)
        pargs = ['annotatealleles', 'expr', '-c', condition]
        if propagate_gq:
            pargs.append('--propagate-gq')
        return self.hc._run_command(self, pargs)

    def annotate_global_py(self, path, annotation, annotation_type):
        """Annotate global from python objects.

        :param str path: annotation path starting in 'global'
        :param annotation: annotation to add to global
        :param :class:`.Type` annotation_type: Hail type of annotation

        :return: Annotated dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        annotation_type._typecheck(annotation)

        annotated = self._jvds.annotateGlobal(annotation_type._convert_to_j(annotation), annotation_type._jtype, path)
        assert annotated.globalSignature().typeCheck(annotated.globalAnnotation()), 'error in java type checking'
        return VariantDataset(self.hc, annotated)

    def annotate_global_list(self, input, root, as_set=False):
        """Load text file into global annotations as Array[String] or
        Set[String].

        **Examples**

        Add a list of genes in a file to global annotations:

        >>> vds = (hc.read('data/example.vds')
        >>>  .annotate_global_list('data/genes.txt', 'global.genes'))

        For the gene list

        .. code-block: text

            $ cat data/genes.txt
            SCN2A
            SONIC-HEDGEHOG
            PRNP

        this adds ``global.genes: Array[String]`` with value ``["SCN2A", "SONIC-HEDGEHOG", "PRNP"]``.

        To filter to those variants in genes listed in *genes.txt* given a variant annotation ``va.gene: String``, annotate as type ``Set[String]`` instead:

        >>> vds = (hc.read('data/example.vds')
        >>>  .annotate_global_list('data/genes.txt', 'global.genes', as_set=True)
        >>>  .filter_variants_expr('global.genes.contains(va.gene)'))

        :param str input: Input text file.
        :param str root: Global annotation path to store text file.
        :param bool as_set: If True, load text file as Set[String],
            otherwise, load as Array[String].

        :return: An annotated dataset with a new global annotation given by the list.
        :rtype: :class:`.VariantDataset`
        """

        pargs = ['annotateglobal', 'list', '-i', input, '-r', root]
        if as_set:
            pargs.append('--as-set')
        return self.hc._run_command(self, pargs)

    def annotate_global_table(self, input, root, config=None):
        """Load delimited text file (text table) into global annotations as
        Array[Struct].

        **Examples**

        Load a file as a global annotation.  Consider the file *data/genes.txt* with contents:

        .. code-block:: text

          GENE    PLI     EXAC_LOF_COUNT
          Gene1   0.12312 2
          ...

        >>> (hc.read('data/example.vds')
        >>>   .annotate_global_table('data/genes.txt', 'global.genes',
        >>>                          TextTableConfig(types='PLI: Double, EXAC_LOF_COUNT: Int')))

        creates a new global annotation ``global.genes`` with type:

        .. code-block:: text

          global.genes: Array[Struct {
              GENE: String,
              PLI: Double,
              EXAC_LOF_COUNT: Int
          }]

        where each line is stored as an element of the array.

        **Notes**

        :param str input: Input text file.
        :param str root: Global annotation path to store text table.
        :param config: Configuration options for importing text files
        :type config: :class:`.TextTableConfig` or None

        :return: Annotated dataset.
        :rtype: :class:`.VariantDataset`
        """

        pargs = ['annotateglobal', 'table', '-i', input, '-r', root]

        if not config:
            config = TextTableConfig()

        pargs.extend(config._as_pargs())

        return self.hc._run_command(self, pargs)

    def annotate_samples_expr(self, condition):
        """Annotate samples with expression.

        **Example**

        Compute per-sample GQ statistics for hets:

        >>> (hc.read('data/example.vds')
        >>>   .annotate_samples_expr('sa.gqHetStats = gs.filter(g => g.isHet).map(g => g.gq).stats()')
        >>>   .export_samples('data/samples.txt', 'sample = s, het_gq_mean = sa.gqHetStats.mean'))

        Compute the list of genes with a singleton LOF:

        >>> (hc.read('data/example.vds')
        >>>   .annotate_variants_table('data/consequence.tsv', 'Variant', code='va.consequence = table.Consequence')
        >>>   .annotate_variants_expr('va.isSingleton = gs.map(g => g.nNonRefAlleles).sum() == 1')
        >>>   .annotate_samples_expr('sa.LOF_genes = gs.filter(g => va.isSingleton && g.isHet && va.consequence == "LOF").map(g => va.gene).collect()'))

        **Notes**

        ``condition`` is in sample context so the following symbols are in scope:

        - ``s`` (*Sample*): :ref:`sample`
        - ``sa``: sample annotations
        - ``global``: global annotations
        - ``gs`` (*Aggregable[Genotype]*): aggregable of :ref:`genotype` for sample ``s``

        :param condition: Annotation expression.
        :type condition: str or list of str

        :return: Annotated dataset.
        :rtype: :class:`.VariantDataset`
        """

        if isinstance(condition, list):
            condition = ','.join(condition)
        pargs = ['annotatesamples', 'expr', '-c', condition]
        return self.hc._run_command(self, pargs)

    def annotate_samples_fam(self, input, quantpheno=False, delimiter='\\\\s+', root='sa.fam', missing='NA'):
        """Import PLINK .fam file into sample annotations.

        **Examples**

        Import case-control phenotype data from a tab-separated `PLINK .fam
        <https://www.cog-genomics.org/plink2/formats#fam>`_ file into sample
        annotations:

        >>> vds.annotate_samples_fam("data/myStudy.fam")

        In Hail, unlike Plink, the user must *explicitly* distinguish between
        case-control and quantitative phenotypes. Importing a quantitative
        phenotype without ``quantPheno=True`` will return an error
        (unless all values happen to be ``0``, ``1``, ``2``, and ``-9``):

        >>> vds.annotate_samples_fam("data/myStudy.fam", quantPheno=True)

        Import case-control phenotype data from an
        arbitrary-whitespace-delimited `PLINK .fam
        <https://www.cog-genomics.org/plink2/formats#fam>`_ file into sample
        annotations:

        >>> vds.annotate_samples_fam("data/myStudy.fam", delimiter="\\s+")

        **Annotations**

        The annotation names, types, and missing values are shown below,
        assuming the default root ``sa.fam``.

        - **sa.fam.famID** (*String*) -- Family ID (missing = "0")
        - **s** (*String*) -- Sample ID
        - **sa.fam.patID** (*String*) -- Paternal ID (missing = "0")
        - **sa.fam.matID** (*String*) -- Maternal ID (missing = "0")
        - **sa.fam.isFemale** (*Boolean*) -- Sex (missing = "NA", "-9", "0")
        - **sa.fam.isCase** (*Boolean*) -- Case-control phenotype (missing = "0", "-9", non-numeric or the ``missing`` argument, if given.
        - **sa.fam.qPheno** (*Double*) -- Quantitative phenotype (missing = "NA" or the ``missing`` argument, if given.

        :param str input: Path to .fam file.
        :param str root: Sample annotation path to store .fam file.
        :param bool quantpheno: If True, .fam phenotype is interpreted as quantitative.
        :param str delimiter: .fam file field delimiter regex.
        :param str missing: The string used to denote missing values.
            For case-control, 0, -9, and non-numeric are also treated
            as missing.

        :return: Annotated dataset with sample annotations from fam file.
        :rtype: :class:`.VariantDataset`

        """

        pargs = ['annotatesamples', 'fam', '-i', input, '--root', root, '--missing', missing]
        if quantpheno:
            pargs.append('--quantpheno')
        if delimiter:
            pargs.append('--delimiter')
            pargs.append(delimiter)
        return self.hc._run_command(self, pargs)

    def annotate_samples_list(self, input, root):
        """Annotate samples with a Boolean indicating presence in a list of samples in a text file.

        **Example**

        Add the sample annotation ``sa.inBatch1: Boolean`` with value true if the sample is in *batch1.txt*:

        >>> vds = (hc.read('data/example.vds')
        >>>  .annotate_samples_list('data/batch1.txt','sa.inBatch1'))

        The file must have no header and one sample per line

        .. code-block: text

            $ cat data/batch1.txt
            SampleA
            SampleB
            ...

        :param str input: Sample list file.
        :param str root: Sample annotation path to store Boolean.

        :return: annotated dataset with a new boolean sample annotation
        :rtype: :class:`.VariantDataset`
        """

        pargs = ['annotatesamples', 'list', '-i', input, '-r', root]
        return self.hc._run_command(self, pargs)

    def annotate_samples_table(self, input, sample_expr, root=None, code=None, config=None):
        """Annotate samples with delimited text file (text table).

        **Examples**

        To annotates samples using `samples1.tsv` with type imputation::

        >>> conf = hail.TextTableConfig(impute=True)
        >>> vds = (hc.read('data/example.vds')
        >>>  .annotate_samples_table('data/samples1.tsv', 'Sample', root='sa.pheno', config=conf))

        Given this file

        .. code-block: text

            $ cat data/samples1.tsv
            Sample	Height	Status  Age
            PT-1234	154.1	ADHD	24
            PT-1236	160.9	Control	19
            PT-1238	NA	ADHD	89
            PT-1239	170.3	Control	55

        the three new sample annotations are ``sa.pheno.Height: Double``, ``sa.pheno.Status: String``, and ``sa.pheno.Age: Int``.

        To annotate without type imputation, resulting in all String types:

        >>> vds = (hc.read('data/example.vds')
        >>>  .annotate_samples_table('data/samples1.tsv', 'Sample', root='sa.phenotypes'))

        **Detailed examples**

        Let's import annotations from a CSV file with missing data and special characters

        .. code-block: text

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

        - Escape the ``PT-ID`` column with backticks in the ``sample_expr`` argument because it contains a dash

        - Pass the non-default delimiter ``,``

        - Pass the non-default missing value ``.``

        - Add the only useful column using ``code`` rather than the ``root`` parameter.

        >>> conf = TextTableConfig(delimiter=',', missing='.')
        >>> vds = (hc.read('data/example.vds')
        >>>  .annotate_samples_table('data/samples2.tsv', '`PT-ID`', code='sa.batch = table.Batch', config=conf))

        Let's import annotations from a file with no header and sample IDs that need to be transformed. Suppose the vds sample IDs are of the form ``NA#####``. This file has no header line, and the sample ID is hidden in a field with other information

        .. code-block: text

            $ cat data/samples3.tsv
            1kg_NA12345   female
            1kg_NA12346   male
            1kg_NA12348   female
            pgc_NA23415   male
            pgc_NA23418   male

        To import it:

        >>> conf = TextTableConfig(noheader=True)
        >>> vds = (hc.read('data/example.vds')
        >>>  .annotate_samples_table('data/samples3.tsv', '_0.split("_")[1]', code='sa.sex = table._1, sa.batch = table._0.split("_")[0]', config=conf))

        **Using the** ``sample_expr`` **argument**

        This argument tells Hail how to get a sample ID out of your table. Each column in the table is exposed to the Hail expr language. Possibilities include ``Sample`` (if your sample id is in a column called 'Sample'), ``_2`` (if your sample ID is the 3rd column of a table with no header), or something more complicated like ``'if ("PGC" ~ ID1) ID1 else ID2'``.  All that matters is that this expr results in a string.  If the expr evaluates to missing, it will not be mapped to any VDS samples.

        **Using the** ``root`` **and** ``code`` **arguments**

        This module requires exactly one of these two arguments to tell Hail how to insert the table into the sample annotation schema.

        The ``root`` argument is the simpler of these two, and simply packages up all table annotations as a ``Struct`` and drops it at the given ``root`` location.  If your table has columns ``Sample``, ``Sex``, and ``Batch``, then ``root='sa.metadata'`` creates the struct ``{Sample, Sex, Batch}`` at ``sa.metadata``, which gives you access to the paths ``sa.metadata.Sample``, ``sa.metadata.Sex``, and ``sa.metadata.Batch``.

        The ``code`` argument expects an annotation expression and has access to ``sa`` (the sample annotations in the VDS) and ``table`` (a struct with all the columns in the table).  ``root='sa.anno'`` is equivalent to ``code='sa.anno = table'``.

        **Common uses for the** ``code`` **argument**

        Don't generate a full struct in a table with only one annotation column

        .. code-block: text

            code='sa.annot = table._1'

        Put annotations on the top level under `sa`

        .. code-block: text

            code='sa = merge(sa, table)'

        Load only specific annotations from the table

        .. code-block: text

            code='sa.annotations = select(table, toKeep1, toKeep2, toKeep3)'

        The above is equivalent to

        .. code-block: text

            code='sa.annotations.toKeep1 = table.toKeep1,
                sa.annotations.toKeep2 = table.toKeep2,
                sa.annotations.toKeep3 = table.toKeep3'


        :param str input: Path to delimited text file.
        :param str sample_expr: Expression for sample id (key).
        :param str root: Sample annotation path to store text table.
        :param str code: Annotation expression.
        :param config: Configuration options for importing text files
        :type config: :class:`.TextTableConfig` or None

        :return: annotated dataset with new samples annotations imported from a text file
        :rtype: :class:`.VariantDataset`
        """

        pargs = ['annotatesamples', 'table', '-i', input, '--sample-expr', sample_expr]
        if root:
            pargs.append('--root')
            pargs.append(root)
        if code:
            pargs.append('--code')
            pargs.append(code)

        if not config:
            config = TextTableConfig()

        pargs.extend(config._as_pargs())

        return self.hc._run_command(self, pargs)

    def annotate_samples_vds(self, right, root=None, code=None):
        """Annotate samples with sample annotations from .vds file.

        :param VariantDataset right: VariantDataset to annotate with.

        :param str root: Sample annotation path to add sample annotations.

        :param str code: Annotation expression.

        """

        return VariantDataset(
            self.hc,
            self.hc._hail.driver.AnnotateSamplesVDS.annotate(
                self._jvds, right._jvds, code, root))

    def annotate_variants_bed(self, input, root, all=False):
        """Annotate variants based on the intervals in a .bed file.

        **Examples**

        Add the variant annotation ``va.cnvRegion: Boolean`` indicating inclusion in at least one interval of the three-column BED file `file1.bed`:

        >>> vds = (hc.read('data/example.vds')
        >>>  .annotate_variants_bed('data/file1.bed', 'va.cnvRegion'))

        Add a variant annotation ``va.cnvRegion: String`` with value given by the fourth column of `file2.bed`::

        >>> vds = (hc.read('data/example.vds')
        >>>  .annotate_variants_bed('data/file2.bed', 'va.cnvRegion'))

        The file formats are

        .. code-block: text

            $ cat data/file1.bed
            track name="BedTest"
            20    1          14000000
            20    17000000   18000000
            ...

            $ cat file2.bed
            track name="BedTest"
            20    1          14000000  cnv1
            20    17000000   18000000  cnv2
            ...


        **Details**

        `UCSC bed files <https://genome.ucsc.edu/FAQ/FAQformat.html#format1>`_ can have up to 12 fields, but Hail will only ever look at the first four.  The first three fields are required (``chrom``, ``chromStart``, and ``chromEnd``).  If a fourth column is found, Hail will parse this field as a string and load it into the specified annotation path.  If the bed file has only three columns, Hail will assign each variant a Boolean annotation, true if and only if the variant lies in the union of the intervals. Hail ignores header lines in BED files.

        If the ``all`` parameter is set to ``True`` and a fourth column is present, the annotation will be the set (possibly empty) of fourth column strings as a ``Set[String]`` for all intervals that overlap the given variant.

        .. caution:: UCSC BED files are end-exclusive but 0-indexed, so the line "5  100  105" is interpreted in Hail as loci `5:101, 5:102, 5:103, 5:104. 5:105`. Details `here <http://genome.ucsc.edu/blog/the-ucsc-genome-browser-coordinate-counting-systems/>`_.

        :param str input: Path to .bed file.

        :param str root: Variant annotation path to store annotation.

        :param bool all: Store values from all overlapping intervals as a set.

        :return: Annotated dataset with new variant annotations imported from a .bed file.
        :rtype: :class:`.VariantDataset`

        """

        pargs = ['annotatevariants', 'bed', '-i', input, '--root', root]
        if all:
            pargs.append('--all')
        return self.hc._run_command(self, pargs)

    def annotate_variants_expr(self, condition):
        """Annotate variants with expression.

        **Examples**

        Compute GQ statistics about heterozygotes per variant:

        >>> (hc.read('data/example.vds')
        >>>    .annotate_variants_expr('va.gqHetStats = '
        >>>                                'gs.filter(g => g.isHet).map(g => g.gq).stats()'))

        Collect a list of sample IDs with non-ref calls in LOF variants:

        >>> (hc.read('data/example.vds')
        >>>    .annotate_variants_expr('va.nonRefSamples = gs.filter(g => g.isCalledNonRef).map(g => s.id).collect()'))

        **Notes**

        ``condition`` is in variant context so the following symbols are in scope:

          - ``v`` (*Variant*): :ref:`variant`
          - ``va``: variant annotations
          - ``global``: global annotations
          - ``gs`` (*Aggregable[Genotype]*): aggregable of :ref:`genotype` for variant ``v``

        For more information, see the documentation on writing `expressions <overview.html#expressions>`_
        and using the `Hail Expression Language <../expr_lang.html>`_.

        :param condition: Annotation expression or list of annotation expressions.
        :type condition: str or list of str

        :return: Annotated dataset.
        :rtype: :class:`.VariantDataset`

        """
        if isinstance(condition, list):
            condition = ','.join(condition)
        pargs = ['annotatevariants', 'expr', '-c', condition]
        return self.hc._run_command(self, pargs)

    def annotate_variants_intervals(self, input, root, all=False):
        """Annotate variants from an interval list file.

        **Examples**

        Consider the file, *data/exons.interval_list*, in
        ``chromosome:start-end`` format:

        .. code-block:: text

            $ cat data/exons.interval_list
            1:5122980-5123054
            1:5531412-5531715
            1:5600022-5601025
            1:5610246-5610349

        The following invocation produces a vds with a new variant annotation,
        ``va.inExon``. The annotation ``va.inExon`` is ``true`` for every
        variant included by ``exons.interval_list`` and false otherwise.

        >>> vds.annotate_variants_intervals('data/exons.interval_list', 'va.inExon')

        Consider the tab-separated, five-column file *data/exons2.interval_list*:

        .. code-block:: text

            $ cat data/exons2.interval_list
            1   5122980 5123054 + gene1
            1   5531412 5531715 + gene1
            1   5600022 5601025 - gene2
            1   5610246 5610349 - gene2

        This file maps from variant intervals to gene names. The following
        invocation produces a vds with a new variant annotation ``va.gene``. The
        annotation ``va.gene`` is set to the gene name occurring in the fifth
        column and ``NA`` otherwise.

        >>> vds.annotate_variants_intervals('data/exons2.interval_list', 'va.gene')

        **Notes**

        There are two formats for interval list files.  The first appears as
        ``chromosome:start-end`` as in the first example.  This format will
        annotate variants with a *Boolean*, which is ``true`` if that variant is
        found in any interval specified in the file and `false` otherwise.

        The second interval list format is a TSV with fields chromosome, start,
        end, strand, target.  **There should not be a header.** This file will
        annotate variants with the *String* in the fifth column (target). If
        ``all=True``, the annotation will be the, possibly empty,
        ``Set[String]`` of fifth column strings (targets) for all intervals that
        overlap the given variant.

        :param str input: Path to .interval_list.
        :param str root: Variant annotation path to store annotation.
        :param bool all: If true, store values from all overlapping
            intervals as a set.

        :return: Annotated dataset.
        :rtype: :py:class:`.VariantDataset`

        """

        pargs = ['annotatevariants', 'intervals', '-i', input, '--root', root]
        if all:
            pargs.append('--all')
        return self.hc._run_command(self, pargs)

    def annotate_variants_loci(self, path, locus_expr, root=None, code=None, config=None):
        """Annotate variants from an delimited text file (text table) indexed
        by loci.

        :param str path: Path to delimited text file.
        :param str locus_expr: Expression for locus (key).
        :param str root: Variant annotation path to store annotation.
        :param str code: Annotation expression.
        :param config: Configuration options for importing text files
        :type config: :class:`.TextTableConfig` or None

        :return: Annotated dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        pargs = ['annotatevariants', 'loci', '--locus-expr', locus_expr]

        if root:
            pargs.append('--root')
            pargs.append(root)

        if code:
            pargs.append('--code')
            pargs.append(code)

        if not config:
            config = TextTableConfig()

        pargs.extend(config._as_pargs())

        if isinstance(path, str):
            pargs.append(path)
        else:
            for p in path:
                pargs.append(p)

        return self.hc._run_command(self, pargs)

    def annotate_variants_table(self, path, variant_expr, root=None, code=None, config=None):
        """Annotate variant with delimited text file (text table).

        :param path: Path to delimited text files.
        :type path: str or list of str
        :param str variant_expr: Expression for Variant (key).
        :param str root: Variant annotation path to store text table.
        :param str code: Annotation expression.
        :param config: Configuration options for importing text files
        :type config: :class:`.TextTableConfig` or None

        :return: Annotated dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        pargs = ['annotatevariants', 'table', '--variant-expr', variant_expr]

        if root:
            pargs.append('--root')
            pargs.append(root)

        if code:
            pargs.append('--code')
            pargs.append(code)

        if not config:
            config = TextTableConfig()

        pargs.extend(config._as_pargs())

        if isinstance(path, str):
            pargs.append(path)
        else:
            for p in path:
                pargs.append(p)

        return self.hc._run_command(self, pargs)

    def annotate_variants_vds(self, other, code=None, root=None):
        '''Annotate variants with variant annotations from .vds file.

        **Examples**

        Copy the ``anno1`` annotation from ``other`` to ``va.annot``:

        >>> vds.annotate_variants_vds(code='va.annot = vds.anno1')

        Merge the variant annotations from the two vds together and places them
        at ``va``:

        >>> vds.annotate_variants_vds(code='va = merge(va, vds)')

        Select a subset of the annotations from ``other``:

        >>> vds.annotate_variants_vds(code='va.annotations = select(vds, toKeep1, toKeep2, toKeep3)')

        The previous expression is equivalent to:

        >>> vds.annotate_variants_vds(code="""va.annotations.toKeep1 = vds.toKeep1,
        >>>                                   va.annotations.toKeep2 = vds.toKeep2,
        >>>                                   va.annotations.toKeep3 = vds.toKeep3""")

        **Notes**

        Using this method requires one of the two optional arguments: ``code``
        and ``root``. They specify how to insert the annotations from ``other``
        into the this vds's variant annotations.

        The ``root`` argument copies all the variant annotations from ``other``
        to the specified annotation path.

        The ``code`` argument expects an annotation expression whose scope
        includes, ``va``, the variant annotations in the current VDS, and ``vds``,
        the variant annotations in ``other``.

        VDSes with multi-allelic variants may produce surprising results because
        all alternate alleles are considered part of the variant key. For
        example:

        - The variant ``22:140012:A:T,TTT`` will not be annotated by
          ``22:140012:A:T`` or ``22:140012:A:TTT``

        - The variant ``22:140012:A:T`` will not be annotated by
          ``22:140012:A:T,TTT``

        It is possible that an unsplit dataset contains no multiallelic
        variants, so ignore any warnings Hail prints if you know that to be the
        case.  Otherwise, run :py:meth:`.split_multi` before
        :py:meth:`.annotate_variants_vds`.

        :param VariantDataset other: VariantDataset to annotate with.
        :param str root: Sample annotation path to add variant annotations.
        :param str code: Annotation expression.

        :return: Annotated dataset.
        :rtype: :py:class:`.VariantDataset`
        '''

        return VariantDataset(
            self.hc,
            self.hc._hail.driver.AnnotateVariantsVDS.annotate(
                self._jvds, other._jvds, code, root))

    def cache(self):
        """Mark this dataset to be cached in memory.

        :py:meth:`~hail.VariantDataset.cache` is the same as :func:`persist("MEMORY_ONLY") <hail.VariantDataset.persist>`.
        """

        pargs = ['cache']
        self.hc._run_command(self, pargs)

    def concordance(self, right):
        """Calculate call concordance with another dataset.

        Performs inner join on variants, outer join on samples.

        :param right: right hand dataset for concordance
        :type right: :class:`.VariantDataset`

        :return: Returns a pair of datasets with the sample and
            variant concordance, respectively.
        :rtype: (:py:class:`.VariantDataset`, :py:class:`.VariantDataset`)

        """

        result = env.hail.driver.Concordance.calculate(
            self._jvds, right._jvds)
        return (VariantDataset(self.hc, result._1()),
                VariantDataset(self.hc, result._2()))

    def count(self, genotypes=False):
        """Return number of samples, variants and genotypes.

        :param bool genotypes: If True, include number of called
            genotypes and genotype call rate.
        """

        try:
            return (scala_package_object(self.hc._hail.driver)
                    .count(self._jvds, genotypes)
                    .toJavaMap())
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def deduplicate(self):
        """Remove duplicate variants.

        :return: Deduplicated dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        pargs = ['deduplicate']
        return self.hc._run_command(self, pargs)

    def downsample_variants(self, keep):
        """Downsample variants.

        :param int keep: (Expected) number of variants to keep.

        :return: Downsampled dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        pargs = ['downsamplevariants', '--keep', str(keep)]
        return self.hc._run_command(self, pargs)

    def export_gen(self, output):
        """Export dataset as GEN and SAMPLE file.

        **Examples**

        Import dosage data, filter variants based on INFO score, and export data to a GEN and SAMPLE file:

        >>> vds = (hc.index_bgen("data/example.bgen")
        >>>         .import_bgen("data/example.bgen", sample_file="data/example.sample"))
        >>> (vds.filter_variants_expr("gs.infoScore() >= 0.9")
        >>>     .export_gen("data/example_filtered"))

        **Notes**

        Writes out the internal VDS to a GEN and SAMPLE fileset in the `Oxford spec <http://www.stats.ox.ac.uk/%7Emarchini/software/gwas/file_format.html>`_.

        The first 6 columns of the resulting GEN file are the following:

        - Chromosome (``v.contig``)
        - Variant ID (``va.varid`` if defined, else Chromosome:Position:Ref:Alt)
        - rsID (``va.rsid`` if defined, else ".")
        - position (``v.start``)
        - reference allele (``v.ref``)
        - alternate allele (``v.alt``)

        Probability dosages:

        - 3 probabilities per sample ``(pHomRef, pHet, pHomVar)``.
        - Any filtered genotypes will be output as ``(0.0, 0.0, 0.0)``.
        - If the input data contained Phred-scaled likelihoods, the probabilities in the GEN file will be the normalized genotype probabilities assuming a uniform prior.
        - If the input data did not have genotype probabilities such as data imported using :py:meth:`~hail.HailContext.import_plink`, all genotype probabilities will be ``(0.0, 0.0, 0.0)``.

        The sample file has 3 columns:

        - ID_1 and ID_2 are identical and set to the sample ID (``s.id``).
        - The third column ("missing") is set to 0 for all samples.

        :param str output: Output file base.  Will write GEN and SAMPLE files.
        """

        pargs = ['exportgen', '--output', output]
        self.hc._run_command(self, pargs)

    def export_genotypes(self, output, condition, types=None, export_ref=False, export_missing=False):
        """Export genotype-level information to delimited text file.

        **Examples**

        Export genotype information with identifiers that form the header:

        >>> (hc.read('data/example.vds')
        >>>  .export_genotypes('data/genotypes.tsv', 'SAMPLE=s, VARIANT=v, GQ=g.gq, DP=g.dp, ANNO1=va.anno1, ANNO2=va.anno2'))

        Export the same information without identifiers, resulting in a file with no header:

        >>> (hc.read('data/example.vds')
        >>>  .export_genotypes('data/genotypes.tsv', 's, v, s.id, g.dp, va.anno1, va.anno2'))

        **Details**

        :py:meth:`~hail.VariantDataset.export_genotypes` outputs one line per cell (genotype) in the data set, though HomRef and missing genotypes are not output by default. Use the ``export_ref`` and ``export_missing`` parameters to force export of HomRef and missing genotypes, respectively.

        The ``condition`` argument is a comma-separated list of fields or expressions, all of which must be of the form ``IDENTIFIER = <expression>``, or else of the form ``<expression>``.  If some fields have identifiers and some do not, Hail will throw an exception. The accessible namespace includes ``g``, ``s``, ``sa``, ``v``, ``va``, and ``global``.

        :param str output: Output path.
        :param str condition: Annotation expression for values to export.
        :param types: Path to write types of exported values.
        :type types: str or None
        :param bool export_ref: If True, export reference genotypes.
        :param bool export_missing: If True, export missing genotypes.
        """

        pargs = ['exportgenotypes', '--output', output, '-c', condition]
        if types:
            pargs.append('--types')
            pargs.append(types)
        if export_ref:
            pargs.append('--print-ref')
        if export_missing:
            pargs.append('--print-missing')
        self.hc._run_command(self, pargs)

    def export_plink(self, output, fam_expr='id = s.id'):
        """Export dataset as `PLINK2 <https://www.cog-genomics.org/plink2/formats>`_ BED, BIM and FAM.

        **Examples**

        Import data from a VCF file, split multi-allelic variants, and export to a PLINK binary file:

        >>> (hc.import_vcf('data/example.vcf')
        >>>   .split_multi()
        >>>   .export_plink('data/plink'))

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

        - ``s`` (*Sample*): :ref:`sample`
        - ``sa``: sample annotations
        - ``global``: global annotations

        The BIM file ID field is set to ``CHR:POS:REF:ALT``.

        This code::

        >>> (hc.import_vcf('data/example.vcf')
        >>>   .split_multi()
        >>>   .export_plink('data/plink'))

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

        pargs = ['exportplink', '--output', output, '--fam-expr', fam_expr]
        self.hc._run_command(self, pargs)

    def export_samples(self, output, condition, types=None):
        """Export sample information to delimited text file.

        **Examples**

        Export some sample QC metrics:

        >>> (hc.read('data/example.vds')
        >>>   .sample_qc()
        >>>   .export_samples('data/samples.tsv', 'SAMPLE = s, CALL_RATE = sq.qc.callRate, NHET = sa.qc.nHet'))

        This will produce a file with a header and three columns.  To
        produce a file with no header, just leave off the assignment
        to the column identifier:

        >>> (hc.read('data/example.vds')
        >>>   .sample_qc()
        >>>   .export_samples('data/samples.tsv', 's, CALL_RATE = sq.qc.rTiTv'))

        **Notes**

        One line per sample will be exported.  As :py:meth:`~hail.VariantDataset.export_samples` runs in sample context, the following symbols are in scope:

        - ``s`` (*Sample*): :ref:`sample`
        - ``sa``: sample annotations
        - ``global``: global annotations
        - ``gs`` (*Aggregable[Genotype]*): aggregable of :ref:`genotype` for sample ``s``

        :param str output: Output file.
        :param str condition: Annotation expression for values to export.
        :param types: Path to write types of exported values.
        :type types: str or None
        """

        pargs = ['exportsamples', '--output', output, '-c', condition]
        if types:
            pargs.append('--types')
            pargs.append(types)
        self.hc._run_command(self, pargs)

    def export_variants(self, output, condition, types=None):
        """Export variant information to delimited text file.

        **Examples**

        Export a four column TSV with ``v``, ``va.pass``, ``va.filters``, and
        one computed field: ``1 - va.qc.callRate``.

        >>> vds.export_variants('data/file.tsv',
        >>>   'VARIANT = v, PASS = va.pass, FILTERS = va.filters, MISSINGNESS = 1 - va.qc.callRate')

        It is also possible to export without identifiers, which will result in
        a file with no header. In this case, the expressions should look like
        the examples below:

        >>> vds.export_variants('data/file.tsv', 'v, va.pass, va.qc.AF')

        .. note::

            Either all fields must be named, or no field must be named.

        In the common case that a group of annotations needs to be exported (for
        example, the annotations produced by ``variantqc``), one can use the
        ``struct.*`` syntax.  This syntax produces one column per field in the
        struct, and names them according to the struct field name.

        For example, the following invocation (assuming ``va.qc`` was generated
        by :py:meth:`.variant_qc`):

        >>> vds.export_variants('data/file.tsv', 'variant = v, va.qc.*')

        will produce the following set of columns:

        .. code-block:: text

            variant  callRate  AC  AF  nCalled  ...

        Note that using the ``.*`` syntax always results in named arguments, so it
        is not possible to export header-less files in this manner.  However,
        naming the "splatted" struct will apply the name in front of each column
        like so:

        >>> vds.export_variants('data/file.tsv', 'variant = v, QC = va.qc.*')

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

        Much like the filtering methods, exporting allows flexible expressions
        to be written on the command line. While the filtering methods expect an
        expression that evaluates to true or false, this method expects a
        comma-separated list of fields to print. These fields *must* take the
        form ``IDENTIFIER = <expression>``.


        :param str output: Output file.
        :param str condition: Annotation expression for values to export.
        :param types: Path to write types of exported values.
        :type types: str or None
        """

        pargs = ['exportvariants', '--output', output, '-c', condition]
        if types:
            pargs.append('--types')
            pargs.append(types)
        self.hc._run_command(self, pargs)

    def export_variants_cass(self, variant_condition, genotype_condition,
                             address,
                             keyspace,
                             table,
                             export_missing=False,
                             export_ref=False):
        """Export variant information to Cassandra."""

        pargs = ['exportvariantscass', '-v', variant_condition, '-g', genotype_condition,
                 '-a', address, '-k', keyspace, '-t', table]
        if export_missing:
            pargs.append('--export-missing')
        if export_ref:
            pargs.append('--export-ref')
        self.hc._run_command(self, pargs)

    def export_variants_solr(self, variant_condition, genotype_condition,
                             solr_url=None,
                             solr_cloud_collection=None,
                             zookeeper_host=None,
                             drop=False,
                             num_shards=1,
                             export_missing=False,
                             export_ref=False,
                             block_size=100):
        """Export variant information to Solr."""

        pargs = ['exportvariantssolr', '-v', variant_condition, '-g', genotype_condition, '--block-size', block_size]
        if solr_url:
            pargs.append('-u')
            pargs.append(solr_url)
        if solr_cloud_collection:
            pargs.append('-c')
            pargs.append(solr_cloud_collection)
        if zookeeper_host:
            pargs.append('-z')
            pargs.append(zookeeper_host)
        if drop:
            pargs.append('--drop')
        if num_shards:
            pargs.append('--num-shards')
            pargs.append(num_shards)
        if export_missing:
            pargs.append('--export-missing')
        if export_ref:
            pargs.append('--export-ref')
        self.hc._run_command(self, pargs)

    def export_vcf(self, output, append_to_header=None, export_pp=False, parallel=False):
        """Export as .vcf file.

        :param str output: Path of .vcf file to write.
        :param append_to_header: Path of file to append to .vcf header.
        :type append_to_header: str or None
        :param bool export_pp: If True, export Hail pl genotype field as VCF PP FORMAT field.
        :param bool parallel: If True, export .vcf in parallel.
        """

        pargs = ['exportvcf', '--output', output]
        if append_to_header:
            pargs.append('-a')
            pargs.append(append_to_header)
        if export_pp:
            pargs.append('--export-pp')
        if parallel:
            pargs.append('--parallel')
        self.hc._run_command(self, pargs)

    def write(self, output, overwrite=False):
        """Write as VDS file.

        **Examples**

        Import data from a VCF file and then write the data to a VDS file:

        >>> (hc.import_vcf("data/sample.vcf.bgz")
        >>>    .write("data/sample.vds"))

        :param str output: Path of .vds file to write.
        :param bool overwrite: If True, overwrite any existing VDS file. Cannot be used to read from and write to the same path.
        """

        pargs = ['write', '-o', output]
        if overwrite:
            pargs.append('--overwrite')
        self.hc._run_command(self, pargs)

    def filter_alleles(self, condition, annotation=None, subset=True, keep=True, filter_altered_genotypes=False):
        """Filter a user-defined set of alternate alleles for each variant.
        If all alternate alleles of a variant are filtered, the
        variant itself is filtered.  The condition expression is
        evaluated for each alternate allele, but not for
        the reference allele (i.e. ``aIndex`` will never be zero).

        **Example**

        To remove alternate alleles with zero allele count and
        update the alternate allele count annotation with the new
        indices:

        >>> (hc.read('example.vds')
        >>>   .filter_alleles('va.info.AC[aIndex - 1] == 0',
        >>>     'va.info.AC = aIndices[1:].map(i => va.info.AC[i - 1])',
        >>>     keep=False))

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

        The following symbols are in scope for ``condition``:

        - ``v`` (*Variant*): :ref:`variant`
        - ``va``: variant annotations
        - ``aIndex`` (*Int*): the index of the allele being tested

        The following symbols are in scope for ``annotation``:

        - ``v`` (*Variant*): :ref:`variant`
        - ``va``: variant annotations
        - ``aIndices`` (*Array[Int]*): the array of old indices (such that ``aIndices[newIndex] = oldIndex`` and ``aIndices[0] = 0``)

        :param condition: Filter expression involving v (variant), va (variant annotations), and aIndex (allele index)
        :type condition: str
        :param annotation: Annotation modifying expression involving v (new variant), va (old variant annotations),
            and aIndices (maps from new to old indices)
        :param bool subset: If true, subsets PL and AD, otherwise downcodes the PL and AD.
            Genotype and GQ are set based on the resulting PLs.
        :param bool keep: If true, keep variants matching condition
        :param bool filter_altered_genotypes: If true, genotypes that contain filtered-out alleles are set to missing.

        :return: Filtered dataset.
        :rtype: :class:`.VariantDataset`
        """

        pargs = ['filteralleles',
                 '--keep' if keep else '--remove',
                 '--subset' if subset else '--downcode',
                 '-c', condition]

        if annotation:
            pargs.extend(['-a', annotation])

        if filter_altered_genotypes:
            pargs.append('--filterAlteredGenotypes')

        return self.hc._run_command(self, pargs)

    def filter_genotypes(self, condition, keep=True):
        """Filter genotypes based on expression.

        **Examples**

        Filter genotypes by allele balance dependent on genotype call:

        >>> (vds.filter_genotypes('let ab = g.ad[1] / g.ad.sum in'
        >>>                      '((g.isHomRef && ab <= 0.1) ||'
        >>>                      '(g.isHet && ab >= 0.25 && ab <= 0.75) ||'
        >>>                      '(g.isHomVar && ab >= 0.9))'))

        **Notes**


        ``condition`` is in genotype context so the following symbols are in scope:

        - ``s`` (*Sample*): :ref:`sample`
        - ``v`` (*Variant*): :ref:`variant`
        - ``sa``: sample annotations
        - ``va``: variant annotations
        - ``global``: global annotations

        For more information, see the documentation on `data representation, annotations <overview.html#>`_, and
        the `expression language <../expr_lang.html>`_.

        .. caution::
            When ``condition`` evaluates to missing, the genotype will be removed regardless of whether ``keep=True`` or ``keep=False``.

        :param condition: Expression for filter condition.
        :type condition: str

        :return: Filtered dataset.
        :rtype: :class:`.VariantDataset`
        """

        pargs = ['filtergenotypes',
                 '--keep' if keep else '--remove',
                 '-c', condition]
        return self.hc._run_command(self, pargs)

    def filter_multi(self):
        """Filter out multi-allelic sites.

        This method is much less computationally expensive than
        :py:meth:`.split_multi`, and can also be used to produce
        a dataset that can be used with methods that do not
        support multiallelic variants.

        :return: Dataset with no multiallelic sites, which can
            be used for biallelic-only methods.
        :rtype: :class:`.VariantDataset`
        """

        pargs = ['filtermulti']
        return self.hc._run_command(self, pargs)

    def filter_samples_all(self):
        """Removes all samples from dataset.

        The variants, variant annotations, and global annnotations will remain,
        producing a sites-only dataset.

        :return: Sites-only dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        pargs = ['filtersamples', 'all']
        return self.hc._run_command(self, pargs)

    def filter_samples_expr(self, condition, keep=True):
        """Filter samples based on expression.

        **Examples**

        Filter samples by phenotype (assumes sample annotation *sa.isCase* exists and is a Boolean variable):

        >>> vds.filter_samples_expr("sa.isCase")

        Remove samples with an ID that matches a regular expression:

        >>> vds.filter_samples_expr('"^NA" ~ s' , keep=False)

        Filter samples from sample QC metrics and write output to a new dataset:

        >>> (vds.sample_qc()
        >>>     .filter_samples_expr('sa.qc.callRate >= 0.99 && sa.qc.dpMean >= 10')
        >>>     .write("data/filter_samples.vds"))

        **Notes**

        ``condition`` is in sample context so the following symbols are in scope:

        - ``s`` (*Sample*): :ref:`sample`
        - ``sa``: sample annotations
        - ``global``: global annotations
        - ``gs`` (*Aggregable[Genotype]*): aggregable of :ref:`genotype` for sample ``s``

        For more information, see the documentation on `data representation, annotations <overview.html#>`_, and
        the `expression language <../expr_lang.html>`_.

        .. caution::
            When ``condition`` evaluates to missing, the sample will be removed regardless of whether ``keep=True`` or ``keep=False``.


        :param condition: Expression for filter condition.
        :type condition: str

        :return: Filtered dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        pargs = ['filtersamples', 'expr',
                 '--keep' if keep else '--remove',
                 '-c', condition]
        return self.hc._run_command(self, pargs)

    def filter_samples_list(self, input, keep=True):
        """Filter samples with a sample list file.

        **Example**

        >>> vds = (hc.read('data/example.vds')
        >>>   .filter_samples_list('exclude_samples.txt', keep=False))

        The file at the path ``input`` should contain on sample per
        line with no header or other fields.

        :param str input: Path to sample list file.

        :return: Filtered dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        pargs = ['filtersamples', 'list',
                 '--keep' if keep else '--remove',
                 '-i', input]
        return self.hc._run_command(self, pargs)

    def filter_variants_all(self):
        """Discard all variants, variant annotations and genotypes.

        Samples, sample annotations and global annotations are retained. This
         is the same as :func:`filter_variants_expr('false'), but much faster.

        **Example**

        >>> (hc.read('data/example.vds')
        >>>  .filter_variants_all())

        :return: Samples-only dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        pargs = ['filtervariants', 'all']
        return self.hc._run_command(self, pargs)

    def filter_variants_expr(self, condition, keep=True):
        """Filter variants based on expression.

        **Examples**

        Keep variants in the gene CHD8 (assumes the variant annotation ``va.gene`` exists):

        >>> vds_filtered = (hc.read('data/example.vds')
        >>>                   .filter_variants_expr('va.gene == "CHD8"'))


        Remove variants on chromosome 1:

        >>> vds_filtered = (hc.read('data/example.vds')
        >>>                   .filter_variants_expr('v.contig == "1"',
        >>>                                          keep=False))


        **Notes**

        ``condition`` is in variant context so the following symbols are in scope:

        - ``v`` (*Variant*): :ref:`variant`
        - ``va``: variant annotations
        - ``global``: global annotations
        - ``gs`` (*Aggregable[Genotype]*): aggregable of :ref:`genotype` for variant ``v``

        For more information, see the documentation on `data representation, annotations <overview.html#>`_, and
        the `expression language <../expr_lang.html>`_.

        .. caution::
           When ``condition`` evaluates to missing, the variant will be removed regardless of whether ``keep=True`` or ``keep=False``.

        :param condition: Expression for filter condition.
        :type condition: str

        :return: Filtered dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        pargs = ['filtervariants', 'expr',
                 '--keep' if keep else '--remove',
                 '-c', condition]
        return self.hc._run_command(self, pargs)

    def filter_variants_intervals(self, input, keep=True):
        """Filter variants with an .interval_list file.

        **Examples**

        If *intervals.txt* contains intervals in the interval_list format, the
        following expression will produce a :py:class:`.VariantDataset` containg
        only variants included by the given intervals:

        >>> vds.filter_variants_intervals('data/intervals.txt')

        **The File Format**

        Hail expects an interval file to contain either three or five fields per
        line in the following formats:

        - ``contig:start-end``
        - ``contig  start  end`` (tab-separated)
        - ``contig  start  end  direction  target`` (tab-separated)

        In either case, Hail only uses the ``contig``, ``start``, and ``end``
        fields.  Each variant is evaluated against each line in the interval
        file, and any match will mark the variant to be included if
        ``keep=True`` and excluded if ``keep=False``.

        .. note::

            ``start`` and ``end`` match positions inclusively, e.g. ``start <= position <= end``

        :param str input: Path to .interval_list file.

        :return: Filtered dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        pargs = ['filtervariants', 'intervals',
                 '--keep' if keep else '--remove',
                 '-i', input]
        return self.hc._run_command(self, pargs)

    def filter_variants_list(self, input, keep=True):
        """Filter variants with a list of variants.

        **Examples**

        Keep all variants that occur in *data/variants.txt* (removing all other
        variants):

        >>> vds.filter_variants_list('data/variants.txt')

        Remove all variants that occur in *data/variants.txt*:

        >>> vds.filter_variants_list('data/variants.txt', keep=False)

        **File Format**

        Hail expects the given file to contain a variant per line following
        format: ``contig:pos:ref:alt1,alt2,...,altN``.

        :param str input: Path to variant list file.

        :return: Filtered dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        pargs = ['filtervariants', 'list',
                 '--keep' if keep else '--remove',
                 '-i', input]
        return self.hc._run_command(self, pargs)

    @property
    def globals(self):
        """Return global annotations as a python object.

        :return: Dataset global annotations.
        """
        if self._globals is None:
            self._globals = self.global_schema._convert_to_py(self._jvds.globalAnnotation())
        return self._globals

    def grm(self, format, output, id_file=None, n_file=None):
        """Compute the Genetic Relatedness Matrix (GMR).

        :param str format: Output format.  One of: rel, gcta-grm, gcta-grm-bin.
        :param str id_file: ID file.
        :param str n_file: N file, for gcta-grm-bin only.
        :param str output: Output file.
        """

        pargs = ['grm', '-f', format, '-o', output]
        if id_file:
            pargs.append('--id-file')
            pargs.append(id_file)
        if n_file:
            pargs.append('--N-file')
            pargs.append(n_file)
        self.hc._run_command(self, pargs)

    def hardcalls(self):
        """Drop all genotype fields except the GT field.

        A hard-called dataset is about 2 orders of magnitude
        smaller than a standard sequencing dataset, so this
        method is good for creating a much smaller, faster
        representation for downstream processing that only
        uses the GT field.

        :return: Dataset with no genotype metadata.
        :rtype: :py:class:`.VariantDataset`
        """

        pargs = ['hardcalls']
        return self.hc._run_command(self, pargs)

    def ibd(self, output, maf=None, unbounded=False, min=None, max=None):
        """Compute matrix of identity-by-descent estimations.

        **Examples**

        To estimate and write the full IBD matrix to *ibd.tsv*, estimated using minor allele frequencies computed from the dataset itself:

        >>> (hc.read('data/example.vds')
        >>>  .ibd('data/ibd.tsv'))

        To estimate IBD using minor allele frequencies stored in ``va.panel_maf`` and write to *ibd.tsv* only those sample pairs with ``pi_hat`` between 0.2 and 0.9 inclusive:

        >>> (hc.read('data/example.vds')
        >>>  .ibd('data/ibd.tsv', maf='va.panel_maf', min=0.2, max=0.9))

        **Details**

        The implementation is based on the IBD algorithm described in the `PLINK paper <http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1950838>`_.

        :py:meth:`~hail.VariantDataset.ibd` requires the dataset to be bi-allelic (otherwise run :py:meth:`~hail.VariantDataset.split_multi`) and does not perform LD pruning. Linkage disequilibrium may bias the result so consider filtering variants first.

        Conceptually, the output is a symmetric, sample-by-sample matrix. The output .tsv has the following form

        .. code-block: text

            SAMPLE_ID_1	SAMPLE_ID_2	Z0	Z1	Z2	PI_HAT
            sample1	sample2	1.0000	0.0000	0.0000	0.0000
            sample1	sample3	1.0000	0.0000	0.0000	0.0000
            sample1	sample4	0.6807	0.0000	0.3193	0.3193
            sample1	sample5	0.1966	0.0000	0.8034	0.8034

        :param str output: Output .tsv file for IBD matrix.
        :param maf: Expression for the minor allele frequency.
        :type maf: str or None
        :param bool unbounded: Allows the estimations for Z0, Z1, Z2,
            and PI_HAT to take on biologically nonsensical values
            (e.g. outside of [0,1]).
        :param min: "Sample pairs with a PI_HAT below this value will
            not be included in the output. Must be in [0,1].
        :type min: float or None
        :param max: Sample pairs with a PI_HAT above this value will
            not be included in the output. Must be in [0,1].
        :type max: float or None
        """

        pargs = ['ibd', '-o', output]
        if maf:
            pargs.append('-m')
            pargs.append(maf)
        if unbounded:
            pargs.append('--unbounded')
        if min:
            pargs.append('--min')
            pargs.append(min)
        if max:
            pargs.append('--min')
            pargs.append(max)
        self.hc._run_command(self, pargs)

    def impute_sex(self, maf_threshold=0.0, include_par=False, female_threshold=0.2, male_threshold=0.8, pop_freq=None):
        """Impute sex of samples by calculating inbreeding coefficient on the
        X chromosome.

        :param float maf_threshold: Minimum minor allele frequency threshold.
        :param bool include_par: Include pseudoautosomal regions.
        :param float female_threshold: Samples are called females if F < femaleThreshold
        :param float male_threshold: Samples are called males if F > maleThreshold
        :param str pop_freq: Variant annotation for estimate of MAF.
            If None, MAF will be computed.

        :return: Annotated dataset.
        :rtype: :py:class:`.VariantDataset`
        """

        pargs = ['imputesex']
        if maf_threshold:
            pargs.append('--maf-threshold')
            pargs.append(str(maf_threshold))
        if include_par:
            pargs.append('--include_par')
        if female_threshold:
            pargs.append('--female-threshold')
            pargs.append(str(female_threshold))
        if male_threshold:
            pargs.append('--male-threshold')
            pargs.append(str(male_threshold))
        if pop_freq:
            pargs.append('--pop-freq')
            pargs.append(pop_freq)
        return self.hc._run_command(self, pargs)

    def join(self, right):
        """Join datasets.

        This method performs an inner join on variants,
        concatenates samples, and takes variant and
        global annotations from the left dataset (self).

        :param right: right-hand dataset
        :type right: :py:class:`.VariantDataset`

        :return: Joined dataset
        :rtype: :py:class:`.VariantDataset`
        """
        try:
            return VariantDataset(self.hc, env.hail.driver.Join.join(self._jvds, right._jvds))
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def linreg(self, y, covariates=[], root='va.linreg', minac=1, minaf=0.0):
        r"""Test each variant for association using the linear regression
        model.

        **Example**

        To run linear regression with response and two covariates imported from a TSV file:

        >>> (hc.read('data/example.vds')
        >>>   .annotate_samples_table('data/pheno.tsv', root='sa.pheno', config=TextTableConfig(impute=True))
        >>>   .linreg('sa.pheno.isCase', covariates=['sa.pheno.age', 'sa.pheno.isFemale']))

        **Notes**

        The :py:meth:`.linreg` command computes, for each variant, statistics of
        the :math:`t`-test for the genotype coefficient of the linear function
        of best fit from sample genotype and covariates to quantitative
        phenotype or case-control status. Hail only includes samples for which
        phenotype and all covariates are defined. For each variant, Hail imputes
        missing genotypes as the mean of called genotypes.

        Assuming there are sample annotations ``sa.pheno.height``,
        ``sa.cov.age``, ``sa.cov.isFemale``, and ``sa.cov.PC1``, the command:

        >>> vds.linreg('sa.pheno.height', covariates=['sa.cov.age', 'sa.cov.isFemale', 'sa.cov.PC1'])

        considers a model of the form

        .. math::

            \mathrm{height} = \beta_0 + \beta_1 \, \mathrm{gt} + \beta_2 \, \mathrm{age} + \beta_3 \, \mathrm{isFemale} + \beta_4 \, \mathrm{PC1} + \varepsilon, \quad \varepsilon \sim \mathrm{N}(0, \sigma^2)

        where the genotype :math:`\mathrm{gt}` is coded as :math:`0` for HomRef, :math:`1` for
        Het, and :math:`2` for HomVar, and the Boolean covariate :math:`\mathrm{isFemale}`
        is coded as :math:`1` for true (female) and :math:`0` for false (male). The null
        model sets :math:`\beta_1 = 0`.

        :py:meth:`.linreg` skips variants that don't vary across the included samples,
        such as when all genotypes are homozygous reference. One can further
        restrict computation to those variants with at least :math:`k` observed
        alternate alleles (AC) or alternate allele frequency (AF) at least
        :math:`p` in the included samples using the options ``minac=k`` or
        ``minaf=p``, respectively. Unlike the :py:meth:`.filter_variants_expr`
        command, these filters do not remove variants from the underlying
        variant dataset. Adding both filters is equivalent to applying the more
        stringent of the two, as AF equals AC over twice the number of included
        samples.

        Phenotype and covariate sample annotations may also be specified using `programmatic expressions <../expr_lang.html>`_ without identifiers, such as:

        .. code-block:: text

          if (sa.isFemale) sa.cov.age else (2 * sa.cov.age + 10)

        For Boolean covariate types, true is coded as 1 and false as 0. In particular, for the sample annotation ``sa.fam.isCase`` added by importing a FAM file with case-control phenotype, case is 1 and control is 0.

        Hail's linear regression test corresponds to the ``q.lm`` test in
        `EPACTS <http://genome.sph.umich.edu/wiki/EPACTS#Single_Variant_Tests>`_. For
        each variant, Hail imputes missing genotypes as the mean of called
        genotypes, whereas EPACTS subsets to those samples with called
        genotypes. Hence, Hail and EPACTS results will currently only agree for
        variants with no missing genotypes.

        The standard least-squares linear regression model is derived in Section
        3.2 of `The Elements of Statistical Learning, 2nd Edition
        <http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf>`_. See
        equation 3.12 for the t-statistic which follows the t-distribution with
        :math:`n - k - 2` degrees of freedom, under the null hypothesis of no
        effect, with :math:`n` samples and :math:`k` covariates in addition to
        genotype and intercept.

        **Annotations**

        Four variant annotations are then added with root ``va.linreg`` as shown
        in the table. These annotations can then be accessed by other methods,
        including exporting to TSV with other variant annotations.

        - **va.linreg.beta** (*Double*) -- fit genotype coefficient, :math:`\hat\beta_1`
        - **va.linreg.se** (*Double*) -- estimated standard error, :math:`\widehat{\mathrm{se}}`
        - **va.linreg.tstat** (*Double*) -- :math:`t`-statistic, equal to :math:`\hat\beta_1 / \widehat{\mathrm{se}}`
        - **va.linreg.pval** (*Double*) -- :math:`p`-value

        :param str y: Response expression
        :param covariates: list of covariate expressions
        :type covariates: list of str
        :param str root: Variant annotation path to store result of linear regression.
        :param int minac: Minimum alternate allele count.
        :param float minaf: Minimum alternate allele frequency.

        :return: Dataset with linear regression variant annotations.
        :rtype: :py:class:`.VariantDataset`
        """

        try:
            return VariantDataset(
                self.hc, env.hail.methods.LinearRegression.apply(
                    self._jvds, y, jarray(env.jvm.java.lang.String, covariates), root, minac, minaf))
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def logreg(self, test, y, covariates=[], root='va.logreg'):
        """Test each variant for association using the logistic regression
        model.

        **Example**

        To run logistic regression using the Wald test with response and two covariates imported from a TSV file:

        >>> (hc.read('data/example.vds')
        >>>   .annotate_samples_table('data/pheno.tsv', root='sa.pheno', config=TextTableConfig(impute=True))
        >>>   .logreg('wald', 'sa.pheno.isCase', covariates=['sa.pheno.age', 'sa.pheno.isFemale']))

        **Notes**

        The :py:meth:`~hail.VariantDataset.logreg` command performs,
        for each variant, a significance test of the genotype in
        predicting a binary (case-control) phenotype based on the
        logistic regression model. Hail supports the Wald test,
        likelihood ratio test (LRT), and Rao score test. Hail only
        includes samples for which phenotype and all covariates are
        defined. For each variant, Hail imputes missing genotypes as
        the mean of called genotypes.

        Assuming there are sample annotations ``sa.pheno.isCase``,
        ``sa.cov.age``, ``sa.cov.isFemale``, and ``sa.cov.PC1``, the
        command:

        >>> vds.logreg('sa.pheno.isCase', covariates=['sa.cov.age' , 'sa.cov.isFemale', 'sa.cov.PC1'])

        considers a model of the form

        .. math::

          \mathrm{Prob}(\mathrm{isCase}) = \mathrm{sigmoid}(\\beta_0 + \\beta_1 \, \mathrm{gt} + \\beta_2 \, \mathrm{age} + \\beta_3 \, \mathrm{isFemale} + \\beta_4 \, \mathrm{PC1} + \\varepsilon), \quad \\varepsilon \sim \mathrm{N}(0, \sigma^2)

        where :math:`\mathrm{sigmoid}` is the `sigmoid
        function <https://en.wikipedia.org/wiki/Sigmoid_function>`_, the
        genotype :math:`\mathrm{gt}` is coded as 0 for HomRef, 1 for
        Het, and 2 for HomVar, and the Boolean covariate
        :math:`\mathrm{isFemale}` is coded as 1 for true (female) and
        0 for false (male). The null model sets :math:`\\beta_1 = 0`.

        The resulting variant annotations depend on the test statistic
        as shown in the tables below. These annotations can then be
        accessed by other methods, including exporting to TSV with
        other variant annotations.

        ===== ======================== ====== =====
        Test  Annotation               Type   Value
        ===== ======================== ====== =====
        Wald  ``va.logreg.wald.beta``  Double fit genotype coefficient, :math:`\hat\\beta_1`
        Wald  ``va.logreg.wald.se``    Double estimated standard error, :math:`\widehat{\mathrm{se}}`
        Wald  ``va.logreg.wald.zstat`` Double Wald :math:`z`-statistic, equal to :math:`\hat\\beta_1 / \widehat{\mathrm{se}}`
        Wald  ``va.logreg.wald.pval``  Double Wald test p-value testing :math:`\\beta_1 = 0`
        LRT   ``va.logreg.lrt.beta``   Double fit genotype coefficient, :math:`\hat\\beta_1`
        LRT   ``va.logreg.lrt.chi2``   Double likelihood ratio test statistic (deviance) testing :math:`\\beta_1 = 0`
        LRT   ``va.logreg.lrt.pval``   Double likelihood ratio test p-value
        Score ``va.logreg.score.chi2`` Double score statistic testing :math:`\\beta_1 = 0`
        Score ``va.logreg.score.pval`` Double score test p-value
        ===== ======================== ====== =====

        For the Wald and likelihood ratio tests, Hail fits the logistic model for each variant using Newton iteration and only emits the above annotations when the maximum likelihood estimate of the coefficients converges. To help diagnose convergence issues, Hail also emits three variant annotations which summarize the iterative fitting process:

        ========= =========================== ======= =====
        Test      Annotation                  Type    Value
        ========= =========================== ======= =====
        Wald, LRT ``va.logreg.fit.nIter``     Int     number of iterations until convergence, explosion, or reaching the max (25)
        Wald, LRT ``va.logreg.fit.converged`` Boolean true if iteration converged
        Wald, LRT ``va.logreg.fit.exploded``  Boolean true if iteration exploded
        ========= =========================== ======= =====

        We consider iteration to have converged when every coordinate of :math:`\\beta` changes by less than :math:`10^{-6}`. Up to 25 iterations are attempted; in testing we find 4 or 5 iterations nearly always suffice. Convergence may also fail due to explosion, which refers to low-level numerical linear algebra exceptions caused by manipulating ill-conditioned matrices. Explosion may result from (nearly) linearly dependent covariates or complete `separation <https://en.wikipedia.org/wiki/Separation_(statistics)>`_.

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

        Phenotype and covariate sample annotations may also be specified using `programmatic expressions <../expr_lang.html>`_ without identifiers, such as:

        .. code-block:: text

          if (sa.isFemale) sa.cov.age else (2 * sa.cov.age + 10)

        For Boolean covariate types, true is coded as 1 and false as 0. In particular, for the sample annotation ``sa.fam.isCase`` added by importing a FAM file with case-control phenotype, case is 1 and control is 0.

        Hail's logistic regression tests correspond to the ``b.wald``, ``b.lrt``, and ``b.score`` tests in `EPACTS <http://genome.sph.umich.edu/wiki/EPACTS#Single_Variant_Tests>`_. For each variant, Hail imputes missing genotypes as the mean of called genotypes, whereas EPACTS subsets to those samples with called genotypes. Hence, Hail and EPACTS results will currently only agree for variants with no missing genotypes.

        See `Recommended joint and meta-analysis strategies for case-control association testing of single low-count variants <http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4049324/>`_ for an empirical comparison of the logistic Wald, LRT, score, and Firth tests. The theoretical foundations of the Wald, likelihood ratio, and score tests may be found in Chapter 3 of Gesine Reinert's notes `Statistical Theory <http://www.stats.ox.ac.uk/~reinert/stattheory/theoryshort09.pdf>`_.

        :param str test: Statistical test, one of: wald, lrt, or score.
        :param str y: Response expression.  Must evaluate to Boolean or
            numeric with all values 0 or 1.
        :param covariates: list of covariate expressions
        :type covariates: list of str
        :param str root: Variant annotation path to store result of linear regression.

        :return: Dataset with logistic regression variant annotations.
        :rtype: :py:class:`.VariantDataset`

        """

        try:
            return VariantDataset(self.hc, env.hail.methods.LogisticRegression.apply(self._jvds, test, y,
                                                                                          jarray(
                                                                                              env.jvm.java.lang.String,
                                                                                              covariates), root))
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def mendel_errors(self, output, fam):
        """Find Mendel errors; count per variant, individual and nuclear
        family.

        **Implementation Details**

        This method finds all violations of Mendelian inheritance in each (dad,
        mom, kid) trio of samples.

        The following expression,

        >>> vds.mendel_errors('genomes', 'trios.fam')

        outputs four TSV files according to the `Plink mendel
        formats <https://www.cog-genomics.org/plink2/formats#mendel>`_:

        - ``genomes.mendel`` -- all mendel errors: FID KID CHR SNP CODE ERROR
        - ``genomes.fmendel`` -- error count per nuclear family: FID PAT MAT CHLD N NSNP
        - ``genomes.imendel`` -- error count per individual: FID IID N NSNP
        - ``genomes.lmendel`` -- error count per variant: CHR SNP N

        **FID**, **KID**, **PAT**, **MAT**, and **IID** refer to family, kid,
        dad, mom, and individual ID, respectively, with missing values set to
        ``0``.

        SNP denotes the variant identifier ``chr:pos:ref:alt``.

        N counts all errors, while NSNP only counts SNP errors (NSNP is not in Plink).

        CHLD is the number of children in a nuclear family.

        The CODE of each Mendel error is determined by the table below,
        extending the `Plink
        classification <https://www.cog-genomics.org/plink2/basic_stats#mendel>`_.

        Those individuals implicated by each code are in bold.

        The copy state of a locus with respect to a trio is defined as follows,
        where PAR is the pseudo-autosomal region (PAR).

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

        **Notes**

        This method only considers children with two parents and a defined sex.

        PAR is currently defined with respect to reference
        `GRCh37 <http://www.ncbi.nlm.nih.gov/projects/genome/assembly/grc/human/>`_:

        - X: 60001-2699520
        - X: 154931044-155260560
        - Y: 10001-2649520
        - Y: 59034050-59363566

        This method assumes all contigs apart from X and Y are fully autosomal;
        mitochondria, decoys, etc. are not given special treatment.

        :param str output: Output root filename.
        :param str fam: Path to .fam file.
        """

        pargs = ['mendelerrors', '-o', output, '-f', fam]
        self.hc._run_command(self, pargs)

    def pca(self, scores, loadings=None, eigenvalues=None, k=10, as_array=False):
        """Run Principal Component Analysis (PCA) on the matrix of genotypes.

        **Examples**

        Compute the top 10 principal component scores, stored as sample annotations ``sa.scores.PC1``, ..., ``sa.scores.PC10`` of type Double:

        >>> vds = (hc.read('data/example.vds')
        >>>  .pca('sa.scores'))

        Compute the top 5 principal component scores, loadings, and eigenvalues, stored as annotations ``sa.scores``, ``va.loadings``, and ``global.evals`` of type Array[Double]:

        >>> vds = (hc.read('data/example.vds')
        >>>  .pca('sa.scores', 'va.loadings', 'global.evals', 5, as_array=True))

        **Details**

        Hail supports principal component analysis (PCA) of genotype data, a now-standard procedure `Patterson, Price and Reich, 2006 <http://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.0020190>`_. This method expects a variant dataset with biallelic autosomal variants. Scores are computed and stored as sample annotations of type Struct by default; variant loadings and eigenvalues can optionally be computed and stored in variant and global annotations, respectively.

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

        pargs = ['pca', '--scores', scores, '-k', str(k)]
        if loadings:
            pargs.append('--loadings')
            pargs.append(loadings)
        if eigenvalues:
            pargs.append('--eigenvalues')
            pargs.append(eigenvalues)
        if as_array:
            pargs.append('--arrays')
        return self.hc._run_command(self, pargs)

    def persist(self, storage_level="MEMORY_AND_DISK"):
        """Persist the current dataset to memory and/or disk.

        **Examples**

        Persist the dataset to both memory and disk:

        >>> vds.persist()

        **Notes**

        The :py:meth:`~hail.VariantDataset.persist` and :py:meth:`~hail.VariantDataset.cache` commands allow you to store the current dataset on disk
        or in memory to avoid redundant computation and improve the performance of Hail pipelines.

        :py:meth:`~hail.VariantDataset.cache` is an alias for :func:`persist("MEMORY_ONLY") <hail.VariantDataset.persist>`.  Most users will want "MEMORY_AND_DISK".
        See the `Spark documentation <http://spark.apache.org/docs/latest/programming-guide.html#rdd-persistence>`_ for a more in-depth discussion of persisting data.

        :param storage_level: Storage level.  One of: NONE, DISK_ONLY,
            DISK_ONLY_2, MEMORY_ONLY, MEMORY_ONLY_2, MEMORY_ONLY_SER,
            MEMORY_ONLY_SER_2, MEMORY_AND_DISK, MEMORY_AND_DISK_2,
            MEMORY_AND_DISK_SER, MEMORY_AND_DISK_SER_2, OFF_HEAP

        """

        pargs = ['persist']
        if storage_level:
            pargs.append('-s')
            pargs.append(storage_level)
        self.hc._run_command(self, pargs)

    @property
    def global_schema(self):
        """
        Returns the signature of the global annotations contained in this VDS.

        >>> print(vds.global_schema)

        :rtype: :class:`.Type`
        """
        if self._global_schema is None:
            self._global_schema = Type._from_java(self._jvds.globalSignature())
        return self._global_schema

    @property
    def sample_schema(self):
        """
        Returns the signature of the sample annotations contained in this VDS.

        >>> print(vds.sample_schema)

        :rtype: :class:`.Type`
        """

        if self._sa_schema is None:
            self._sa_schema = Type._from_java(self._jvds.saSignature())
        return self._sa_schema

    @property
    def variant_schema(self):
        """
        Returns the signature of the variant annotations contained in this VDS.

        >>> print(vds.variant_schema)

        :rtype: :class:`.Type`
        """

        if self._va_schema is None:
            self._va_schema = Type._from_java(self._jvds.vaSignature())
        return self._va_schema

    def query_samples_typed(self, exprs):
        """Perform aggregation queries over samples and sample annotations, and returns python object(s) and types.

        **Example**

        >>> low_callrate_samples, t = vds.query_samples_typed(
        >>>     'samples.filter(s => sa.qc.callRate < 0.95).collect()')

        See :py:meth:`.query_samples` for more information.

        :param exprs: one or more query expressions
        :type exprs: str or list of str
        :rtype: (list, list of :class:`.Type`)
        """

        if not isinstance(exprs, list):
            exprs = [exprs]
        result_list = self._jvds.querySamples(jarray(env.jvm.java.lang.String, exprs))
        ptypes = [Type._from_java(x._2()) for x in result_list]
        annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in xrange(len(ptypes))]
        return annotations, ptypes

    def query_samples(self, exprs):
        """Perform aggregation queries over samples and sample annotations, and returns python object(s).

        **Example**

        >>> low_callrate_samples = vds.query_samples(
        >>>     'samples.filter(s => sa.qc.callRate < 0.95).collect()')

        **Details**

        This method evaluates Hail expressions over samples and sample
        annotations.  The ``exprs`` argument requires either a list of
        strings or a single string (which will be interpreted as a list
        with one element).  The method returns a list of results (which
        contains one element if the input parameter was a single str).

        The namespace of the expressions includes:

        - ``global``: global annotations
        - ``samples`` (*Aggregable[Sample]*): aggregable of :ref:`sample`

        Map and filter expressions on this aggregable have the additional
        namespace:

        - ``global``: global annotations
        - ``s``: sample
        - ``sa``: sample annotations

        :param exprs: one or more query expressions
        :type exprs: str or list of str

        :rtype: list
        """
        r, t = self.query_samples_typed(exprs)
        return r

    def query_variants_typed(self, exprs):
        """Perform aggregation queries over variants and variant annotations, and returns python objects and types.

        **Example**

        >>> lof_variant_count, t = vds.query_variants(
        >>>     'variants.filter(v => va.csq == "LOF").count()')

        See :py:meth:`.query_variants` for more information.

        :param exprs: one or more query expressions
        :type exprs: str or list of str

        :rtype: (list, list of :class:`.Type`)
        """


        if not isinstance(exprs, list):
            exprs = [exprs]
        result_list = self._jvds.queryVariants(jarray(env.jvm.java.lang.String, exprs))
        ptypes = [Type._from_java(x._2()) for x in result_list]
        annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in xrange(len(ptypes))]
        return annotations, ptypes

    def query_variants(self, exprs):
        """Perform aggregation queries over variants and variant annotations.

        **Example**

        >>> lof_variant_count = vds.query_variants(
        >>>     'variants.filter(v => va.csq == "LOF").count()')

        **Details**

        This method evaluates Hail expressions over variants and variant
        annotations.  The ``exprs`` argument requires either a list of
        strings or a single string (which will be interpreted as a list
        with one element).  The method returns a list of results (which
        contains one element if the input parameter was a single str).

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
        to execute multiple query methods.  This:

        >>> result1 = vds.query_variants('variants.count()')
        >>> result2 = vds.query_variants('variants.filter(variants.filter(v => v.altAllele.isSNP()).count()')

        will be nearly twice as slow as this:

        >>> exprs = ['variants.count()', 'variants.filter(variants.filter(v => v.altAllele.isSNP()).count()']
        >>> results = vds.query_variants(exprs)

        :param exprs: one or more query expressions
        :type exprs: str or list of str

        :rtype: list
        """

        r, t = self.query_variants_typed(exprs)
        return r

    def rename_samples(self, input):
        """Rename samples.

        **Example**


        >>> vds = (hc.read('data/example.vds')
        >>>  .rename_samples('data/sample.map'))

        **Details**

        The input file is a two-column, tab-separated file with no header. The first column is the current sample
        name, the second column is the new sample name.  Samples which do not
        appear in the first column will not be renamed.  Lines in the input that
        do not correspond to any sample in the current dataset will be ignored.

        :py:meth:`~hail.VariantDataset.export_samples` can be used to generate a template for renaming
        samples. For example, suppose you want to rename samples to remove
        spaces.  First, run:

        >>> (hc.read('data/example.vds')
        >>>  .export_samples('data/sample.map', 's.id, s.id'))

        Then edit *sample.map* to remove spaces from the sample names in the
        second column and run the example above. Renaming samples is fast so there is no need to save out the resulting dataset
        before performing analyses.

        :param str input: Input file.

        :return: Dataset with remapped sample IDs.
        :rtype: :class:`.VariantDataset`
        """

        pargs = ['renamesamples', '-i', input]
        return self.hc._run_command(self, pargs)

    def repartition(self, num_partitions, shuffle=True):
        """Increase or decrease the dataset sharding.  Can improve performance
        after large filters.

        **Examples**

        Force the number of partitions to be 5:

        >>> vds_repartitioned = vds.repartition(5)

        :param int num_partitions: Desired number of partitions.
        :param bool shuffle: If True, use shuffle to repartition.

        :return: Dataset with the number of partitions equal to at most ``num_partitions``
        :rtype: :class:`.VariantDataset`
        """

        pargs = ['repartition', '--partitions', str(num_partitions)]
        if not shuffle:
            pargs.append('--no-shuffle')
        return self.hc._run_command(self, pargs)

    def same(self, other):
        """Compare two datasets.

        :param other: dataset to compare against
        :type other: :class:`.VariantDataset`

        :rtype: bool
        """
        try:
            return self._jvds.same(other._jvds, 1e-6)
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def sample_qc(self):
        """Compute per-sample QC metrics.

        :return: Annotated dataset.
        :rtype: :class:`.VariantDataset`
        """

        pargs = ['sampleqc']
        return self.hc._run_command(self, pargs)

    def sparkinfo(self):
        """Displays the number of partitions and persistence level of the dataset."""

        self.hc._run_command(self, ['sparkinfo'])

    def split_multi(self, propagate_gq=False, keep_star_alleles=False):
        """Split multiallelic variants.

        **Examples**

        >>> (hc.import_vcf('data/sample.vcf')
        >>>  .split_multi()
        >>>  .write('data/split.vds'))

        **Implementation Details**

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

        >>> (hc.import_vcf('data/sample.vcf')
        >>>  .split_multi()
        >>>  .filter_variants_expr('va.info.AC[va.aIndex - 1] < 10', keep = False))

        VCFs split by Hail and exported to new VCFs may be
        incompatible with other tools, if action is not taken
        first. Since the "Number" of the arrays in split multiallelic
        sites no longer matches the structure on import ("A" for 1 per
        allele, for example), Hail will export these fields with
        number ".".

        If the desired output is one value per site, then it is
        possible to use annotatevariants expr to remap these
        values. Here is an example:

        >>> (hc.import_vcf('data/sample.vcf')
        >>>  .split_multi()
        >>>  .annotate_variants_expr('va.info.AC = va.info.AC[va.aIndex - 1]')
        >>>  .export_vcf('data/export.vcf'))

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

        :return: A dataset with split biallelic variants.
        :rtype: :py:class:`.VariantDataset`
        """

        pargs = ['splitmulti']
        if propagate_gq:
            pargs.append('--propagate-gq')
        if keep_star_alleles:
            pargs.append('--keep-star-alleles')
        return self.hc._run_command(self, pargs)

    def tdt(self, fam, root='va.tdt'):
        """Find transmitted and untransmitted variants; count per variant and
        nuclear family.

        **Examples**

        Compute TDT association results:

        >>> (hc.read("data/example.vds")
        >>>     .tdt("data/sample.fam")
        >>>     .export_variants("Variant = v, va.tdt.*"))

        **Implementation Details**

        The transmission disequilibrium test tracks the number of times the alternate allele is transmitted (t) or not transmitted (u) from a heterozgyous parent to an affected child under the null that the rate of such transmissions is 0.5.  For variants where transmission is guaranteed (i.e., the Y chromosome, mitochondria, and paternal chromosome X variants outside of the PAR), the test cannot be used.

        The TDT statistic is given by

        .. math::

            (t-u)^2 \over (t+u)

        and follows a 1 degree of freedom chi-squared distribution under the null hypothesis.


        The number of transmissions and untransmissions for each possible set of genotypes is determined from the table below.  The copy state of a locus with respect to a trio is defined as follows, where PAR is the pseudo-autosomal region (PAR).

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

        PAR is currently defined with respect to reference `GRCh37 <http://www.ncbi.nlm.nih.gov/projects/genome/assembly/grc/human/>`_:

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

        :param str fam: Path to FAM file.
        :param root: Variant annotation root to store TDT result.

        :return: A dataset with TDT association results added to variant annotations.
        :rtype: :py:class:`.VariantDataset`
        """

        pargs = ['tdt', '--fam', fam, '--root', root]
        return self.hc._run_command(self, pargs)

    def _typecheck(self):
        """Check if all sample, variant and global annotations are consistent with the schema."""

        pargs = ['typecheck']
        self.hc._run_command(self, pargs)

    def variant_qc(self):
        """Compute common variant statistics (quality control metrics).

        **Example**

        >>> vds = (hc.read('data/example.vds')
        >>>  .variant_qc())

        .. _variantqc_annotations:

        **Annotations**

        :py:meth:`~hail.VariantDataset.variant_qc` computes 16 variant statistics from the genotype data and stores the results as variant annotations that can be accessed with ``va.qc.<identifier>``:

        +---------------------------+--------+----------------------------------------------------+
        | Name                      | Type   | Description                                        |
        +===========================+========+====================================================+
        | ``callRate``              | Double | Fraction of samples with called genotypes          |
        +---------------------------+--------+----------------------------------------------------+
        | ``AF``                    | Double | Calculated minor allele frequency (q)              |
        +---------------------------+--------+----------------------------------------------------+
        | ``AC``                    | Int    | Count of alternate alleles                         |
        +---------------------------+--------+----------------------------------------------------+
        | ``rHeterozygosity``       | Double | Proportion of heterozygotes                        |
        +---------------------------+--------+----------------------------------------------------+
        | ``rHetHomVar``            | Double | Ratio of heterozygotes to homozygous alternates    |
        +---------------------------+--------+----------------------------------------------------+
        | ``rExpectedHetFrequency`` | Double | Expected rHeterozygosity based on HWE              |
        +---------------------------+--------+----------------------------------------------------+
        | ``pHWE``                  | Double | p-value from Hardy Weinberg Equilibrium null model |
        +---------------------------+--------+----------------------------------------------------+
        | ``nHomRef``               | Int    | Number of homozygous reference samples             |
        +---------------------------+--------+----------------------------------------------------+
        | ``nHet``                  | Int    | Number of heterozygous samples                     |
        +---------------------------+--------+----------------------------------------------------+
        | ``nHomVar``               | Int    | Number of homozygous alternate samples             |
        +---------------------------+--------+----------------------------------------------------+
        | ``nCalled``               | Int    | Sum of ``nHomRef``, ``nHet``, and ``nHomVar``      |
        +---------------------------+--------+----------------------------------------------------+
        | ``nNotCalled``            | Int    | Number of uncalled samples                         |
        +---------------------------+--------+----------------------------------------------------+
        | ``nNonRef``               | Int    | Sum of ``nHet`` and ``nHomVar``                    |
        +---------------------------+--------+----------------------------------------------------+
        | ``rHetHomVar``            | Double | Het/HomVar ratio across all samples                |
        +---------------------------+--------+----------------------------------------------------+
        | ``dpMean``                | Double | Depth mean across all samples                      |
        +---------------------------+--------+----------------------------------------------------+
        | ``dpStDev``               | Double | Depth standard deviation across all samples        |
        +---------------------------+--------+----------------------------------------------------+

        Missing values ``NA`` may result (for example, due to division by zero) and are handled properly in filtering and written as "NA" in export modules. The empirical standard deviation is computed with zero degrees of freedom.

        :return: Annotated dataset with new variant QC annotations.
        :rtype: :py:class:`.VariantDataset`
        """

        pargs = ['variantqc']
        return self.hc._run_command(self, pargs)

    def vep(self, config, block_size=1000, root='va.vep', force=False, csq=False):
        """Annotate variants with VEP.

        :py:meth:`~hail.VariantDataset.vep` runs `Variant Effect Predictor <http://www.ensembl.org/info/docs/tools/vep/index.html>`_ with
        the `LOFTEE plugin <https://github.com/konradjk/loftee>`_
        on the current dataset and adds the result as a variant annotation.

        If the variant annotation path defined by ``root`` already exists and its schema matches the VEP schema, then
        Hail only runs VEP for variants for which the annotation is missing.

        **Examples**

        Add VEP annotations to the dataset:

        >>> vds_annotated = vds.vep("data/vep.properties")

        **Configuration**

        :py:meth:`~hail.VariantDataset.vep` needs a configuration file to tell it how to run
        VEP. The format is a `.properties file <https://en.wikipedia.org/wiki/.properties>`_.
        Roughly, each line defines a property as a key-value pair of the form `key = value`. `vep` supports the following properties:

        - **hail.vep.perl** -- Location of Perl. Optional, default: perl.
        - **hail.vep.perl5lib** -- Value for the PERL5LIB environment variable when invoking VEP. Optional, by default PERL5LIB is not set.
        - **hail.vep.path** -- Value of the PATH environment variable when invoking VEP.  Optional, by default PATH is not set.
        - **hail.vep.location** -- Location of the VEP Perl script.  Required.
        - **hail.vep.cache_dir** -- Location of the VEP cache dir, passed to VEP with the `--dir` option.  Required.
        - **hail.vep.lof.human_ancestor** -- Location of the human ancestor file for the LOFTEE plugin.  Required.
        - **hail.vep.lof.conservation_file** -- Location of the conservation file for the LOFTEE plugin.  Required.

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
            --fasta <hail.vep.cache_dir>/homo_sapiens/81_GRCh37/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa
            --minimal
            --assembly GRCh37
            --plugin LoF,human_ancestor_fa:$<hail.vep.lof.human_ancestor>,filter_position:0.05,min_intron_size:15,conservation_file:<hail.vep.lof.conservation_file>
            -o STDOUT

        **Annotations**

        Annotations with the following schema are placed in the location specified by ``root``.
        The schema can be confirmed with :py:meth:`~hail.VariantDataset.print_schema`.

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
                hgnc_id: Int,
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
        :param bool force: If True, force VEP annotation from scratch.
        :param bool csq: If True, annotates VCF CSQ field as a String.
            If False, annotates with the full nested struct schema

        :return: An annotated with variant annotations from VEP.
        :rtype: :py:class:`.VariantDataset`
        """

        pargs = ['vep', '--config', config]
        if block_size:
            pargs.append('--block-size')
            pargs.append(block_size)
        if root:
            pargs.append('--root')
            pargs.append(root)
        if force:
            pargs.append('--force')
        if csq:
            pargs.append('--csq')
        return self.hc._run_command(self, pargs)

    def variants_keytable(self):
        """Convert variants and variant annotations to a KeyTable.

        The resulting KeyTable has schema:

        .. code-block:: text

          Struct {
            v: Variant
            va: variant annotations
          }

        with a single key ``v``.

        :return: A key table with variants and variant annotations.
        :rtype: :class:`.KeyTable`
        """

        try:
            return KeyTable(self.hc, self._jvds.variantsKT())
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def samples_keytable(self):
        """Convert samples and sample annotations to KeyTable.

        The resulting KeyTable has schema:

        .. code-block:: text

          Struct {
            s: Sample
            sa: sample annotations
          }

        with a single key ``s``.

        :return: A key table with samples and sample annotations.
        :rtype: :class:`.KeyTable`
        """

        try:
            return KeyTable(self.hc, self._jvds.samplesKT())
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def make_keytable(self, variant_condition, genotype_condition, key_names):
        """Make a KeyTable with one row per variant.

        Per sample field names in the result are formed by concatening the
        sample ID with the ``genotype_condition`` left hand side with dot (.).
        If the left hand side is empty::

          `` = expr

        then the dot (.) is ommited.

        **Example**

        Consider a ``VariantDataset`` ``vds`` with 2 variants and 3 samples:

        .. code-block:: text

          Variant	FORMAT	A	B	C
          1:1:A:T	GT:GQ	0/1:99	./.	0/0:99
          1:2:G:C	GT:GQ	0/1:89	0/1:99	1/1:93

        Then::

          >>> vds = hc.import_vcf('data/sample.vcf')
          >>> vds.make_keytable('v = v', 'gt = g.gt', 'gq = g.gq', [])

        returns a ``KeyTable`` with schema

        .. code-block:: text

            v: Variant
            A.gt: Int
            B.gt: Int
            C.gt: Int
            A.gq: Int
            B.gq: Int
            C.gq: Int

        in particular, the values would be

        .. code-block:: text

            v	A.gt	B.gt	C.gt	A.gq	B.gq	C.gq
            1:1:A:T	1	NA	0	99	NA	99
            1:2:G:C	1	1	2	89	99	93

        :param variant_condition: Variant annotation expressions.
        :type variant_condition: str or list of str
        :param genotype_condition: Genotype annotation expressions.
        :type genotype_condition: str or list of str
        :param key_names: list of key columns
        :type key_names: list of str

        :rtype: :class:`.KeyTable`
        """

        if isinstance(variant_condition, list):
            variant_condition = ','.join(variant_condition)
        if isinstance(genotype_condition, list):
            genotype_condition = ','.join(genotype_condition)

        jkt = (scala_package_object(env.hail.driver)
               .makeKT(self._jvds, variant_condition, genotype_condition,
                       jarray(env.jvm.java.lang.String, key_names)))
        return KeyTable(self.hc, jkt)
