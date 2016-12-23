from pyhail.java import scala_package_object, jarray
from pyhail.keytable import KeyTable
from pyhail.utils import TextTableConfig

from py4j.protocol import Py4JJavaError

class VariantDataset(object):
    def __init__(self, hc, jvds):
        self.hc = hc
        self.jvds = jvds

    def _raise_py4j_exception(self, e):
        self.hc._raise_py4j_exception(e)

    def sample_ids(self):
        """Return sampleIDs.

        :return: List of sample IDs.

        :rtype: list of str

        """
        try:
            return list(self.jvds.sampleIdsAsArray())
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def num_partitions(self):
        """Number of RDD partitions.

        :rtype: int

        """
        try:
            return self.jvds.nPartitions()
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def num_samples(self):
        """Number of samples.

        :rtype: int

        """
        try:
            return self.jvds.nSamples()
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def num_variants(self):
        """Number of variants.

        :rtype: long

        """
        try:
            return self.jvds.nVariants()
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def was_split(self):
        """Multiallelic variants have been split into multiple biallelic variants.

        Result is True if :py:meth:`~pyhail.VariantDataset.split_multi` has been called on this dataset
        or the dataset was imported with :py:meth:`~pyhail.HailContext.import_plink`, :py:meth:`~pyhail.HailContext.import_gen`,
        or :py:meth:`~pyhail.HailContext.import_bgen`.

        :rtype: bool

        """
        try:
            return self.jvds.wasSplit()
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def is_dosage(self):
        """Genotype probabilities are dosages.

        The result of ``is_dosage()`` will be True if the dataset was imported with :py:meth:`~pyhail.HailContext.import_gen` or
        :py:meth:`~pyhail.HailContext.import_bgen`.

        :rtype: bool

        """
        try:
            return self.jvds.isDosage()
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def file_version(self):
        """File version of dataset.

        :rtype int

        """
        try:
            return self.jvds.fileVersion()
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

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
            return KeyTable(self.hc, self.jvds.aggregateByKey(key_code, agg_code))
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def aggregate_intervals(self, input, condition, output):
        """Aggregate over intervals and export.

        :param str input: Input interval list file.

        :param str condition: Aggregation expression.

        :param str output: Output file.

        """

        pargs = ['aggregateintervals', '-i', input, '-c', condition, '-o', output]
        return self.hc.run_command(self, pargs)

    def annotate_alleles_expr(self, condition, propagate_gq=False):
        """Annotate alleles with expression.

        :param condition: Annotation expression.
        :type condition: str or list of str
        :param bool propagate_gq: Propagate GQ instead of computing from (split) PL.

        """
        if isinstance(condition, list):
            condition = ','.join(condition)
        pargs = ['annotatealleles', 'expr', '-c', condition]
        if propagate_gq:
            pargs.append('--propagate-gq')
        return self.hc.run_command(self, pargs)

    def annotate_global_expr_by_variant(self, condition):
        """Update the global annotations with expression with aggregation over
        variants.

        :param condition: Annotation expression.
        :type condition: str or list of str

        """

        if isinstance(condition, list):
            condition = ','.join(condition)
        pargs = ['annotateglobal', 'exprbyvariant', '-c', condition]
        return self.hc.run_command(self, pargs)

    def annotate_global_expr_by_sample(self, condition):
        """Update the global annotations with expression with aggregation over
        samples.

        :param str condition: Annotation expression.
        :type condition: str or list of str

        """

        if isinstance(condition, list):
            condition = ','.join(condition)
        pargs = ['annotateglobal', 'exprbysample', '-c', condition]
        return self.hc.run_command(self, pargs)

    def annotate_global_list(self, input, root, as_set=False):
        """Load text file into global annotations as Array[String] or
        Set[String].

        **Examples**

        Let's add a gene list into global annotations::

          $ cat genes.txt
          SCN2A
          SONIC-HEDGEHOG
          PRNP
          ALDH4A1
          LEP
          OSM
          TSC1
          TSC2

        To annotate with this list as type ``Array[String]``:

        >>> vds = (hc.read('example.vds')
        >>>  .annotate_global_list('genes.txt', 'global.genes'))

        To see the resulting global annotation schema and values:

        >>> vds.print_schema(print_global=True)
        >>>  .show_globals()

        This prints::

          Global annotation schema:
          global: Struct {
              genes: Array[String]
          }

          hail: info: Global annotations: `global' =
          {
            "genes" : [ "SCN2A", "SONIC-HEDGEHOG", "PRNP", "ALDH4A1", "LEP", "OSM", "TSC1", "TSC2" ]
          }

        Now suppose we already have a variant annotation ``va.gene`` which specifies the gene in which the variant resides.
        To filter to those variants in genes listed in ``genes.txt``, let's annotate as type ``Set[String]`` instead:

        >>> vds2 = (hc.read('example.vds')
        >>>  .annotate_global_list('genes.txt', 'global.genes', as_set=True))
        >>>  .filter_variants_expr('global.genes.contains(va.gene)')

        :param str input: Input text file.

        :param str root: Global annotation path to store text file.

        :param bool as_set: If True, load text file as Set[String],
            otherwise, load as Array[String].
        """

        pargs = ['annotateglobal', 'list', '-i', input, '-r', root]
        if as_set:
            pargs.append('--as-set')
        return self.hc.run_command(self, pargs)

    def annotate_global_table(self, input, root, config=None):
        """Load delimited text file (text table) into global annotations as
        Array[Struct].

        :param str input: Input text file.

        :param str root: Global annotation path to store text table.

        :param config: Configuration options for importing text files
        :type config: :class:`.TextTableConfig` or None

        """

        pargs = ['annotateglobal', 'table', '-i', input, '-r', root]

        if not config:
            config = TextTableConfig()

        pargs.extend(config.as_pargs())

        return self.hc.run_command(self, pargs)

    def annotate_samples_expr(self, condition):
        """Annotate samples with expression.

        :param condition: Annotation expression.
        :type condition: str or list of str

        """

        if isinstance(condition, list):
            condition = ','.join(condition)
        pargs = ['annotatesamples', 'expr', '-c', condition]
        return self.hc.run_command(self, pargs)

    def annotate_samples_fam(self, input, quantpheno=False, delimiter='\\\\s+', root='sa.fam', missing='NA'):
        """Import PLINK .fam file into sample annotations.

        :param str input: Path to .fam file.

        :param str root: Sample annotation path to store .fam file.

        :param bool quantpheno: If True, .fam phenotype is interpreted as quantitative.

        :param str delimiter: .fam file field delimiter regex.

        :param str missing: The string used to denote missing values.
            For case-control, 0, -9, and non-numeric are also treated
            as missing.

        """

        pargs = ['annotatesamples', 'fam', '-i', input, '--root', root, '--missing', missing]
        if quantpheno:
            pargs.append('--quantpheno')
        if delimiter:
            pargs.append('--delimiter')
            pargs.append(delimiter)
        return self.hc.run_command(self, pargs)

    def annotate_samples_list(self, input, root):
        """Annotate samples with a Boolean indicating presence/absence in a list of samples in a text file.

        **Examples**

        We have a file with a list of samples::

          $ cat ~/batch1.txt
          PT-1234
          PT-1235
          PT-1236
          PT-1237
          PT-1238
          PT-1239

        The command

        >>> vds1 = (hc.read('example.vds')
        >>>  .annotate_samples_list('batch1.txt','sa.inBatch1')

        will read the file `batch1.txt` and annotate samples with true if in the list and false otherwise, resulting in the following schema::

          Sample annotations:
          sa: Struct {
              inBatch1: Boolean
          }

        :param str input: Sample list file.

        :param str root: Sample annotation path to store Boolean.

        """

        pargs = ['annotatesamples', 'list', '-i', input, '-r', root]
        return self.hc.run_command(self, pargs)

    def annotate_samples_table(self, input, sample_expr, root=None, code=None, config=None):
        """Annotate samples with delimited text file (text table).

        **Examples**

        Let's import annotations from a .tsv file with phenotypes and age::

          $ cat samples1.tsv
          Sample  Phenotype1   Phenotype2  Age
          PT-1234 24.15        ADHD        24
          PT-1235 31.01        ADHD        25
          PT-1236 25.95        Control     19
          PT-1237 26.80        Control     42
          PT-1238 NA           ADHD        89
          PT-1239 27.53        Control     55

        To load this file into Hail with type imputation:

        >>> vds1 = (hc.read('example1.vds')
        >>>  .annotate_samples_table('samples1.tsv', 'Sample', root='sa.phenotypes'))

        This will read the file and produce annotations of the following schema::

          Sample annotations:
          sa: sa.<identifier>
              phenotypes: sa.phenotypes.<identifier>
                  Phenotype1: String
                  Phenotype2: String
                  Age: String

        To load this file into Hail with type imputation:

        >>> conf = pyhail.TextTableConfig(impute=True)
        >>> vds1 = (hc.read('example1.vds')
        >>>  .annotate_samples_table('samples1.tsv', 'Sample', root='sa.phenotypes', config=conf))

        This will read the file and produce annotations of the following schema::

          Sample annotations:
          sa: sa.<identifier>
              phenotypes: sa.phenotypes.<identifier>
                  Phenotype1: Double
                  Phenotype2: String
                  Age: Int

        Let's import annotations from a .csv file with missing data and special characters::

          $ cat samples2.tsv
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

         - Escape the ``PT-ID`` column with backticks in the ``sample-expr`` argument because it contains a dash
         - Pass the non-default delimiter ``,``
         - Pass the non-default missing value ``.``
         - Add the only useful column using ``code`` rather than ``root`` parameter.

        >>> conf = pyhail.TextTableConfig(delimiter=',', missing='.')
        >>> vds2 = (hc.read('example2.vds')
        >>>  .annotate_samples_table('samples2.tsv', '`PT-ID`', code='sa.batch = table.Batch', config=conf))

        Let's import annotations from a file with no header and sample IDs that need to be transformed. Suppose the vds sample IDs are of the form ``NA#####``. This file has no header line, and the sample ID is hidden in a field with other information::

          $ cat samples3.tsv
          1kg_NA12345     female
          1kg_NA12346     male
          1kg_NA12348     female
          pgc_NA23456     female
          pgc_NA23415     male
          pgc_NA23418     male

        Let's import it:

        >>> conf = pyhail.TextTableConfig(noheader=True)
        >>> vds3 = (hc.read('example3.vds')
        >>>  .annotate_samples_table('samples3.tsv', '_0.split("_")[1]', code='sa.sex = table._1, sa.batch = table._0.split("_")[0]', config=conf))

        **Common uses for the `code` argument**

        Don't generate a full struct in a table with only one annotation column::

          code='sa.annot = table._1'

        Put annotations on the top level under `sa`::

          code='sa = merge(sa, table)'

        Load only specific annotations from the table::

          code='sa.annotations = select(table, toKeep1, toKeep2, toKeep3)'

        The above is equivalent to::

          code='sa.annotations.toKeep1 = table.toKeep1,
                sa.annotations.toKeep2 = table.toKeep2,
                sa.annotations.toKeep3 = table.toKeep3'

        **Notes**

        The generality of this module allows it to load delimited text files, json, or a mixture of the two.

        **Using the ``sample-expr`` argument**

        This argument tells Hail how to get a sample ID out of your table. Each column in the table is exposed to the Hail expr language. Possibilities include ``Sample`` (if your sample id is in a column called 'Sample'), ``_2`` (if your sample ID is the 3rd column of a table with no header), or something more complicated like ``'if ("PGC" ~ ID1) ID1 else ID2'``.  All that matters is that this expr results in a string.  If the expr evaluates to missing, it will not be mapped to any VDS samples.

        **Using the ``root`` and ``code`` arguments**

        This module requires exactly one of these two arguments to tell Hail how to insert the table into the sample annotation schema.

        The ``root`` argument is the simpler of these two, and simply packages up all table annotations as a ``Struct`` and drops it at the given ``root`` location.  If your table has columns ``Sample``, ``Sex``, and ``Batch``, then ``root='sa.metadata'`` creates the struct ``{Sample, Sex, Batch}`` at ``sa.metadata``, which gives you access to the paths ``sa.metadata.Sample``, ``sa.metadata.Sex``, and ``sa.metadata.Batch``.

        The ``code`` argument expects an annotation expression with access to ``sa`` (the sample annotations in the VDS) and ``table`` (a struct with all the columns in the table).  ``root='sa.anno'`` is equivalent to ``code='sa.anno = table'``.

        :param str input: Path to delimited text file.

        :param str sample_expr: Expression for sample id (key).

        :param str root: Sample annotation path to store text table.

        :param str code: Annotation expression.

        :param config: Configuration options for importing text files
        :type config: :class:`.TextTableConfig` or None

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

        pargs.extend(config.as_pargs())

        return self.hc.run_command(self, pargs)

    def annotate_samples_vds(self, right, root=None, code=None):
        """Annotate samples with sample annotations from .vds file.

        :param VariantDataset right: VariantDataset to annotate with.

        :param str root: Sample annotation path to add sample annotations.

        :param str code: Annotation expression.

        """

        return VariantDataset(
            self.hc,
            self.hc.jvm.org.broadinstitute.hail.driver.AnnotateSamplesVDS.annotate(
                self.jvds, right.jvds, code, root))

    def annotate_variants_bed(self, input, root, all=False):
        """Annotate variants with a .bed file.

        **Examples**

        First consider a standard BED file::

          $ cat file1.bed
          track name="BedTest"
          20    1          14000000
          20    17000000   18000000

        In order to annotate with this file, the command should appear as::

        >>> vds1 = (hc.read('example.vds')
        >>>  .annotate_variants_bed('file1.bed', 'va.cnvRegion'))

        This file format produces a Boolean annotation::

          Variant annotations:
          va: va.<identifier>
              <lots of other stuff here>
              cnvRegion: Boolean

        Next consider a bed file with extra header information::

          $ cat file2.bed
          browser position 20:1-18000000
          browser hide all
          track name="BedTest"
          itemRgb="On"
          20    1          14000000  cnv1
          20    17000000   18000000  cnv2

        This file has a more complicated header, but that does not affect Hail's parsing because the header is always skipped (Hail is not a genome browser).  However, it also has a fourth column, so this column will be parsed as a string.  The command line should follow the same format::

        >>> vds2 = (hc.read('example.vds')
        >>>  .annotate_variants_bed('file2.bed', 'va.cnvRegion'))

        The schema will reflect that this annotation is read as a string::

          Variant annotations:
          va: va.<identifier>
              <lots of other stuff here>
              cnvRegion: String

        **Implementation Details**

        `UCSC bed files <https://genome.ucsc.edu/FAQ/FAQformat.html#format1>`_ can have up to 12 fields, but Hail will only ever look at the first four.  The first three fields are required (``chrom``, ``chromStart``, and ``chromEnd``).  If a fourth column is found, Hail will parse this field as a string and load it into the specified annotation path.  If the bed file has only three columns, Hail will assign each variant a Boolean annotation based on whether that variant was a member of any interval.

        If the ``all`` parameter is set to ``True`` and a fourth column is present, the annotation will be the set (possibly empty) of fourth column strings as a ``Set[String]`` for all intervals that overlap the given variant.

        **Note:** UCSC BED files are 0-indexed, which means that the line "5  100  105" will include the loci `5:99, 5:100, 5:101, 5:102, 5:103`.  The last locus included in this interval is two smaller than the listed end!

        :param str input: Path to .bed file.

        :param str root: Variant annotation path to store annotation.

        :param bool all: Store values from all overlapping intervals as a set.

        """

        pargs = ['annotatevariants', 'bed', '-i', input, '--root', root]
        if all:
            pargs.append('--all')
        return self.hc.run_command(self, pargs)

    def annotate_variants_expr(self, condition):
        """Annotate variants with expression.

        :param str condition: Annotation expression.

        """
        if isinstance(condition, list):
            condition = ','.join(condition)
        pargs = ['annotatevariants', 'expr', '-c', condition]
        return self.hc.run_command(self, pargs)

    def annotate_variants_intervals(self, input, root, all=False):
        """Annotate variants from an interval list file.

        :param str input: Path to .interval_list.

        :param str root: Variant annotation path to store annotation.

        :param bool all: If true, store values from all overlapping
            intervals as a set.

        """
        pargs = ['annotatevariants', 'intervals', '-i', input, '--root', root]
        if all:
            pargs.append('--all')
        return self.hc.run_command(self, pargs)

    def annotate_variants_loci(self, path, locus_expr, root=None, code=None, config=None):
        """Annotate variants from an delimited text file (text table) indexed
        by loci.

        :param str path: Path to delimited text file.

        :param str locus_expr: Expression for locus (key).

        :param str root: Variant annotation path to store annotation.

        :param str code: Annotation expression.

        :param config: Configuration options for importing text files
        :type config: :class:`.TextTableConfig` or None

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

        pargs.extend(config.as_pargs())

        if isinstance(path, str):
            pargs.append(path)
        else:
            for p in path:
                pargs.append(p)

        return self.hc.run_command(self, pargs)

    def annotate_variants_table(self, path, variant_expr, root=None, code=None, config=None):
        """Annotate variant with delimited text file (text table).

        :param path: Path to delimited text files.
        :type path: str or list of str

        :param str variant_expr: Expression for Variant (key).

        :param str root: Variant annotation path to store text table.

        :param str code: Annotation expression.

        :param config: Configuration options for importing text files
        :type config: :class:`.TextTableConfig` or None

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

        pargs.extend(config.as_pargs())

        if isinstance(path, str):
            pargs.append(path)
        else:
            for p in path:
                pargs.append(p)

        return self.hc.run_command(self, pargs)

    def annotate_variants_vds(self, other, code=None, root=None):
        """Annotate variants with variant annotations from .vds file.

        :param VariantDataset other: VariantDataset to annotate with.

        :param str root: Sample annotation path to add variant annotations.

        :param str code: Annotation expression.

        """

        return VariantDataset(
            self.hc,
            self.hc.jvm.org.broadinstitute.hail.driver.AnnotateVariantsVDS.annotate(
                self.jvds, other.jvds, code, root))

    def cache(self):
        """Cache in memory.  cache is the same as persist("MEMORY_ONLY")."""

        pargs = ['cache']
        return self.hc.run_command(self, pargs)

    def concordance(self, right):
        """Calculate call concordance with right.  Performs inner join on
        variants, outer join on samples.

        :return: Returns a pair of VariantDatasets with the sample and
            variant concordance, respectively.

        :rtype: (VariantDataset, VariantData)

        """

        result = self.hc.jvm.org.broadinstitute.hail.driver.Concordance.calculate(
            self.jvds, right.jvds)
        return (VariantDataset(self.hc, result._1()),
                VariantDataset(self.hc, result._2()))

    def count(self, genotypes=False):
        """Return number of samples, varaints and genotypes.

        :param bool genotypes: If True, return number of called
            genotypes and genotype call rate.

        """

        try:
            return (scala_package_object(self.hc.jvm.org.broadinstitute.hail.driver)
                    .count(self.jvds, genotypes)
                    .toJavaMap())
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def deduplicate(self):
        """Remove duplicate variants."""

        pargs = ['deduplicate']
        return self.hc.run_command(self, pargs)

    def downsample_variants(self, keep):
        """Downsample variants.

        :param int keep: (Expected) number of variants to keep.

        """

        pargs = ['downsamplevariants', '--keep', str(keep)]
        return self.hc.run_command(self, pargs)

    def export_gen(self, output):
        """Export dataset as .gen file.

        :param str output: Output file base.  Will write .gen and .sample files.

        """

        pargs = ['exportgen', '--output', output]
        return self.hc.run_command(self, pargs)

    def export_genotypes(self, output, condition, types=None, export_ref=False, export_missing=False):
        """Export genotype-level information to delimited text file.

        **Examples**

        To export genotype information with identifiers that form the header:

        >>> (hc.read('example.vds')
        >>>  .export_genotypes('genotypes.tsv', 'SAMPLE=s, VARIANT=v, GQ=g.gq, DP=g.dp, ANNO1=va.anno1, ANNO2=va.anno2'))

        To export the same information without identifiers, resulting in a file with no header:

        >>> (hc.read('example.vds')
        >>>  .export_genotypes('genotypes.tsv', 's, v, s.id, g.dp, va.anno1, va.anno2')

        **Notes**

        ``export_genotypes`` outputs one line per cell (genotype) in the data set, though HomRef and missing genotypes are not output by default. Use the ``export_ref`` and ``export_missing`` parameters to force export of HomRef and missing genotypes, respectively.

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
        return self.hc.run_command(self, pargs)

    def export_plink(self, output, fam_expr = 'id = s.id'):
        """Export as PLINK .bed/.bim/.fam

        :param str output: Output file base.  Will write .bed, .bim and .fam files.

        """

        pargs = ['exportplink', '--output', output, '--fam-expr', fam_expr]
        return self.hc.run_command(self, pargs)

    def export_samples(self, output, condition, types=None):
        """Export sample information to delimited text file.

        :param str output: Output file.

        :param str condition: Annotation expression for values to export.

        :param types: Path to write types of exported values.
        :type types: str or None

        """

        pargs = ['exportsamples', '--output', output, '-c', condition]
        if types:
            pargs.append('--types')
            pargs.append(types)
        return self.hc.run_command(self, pargs)

    def export_variants(self, output, condition, types=None):
        """Export variant information to delimited text file.

        :param str output: Output file.

        :param str condition: Annotation expression for values to export.

        :param types: Path to write types of exported values.
        :type types: str or None

        """

        pargs = ['exportvariants', '--output', output, '-c', condition]
        if types:
            pargs.append('--types')
            pargs.append(types)
        return self.hc.run_command(self, pargs)

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
        return self.hc.run_command(self, pargs)

    def export_variants_solr(self, variant_condition, genotype_condition,
                             solr_url=None,
                             solr_cloud_collection=None,
                             zookeeper_host=None,
                             drop=False,
                             num_shards=1,
                             export_missing=False,
                             export_ref=False,
                             block_size=100):
        """Export variant information to Cassandra."""

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
        return self.hc.run_command(self, pargs)

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
        return self.hc.run_command(self, pargs)

    def write(self, output, overwrite=False):
        """Write as .vds file.

        :param str output: Path of .vds file to write.

        :param bool overwrite: If True, overwrite any existing .vds file.

        """

        pargs = ['write', '-o', output]
        if overwrite:
            pargs.append('--overwrite')
        return self.hc.run_command(self, pargs)

    def filter_alleles(self, condition, annotation=None, subset=True, keep=True, filter_altered_genotypes=False):
        """
        Filter a user-defined set of alternate alleles for each variant.
        If all of a variant's alternate alleles are filtered, the variant itself is filtered.
        The condition expression is evaluated for each alternate allele.

        :param condition: Filter expression involving v (variant), va (variant annotations), and aIndex (allele index)
        :param annotation: Annotation modifying expression involving v (new variant), va (old variant annotations),
            and aIndices (maps from new to old indices) (default: "va = va")
        :param bool subset: If true, subsets the PL and AD, otherwise downcodes the PL and AD.
            Genotype and GQ are set based on the resulting PLs.
        :param bool keep: Keep variants matching condition
        :param bool filter_altered_genotypes: If set, any genotype call that would change due to filtering an allele
            would be set to missing instead.

        """

        pargs = ['filteralleles',
                 '--keep' if keep else '--remove',
                 '--subset' if subset else '--downcode',
                 '-c', condition]
        if annotation:
            pargs.extend(['-a', annotation])
        if filter_altered_genotypes:
            pargs.append('--filterAlteredGenotypes')
        return self.hc.run_command(self, pargs)

    def filter_genotypes(self, condition, keep=True):
        """Filter variants based on expression.

        :param str condition: Expression for filter condition.

        """

        pargs = ['filtergenotypes',
                 '--keep' if keep else '--remove',
                 '-c', condition]
        return self.hc.run_command(self, pargs)

    def filter_multi(self):
        """Filter out multi-allelic sites.

        Returns a VariantDataset with split = True.

        """

        pargs = ['filtermulti']
        return self.hc.run_command(self, pargs)

    def filter_samples_all(self):
        """Discard all samples (and genotypes)."""

        pargs = ['filtersamples', 'all']
        return self.hc.run_command(self, pargs)

    def filter_samples_expr(self, condition, keep=True):
        """Filter samples based on expression.

        :param condition: Expression for filter condition.
        :type condition: str or list of str

        """

        if isinstance(condition, list):
            condition = ','.join(condition)
        pargs = ['filtersamples', 'expr',
                 '--keep' if keep else '--remove',
                 '-c', condition]
        return self.hc.run_command(self, pargs)

    def filter_samples_list(self, input, keep=True):
        """Filter samples with a sample list file.

        :param str input: Path to sample list file.

        """

        pargs = ['filtersamples', 'list',
                 '--keep' if keep else '--remove',
                 '-i', input]
        return self.hc.run_command(self, pargs)

    def filter_variants_all(self):
        """Discard all variants, variant annotations and genotypes."""

        pargs = ['filtervariants', 'all']
        return self.hc.run_command(self, pargs)

    def filter_variants_expr(self, condition, keep=True):
        """Filter variants based on expression.

        :param condition: Expression for filter condition.
        :type condition: str or list of str

        """

        if isinstance(condition, list):
            condition = ','.join(condition)
        pargs = ['filtervariants', 'expr',
                 '--keep' if keep else '--remove',
                 '-c', condition]
        return self.hc.run_command(self, pargs)

    def filter_variants_intervals(self, input, keep=True):
        """Filter variants with an .interval_list file.

        :param str input: Path to .interval_list file.

        """

        pargs = ['filtervariants', 'intervals',
                 '--keep' if keep else '--remove',
                 '-i', input]
        return self.hc.run_command(self, pargs)

    def filter_variants_list(self, input, keep=True):
        """Filter variants with a list of variants.

        :param str input: Path to variant list file.

        """

        pargs = ['filtervariants', 'list',
                 '--keep' if keep else '--remove',
                 '-i', input]
        return self.hc.run_command(self, pargs)

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
        return self.hc.run_command(self, pargs)

    def hardcalls(self):
        """Drop all genotype fields except the GT field."""

        pargs = ['hardcalls']
        return self.hc.run_command(self, pargs)

    def ibd(self, output, maf=None, unbounded=False, min=None, max=None):
        """Compute matrix of identity-by-descent estimations.

        **Examples**

        Suppose we have a variant annotation ``va.panel_maf`` of allele
        frequencies computed from a reference panel. Then

        >>> (hc.read('example.vds')
        >>>  .ibd('ibd.tsv', maf='va.panel_maf', min=0.2, max=0.9)

        outputs to ``ibd.tsv`` only those sample pairs with ``pi_hat`` between 0.2 and 0.9 inclusive, using the the provided expression for the minor allele frequency.

        To writes the full IBD matrix to `ibd.tsv` with the minor allele frequency computed from the dataset:

        >>> (hc.read('example.vds')
        >>>  .ibd('ibd.tsv'

        **Notes**

        The implementation is based on the IBD algorithm described in the `PLINK paper <http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1950838>`_.

        ``ibd`` requires the dataset to be bi-allelic (otherwise run ``split_multi``). ``ibd`` does not perform LD
        pruning though linkage disequilibrium may adversely impact the result.

        Conceptually, the output is a symmetric, sample-by-sample matrix. The output .tsv has the following form::

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
        return self.hc.run_command(self, pargs)

    def impute_sex(self, maf_threshold=0.0, include_par=False, female_threshold=0.2, male_threshold=0.8, pop_freq=None):
        """Impute sex of samples by calculating inbreeding coefficient on the
        X chromosome.

        :param float maf_threshold: Minimum minor allele frequency threshold.

        :param bool include_par: Include pseudoautosomal regions.

        :param float female_threshold: Samples are called females if F < femaleThreshold

        :param float male_threshold: Samples are called males if F > maleThreshold

        :param str pop_freq: Variant annotation for estimate of MAF.
            If None, MAF will be computed.

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
        return self.hc.run_command(self, pargs)

    def join(self, right):
        """Join datasets, inner join on variants, concatenate samples, variant
        and global annotations from self.

        """
        try:
            return VariantDataset(self.hc, self.hc.jvm.org.broadinstitute.hail.driver.Join.join(self.jvds, right.jvds))
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def linreg(self, y, covariates='', root='va.linreg', minac=1, minaf=None):
        """Test each variant for association using the linear regression
        model.

        :param str y: Response sample annotation.

        :param str covariates: Covariant sample annotations, comma separated.

        :param str root: Variant annotation path to store result of linear regression.

        :param float minac: Minimum alternate allele count.

        :param minaf: Minimum alternate allele frequency.
        :type minaf: float or None

        """

        pargs = ['linreg', '-y', y, '-c', covariates, '-r', root, '--mac', str(minac)]
        if minaf:
            pargs.append('--maf')
            pargs.append(str(minaf))
        return self.hc.run_command(self, pargs)

    def logreg(self, test, y, covariates=None, root=None):
        """Test each variant for association using the logistic regression
        model.

        :param str test: Statistical test, one of: wald, lrt, or score.

        :param str y: Response sample annotation.  Must be Boolean or
            numeric with all values 0 or 1.

        :param str covariates: Covariant sample annotations, comma separated.

        :param str root: Variant annotation path to store result of linear regression.

        """

        pargs = ['logreg', '-t', test, '-y', y]
        if covariates:
            pargs.append('-c')
            pargs.append(covariates)
        if root:
            pargs.append('-r')
            pargs.append(root)
        return self.hc.run_command(self, pargs)

    def mendel_errors(self, output, fam):
        """Find Mendel errors; count per variant, individual and nuclear
        family.

        :param str output: Output root filename.

        :param str fam: Path to .fam file.

        """

        pargs = ['mendelerrors', '-o', output, '-f', fam]
        return self.hc.run_command(self, pargs)

    def pca(self, scores, loadings=None, eigenvalues=None, k=10, arrays=False):
        """Run Principal Component Analysis (PCA) on the matrix of genotypes.

        **Examples**

        ``pca`` expects a variant dataset with biallelic autosomal variants. Scores are computed and stored as sample annotations of type Struct by default; variant loadings and eigenvalues can optionally be computed and stored in variant and global annotations, respectively.

        To compute the first 10 principal component scores and store as annotations of type Struct:

        >>> vds1 = hc.read('example.vds').pca('sa.pca')

        The annotation ``sa.pca`` on ``vds1`` is a Stuct with fields ``PC1`` to ``PC10`` of type Double, so for example ``sa.pca.PC1`` is the score from the top PC.

        To compute the first 5 principal component scores, loadings, and eigenvalues, and store as annotations of type Array[Double]:

        >>> vds2 = hc.read('example.vds').pca('sa.scores', 'va.loadings', 'global.evals', 5, True)

        The three new annotations on ``vds2`` are:

        +------------------+---------------+----------------------------------------------+
        | Name             | Type          | Description                                  |
        +==================+===============+==============================================+
        | ``sa.scores``    | Array[Double] | Array of sample scores from the top five PCs |
        +------------------+---------------+----------------------------------------------+
        | ``va.loadings``  | Array[Double] | Array of variant loadings in the top five PCs|
        +------------------+---------------+----------------------------------------------+
        | ``global.evals`` | Array[Double] | Array of the top five eigenvalues            |
        +------------------+---------------+----------------------------------------------+

        **Implementation Details**

        Hail supports principal component analysis (PCA) of genotype data, a now-standard procedure `Patterson, Price and Reich, 2006 <http://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.0020190>`_. PCA is based on the singular value decomposition (SVD) of a standardized genotype matrix :math:`M`, computed as follows.

        An :math:`n \\times m` matrix :math:`C` records raw genotypes, with rows indexed by :math:`n` samples and columns indexed by :math:`m` bialellic autosomal variants; :math:`C_{ij}` is the number of alternate alleles of variant :math:`j` carried by sample :math:`i`, which can be 0, 1, 2, or missing. For each variant :math:`j`, the sample alternate allele frequency :math:`p_j` is computed as half the mean of the non-missing entries of column :math:`j`. Entries of :math:`M` are then mean-centered and variance-normalized as

        .. math::

          M_{ij} = \\frac{C_{ij}-2p_j}{\sqrt{2p_j(1-p_j)m}},

        with :math:`M_{ij} = 0` for :math:`C_{ij}` missing (i.e. mean genotype imputation). This scaling normalizes genotype variances to a common value :math:`1/m` for variants in Hardy-Weinberg equilibrium and is further motivated in the paper cited above. (The resulting amplification of signal from the low end of the allele frequency spectrum will also introduce noise for rare variants; common practice is to filter out variants with minor allele frequency below some cutoff.)  The factor :math:`1/m` gives each sample row approximately unit total variance (assuming linkage equilibrium) and yields the sample correlation or genetic relationship matrix (GRM) as simply :math:`MM^T`.

        PCA then computes the SVD

        .. math::

          M = USV^T

        where columns of :math:`U` are left singular vectors (orthonormal in :math:`\mathbb{R}^n`), columns of :math:`V` are right singular vectors (orthonormal in :math:`\mathbb{R}^m`), and :math:`S=\mathrm{diag}(s_1, s_2, \ldots)` with ordered singular values :math:`s_1 \ge s_2 \ge \cdots \ge 0`. Typically one computes only the first :math:`k` singular vectors and values, yielding the best rank :math:`k` approximation :math:`U_k S_k V_k^T` of :math:`M`; the truncations :math:`U_k`, :math:`S_k` and :math:`V_k` are :math:`n \\times k`, :math:`k \\times k` and :math:`m \\times k` respectively.

        From the perspective of the samples or rows of :math:`M` as data, :math:`V_k` contains the variant loadings for the first :math:`k` PCs while :math:`MV_k = U_k S_k` contains the first :math:`k` PC scores of each sample. The loadings represent a new basis of features while the scores represent the projected data on those features. The eigenvalues of the GRM :math:`MM^T` are the squares of the singular values :math:`s_1^2, s_2^2, \ldots`, which represent the variances carried by the respective PCs. By default, Hail only computes the loadings if the ``-l`` option is specified.

        *Note:* In PLINK/GCTA the GRM is taken as the starting point and it is computed slightly differently with regard to missing data. Here the :math:`ij` entry of :math:`MM^T` is simply the dot product of rows :math:`i` and :math:`j` of :math:`M`; in terms of :math:`C` it is

        .. math::

          \\frac{1}{m}\sum_{l\in\mathcal{C}_i\cap\mathcal{C}_j}\\frac{(C_{il}-2p_l)(C_{jl} - 2p_l)}{2p_l(1-p_l)}

        where :math:`\mathcal{C}_i = \{l \mid C_{il} \\text{ is non-missing}\}`. In PLINK/GCTA the denominator :math:`m` is replaced with the number of terms in the sum :math:`\\lvert\mathcal{C}_i\cap\\mathcal{C}_j\\rvert`, i.e. the number of variants where both samples have non-missing genotypes. While this is arguably a better estimator of the true GRM (trading shrinkage for noise), it has the drawback that one loses the clean interpretation of the loadings and scores as features and projections.

        Separately, for the PCs PLINK/GCTA output the eigenvectors of the GRM; even ignoring the above discrepancy that means the left singular vectors :math:`U_k` instead of the component scores :math:`U_k S_k`. While this is just a matter of the scale on each PC, the scores have the advantage of representing true projections of the data onto features with the variance of a score reflecting the variance explained by the corresponding feature. (In PC bi-plots this amounts to a change in aspect ratio; for use of PCs as covariates in regression it is immaterial.)

        :param str scores: Sample annotation path to store scores.

        :param loadings: Variant annotation path to store site loadings.
        :type loadings: str or None

        :param eigenvalues: Global annotation path to store eigenvalues.
        :type eigenvalues: str or None

        :param k: Number of principal components.
        :type k: int or None

        :param bool arrays: Store annotations as type Array rather than Struct
        :type k: bool or None

        """

        pargs = ['pca', '--scores', scores, '-k', str(k)]
        if loadings:
            pargs.append('--loadings')
            pargs.append(loadings)
        if eigenvalues:
            pargs.append('--eigenvalues')
            pargs.append(eigenvalues)
        if arrays:
            pargs.append('--arrays')
        return self.hc.run_command(self, pargs)

    def persist(self, storage_level="MEMORY_AND_DISK"):
        """Persist the current dataset.

        :param storage_level: Storage level.  One of: NONE, DISK_ONLY,
            DISK_ONLY_2, MEMORY_ONLY, MEMORY_ONLY_2, MEMORY_ONLY_SER,
            MEMORY_ONLY_SER_2, MEMORY_AND_DISK, MEMORY_AND_DISK_2,
            MEMORY_AND_DISK_SER, MEMORY_AND_DISK_SER_2, OFF_HEAP

        """

        pargs = ['persist']
        if storage_level:
            pargs.append('-s')
            pargs.append(storage_level)
        return self.hc.run_command(self, pargs)

    def print_schema(self, output=None, attributes=False, va=False, sa=False, print_global=False):
        """Shows the schema for global, sample and variant annotations.

        :param output: Output file.
        :type output: str or None

        :param bool attributes: If True, print attributes.

        :param bool va: If True, print variant annotations schema.

        :param bool sa: If True, print sample annotations schema.

        :param bool print_global: If True, print global annotations schema.

        """

        pargs = ['printschema']
        if output:
            pargs.append('--output')
            pargs.append(output)
        if attributes:
            pargs.append('--attributes')
        if va:
            pargs.append('--va')
        if sa:
            pargs.append('--sa')
        if print_global:
            pargs.append('--global')
        return self.hc.run_command(self, pargs)

    def rename_samples(self, input):
        """Rename samples.

        **Example**

        >>> vds = (hc.read('example.vds')
        >>>  .rename_samples('sample.map'))

        **Notes**

        The input file is a two column, tab-separated file.  The first column is the current sample
        name, the second column is the new sample name.  Samples which do not
        appear in the first column will not be renamed.  Lines in the input that
        do not correspond to any sample in the current dataset will be ignored.

        ``exportsamples`` can be used to generate a template for renaming
        samples.  For example, suppose you want to rename samples to remove
        spaces.  First, run:

        >>> (hc.read('example.vds')
        >>>  .export_samples('sample.map', 's.id, s.id')

        Then edit ``sample.map`` to remove spaces from the sample names in the
        second column and run the example above. Renaming samples is fast so there is no need to resave the dataset
        before performing analyses.

        :param str input: Input file.

        """

        pargs = ['renamesamples', '-i', input]
        return self.hc.run_command(self, pargs)

    def repartition(self, npartition, shuffle=True):
        """Increase or decrease the dataset sharding.  Can improve performance
        after large filters.

        :param int npartition: Number of partitions.

        :param bool shuffle: If True, shuffle to repartition.

        """

        pargs = ['repartition', '--partitions', str(npartition)]
        if not shuffle:
            pargs.append('--no-shuffle')
        return self.hc.run_command(self, pargs)

    def same(self, other):
        """Compare two VariantDatasets.

        :rtype: bool

        """
        try:
            return self.jvds.same(other.jvds, 1e-6)
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def sample_qc(self, branching_factor=None):
        """Compute per-sample QC metrics.

        :param branching_factor: Branching factor to use in tree aggregate.
        :type branching_factor: int or None

        """

        pargs = ['sampleqc']
        if branching_factor:
            pargs.append('-b')
            pargs.append(branching_factor)
        return self.hc.run_command(self, pargs)

    def show_globals(self, output=None):
        """Print or export all global annotations as JSON

        :param output: Output file.
        :type output: str or None

        """

        pargs = ['showglobals']
        if output:
            pargs.append('-o')
            pargs.append(output)
        return self.hc.run_command(self, pargs)

    def sparkinfo(self):
        """Displays the number of partitions and persistence level of the
        dataset."""

        return self.hc.run_command(self, ['sparkinfo'])

    def split_multi(self, propagate_gq=False, keep_star_alleles=False):
        """Split multiallelic variants.

        **Examples**

        >>> (hc.import_vcf('data/sample.vcf')
        >>>  .split_multi()
        >>>  .write('data/split.vds'))

        **Implementation Details**

        We will explain by example. Consider a hypothetical 3-allelic
        variant::

          A   C,T 0/2:7,2,6:15:45:99,50,99,0,45,99

        split_multi will create two biallelic variants (one for each
        alternate allele) at the same position::

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

        Here is a second example for a het non-ref::

          A   C,T 1/2:2,8,6:16:45:99,50,99,45,0,99

        splits as::

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

        The info field AC in ``data/export.vcf`` will have ``Number=1``.

        **Annotations**

        ``split_multi`` adds the following annotations:

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

        :return: A VariantDataset of biallelic variants with split set
          to true.

        :rtype: VariantDataset

        """

        pargs = ['splitmulti']
        if propagate_gq:
            pargs.append('--propagate-gq')
        if keep_star_alleles:
            pargs.append('--keep-star-alleles')
        return self.hc.run_command(self, pargs)

    def tdt(self, fam, root='va.tdt'):
        """Find transmitted and untransmitted variants; count per variant and
        nuclear family.

        :param str fam: Path to .fam file.

        :param root: Variant annotation root to store TDT result.

        """

        pargs = ['tdt', '--fam', fam, '--root', root]
        return self.hc.run_command(self, pargs)

    def typecheck(self):
        """Check if all sample, variant and global annotations are consistent
        with the schema.

        """

        pargs = ['typecheck']
        return self.hc.run_command(self, pargs)

    def variant_qc(self):
        """Compute per-variant QC metrics.

        **Example**

        >>> vds = (hc.read('example.vds')
        >>>  .variantqc())

        .. _variantqc_annotations:

        **Annotations**

        ``variantqc`` computes 16 variant statistics from the genotype data and stores the results as variant annotations that can be accessed with ``va.qc.<identifier>``:

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

        """

        pargs = ['variantqc']
        return self.hc.run_command(self, pargs)

    def vep(self, config, block_size=None, root=None, force=False, csq=False):
        """Annotate variants with VEP.

        :param str config: Path to VEP configuration file.

        :param block_size: Number of variants to annotate per VEP invocation.
        :type block_size: int or None

        :param str root: Variant annotation path to store VEP output.

        :param bool force: If true, force VEP annotation from scratch.

        :param bool csq: If True, annotates VCF CSQ field as a String.
            If False, annotates with the full nested struct schema

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
        return self.hc.run_command(self, pargs)

    def variants_keytable(self):
        """Convert variants and variant annotations to a KeyTable."""

        try:
            return KeyTable(self.hc, self.jvds.variantsKT())
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def samples_keytable(self):
        """Convert samples and sample annotations to KeyTable."""

        try:
            return KeyTable(self.hc, self.jvds.samplesKT())
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def make_keytable(self, variant_condition, genotype_condition, key_names):
        """Make a KeyTable with one row per variant.

        Per sample field names in the result are formed by concatening
        the sample ID with the genotype_condition left hand side with
        dot (.).  If the left hand side is empty::

          `` = expr

        then the dot (.) is ommited.

        **Example**

        Consider a ``VariantDataset`` ``vds`` with 2 variants and 3 samples::

          Variant	FORMAT	A	B	C
          1:1:A:T	GT:GQ	0/1:99	./.	0/0:99
          1:2:G:C	GT:GQ	0/1:89	0/1:99	1/1:93

        Then::

          >>> vds = hc.import_vcf('data/sample.vcf')
          >>> vds.make_keytable('v = v', 'gt = g.gt', gq = g.gq', [])

        returns a ``KeyTable`` with schema::

          v: Variant
          A.gt: Int
          B.gt: Int
          C.gt: Int
          A.gq: Int
          B.gq: Int
          C.gq: Int

        in particular, the values would be::

          v	A.gt	B.gt	C.gt	A.gq	B.gq	C.gq
          1:1:A:T	1	NA	0	99	NA	99
          1:2:G:C	1	1	2	89	99	93

        :param variant_condition: Variant annotation expressions.
        :type variant_condition: str or list of str

        :param genotype_condition: Genotype annotation expressions.
        :type genotype_condition: str or list of str

        :param key_names: list of key columns
        :type key_names: list of str

        :rtype: KeyTable

        """
        
        if isinstance(variant_condition, list):
            variant_condition = ','.join(variant_condition)
        if isinstance(genotype_condition, list):
            genotype_condition = ','.join(genotype_condition)

        jkt = (scala_package_object(self.hc.jvm.org.broadinstitute.hail.driver)
               .makeKT(self.jvds, variant_condition, genotype_condition,
                       jarray(self.hc.gateway, self.hc.jvm.java.lang.String, key_names)))
        return KeyTable(self.hc, jkt)
