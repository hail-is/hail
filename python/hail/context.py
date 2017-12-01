from __future__ import print_function  # Python 2 and 3 print compatibility

from hail.typecheck import *
from pyspark import SparkContext
from pyspark.sql import SQLContext

from hail.dataset import VariantDataset
from hail.typ import Type, TInt64, TDict, TString
from hail.java import *
from hail.keytable import KeyTable
from hail.stats import UniformDist, TruncatedBetaDist, BetaDist
from hail.utils import wrap_to_list, get_env_or_default
from hail.history import *
from hail.representation.genomeref import GenomeReference


class HailContext(HistoryMixin):
    """The main entry point for Hail functionality.

    .. warning::
        Only one Hail context may be running in a Python session at any time. If you
        need to reconfigure settings, restart the Python session or use the :py:meth:`.HailContext.stop` method.

    :param sc: Spark context, one will be created if None.
    :type sc: :class:`.pyspark.SparkContext`

    :param appName: Spark application identifier.

    :param master: Spark cluster master.

    :param local: Local resources to use.

    :param log: Log path.

    :param bool quiet: Don't write logging information to standard error.

    :param append: Write to end of log file instead of overwriting.

    :param min_block_size: Minimum file split size in MB.

    :param branching_factor: Branching factor for tree aggregation.

    :param tmp_dir: Temporary directory for file merging. If None, use the value
                    of the environment variable TMPDIR, unless TMPDIR is unset,
                    in which case use '/tmp'.

    :param default_reference: Default reference genome to use. Can be set to "GRCh37" for
                             :py:meth:`.GenomeReference.GRCh37`, "GRCh38" for :py:meth:`.GenomeReference.GRCh38`,
                             or the path to a JSON file for constructing a reference genome as described in
                             :py:meth:`.GenomeReference.from_file`.
    :type default_reference: str

    :ivar sc: Spark context
    :vartype sc: :class:`.pyspark.SparkContext`

    """

    @record_init
    @typecheck_method(sc=nullable(SparkContext),
                      app_name=strlike,
                      master=nullable(strlike),
                      local=strlike,
                      log=strlike,
                      quiet=bool,
                      append=bool,
                      min_block_size=integral,
                      branching_factor=integral,
                      tmp_dir=nullable(strlike),
                      default_reference=strlike)
    def __init__(self, sc=None, app_name="Hail", master=None, local='local[*]',
                 log='hail.log', quiet=False, append=False,
                 min_block_size=1, branching_factor=50, tmp_dir=None,
                 default_reference="GRCh37"):

        if Env._hc:
            raise FatalError('Hail Context has already been created, restart session '
                             'or stop Hail context to change configuration.')

        SparkContext._ensure_initialized()

        self._gateway = SparkContext._gateway
        self._jvm = SparkContext._jvm

        # hail package
        self._hail = getattr(self._jvm, 'is').hail

        Env._jvm = self._jvm
        Env._gateway = self._gateway

        jsc = sc._jsc.sc() if sc else None

        tmp_dir = get_env_or_default(tmp_dir, 'TMPDIR', '/tmp')

        # we always pass 'quiet' to the JVM because stderr output needs
        # to be routed through Python separately.
        self._jhc = self._hail.HailContext.apply(
            jsc, app_name, joption(master), local, log, True, append,
            min_block_size, branching_factor, tmp_dir)

        self._jsc = self._jhc.sc()
        self.sc = sc if sc else SparkContext(gateway=self._gateway, jsc=self._jvm.JavaSparkContext(self._jsc))
        self._jsql_context = self._jhc.sqlContext()
        self._sql_context = SQLContext(self.sc, self._jsql_context)
        self._counter = 1

        super(HailContext, self).__init__()

        # do this at the end in case something errors, so we don't raise the above error without a real HC
        Env._hc = self

        self._default_ref = None
        Env.hail().variant.GenomeReference.setDefaultReference(self._jhc, default_reference)

        sys.stderr.write('Running on Apache Spark version {}\n'.format(self.sc.version))
        if self._jsc.uiWebUrl().isDefined():
            sys.stderr.write('SparkUI available at {}\n'.format(self._jsc.uiWebUrl().get()))

        if not quiet:
            connect_logger('localhost', 12888)

        sys.stderr.write(
            'Welcome to\n'
            '     __  __     <>__\n'
            '    / /_/ /__  __/ /\n'
            '   / __  / _ `/ / /\n'
            '  /_/ /_/\_,_/_/_/   version {}\n'.format(self.version))

        if self.version.startswith('devel'):
            sys.stderr.write('WARNING: This is an unstable development build.\n')

    def _set_history(self, history):
        assert not self._history_was_set, "Cannot set history for HailContext more than once."
        self._history = history.set_varid("hc")
        self._history_was_set = True
        
    @staticmethod
    def get_running():
        """Return the running Hail context in this Python session.

        **Example**

        .. doctest::
            :options: +SKIP

            >>> HailContext()  # oops! Forgot to bind to 'hc'
            >>> hc = HailContext.get_running()  # recovery

        Useful to recover a Hail context that has been created but is unbound.

        :return: Current Hail context.
        :rtype: :class:`.HailContext`
        """

        return Env.hc()

    @property
    def version(self):
        """Return the version of Hail associated with this HailContext.

        :rtype: str
        """
        return self._jhc.version()

    @property
    @record_property
    def default_reference(self):
        """Return the default reference genome.

        :rtype: :class:`.GenomeReference`
        """

        if not self._default_ref:
            self._default_ref = GenomeReference._from_java(Env.hail().variant.GenomeReference.defaultReference())
        return self._default_ref

    @handle_py4j
    @typecheck_method(regex=strlike,
                      path=oneof(strlike, listof(strlike)),
                      max_count=integral)
    def grep(self, regex, path, max_count=100):
        """Grep big files, like, really fast.

        **Examples**

        Print all lines containing the string ``hello`` in *file.txt*:

        >>> hc.grep('hello','data/file.txt')

        Print all lines containing digits in *file1.txt* and *file2.txt*:

        >>> hc.grep('\d', ['data/file1.txt','data/file2.txt'])

        **Background**

        :py:meth:`~hail.HailContext.grep` mimics the basic functionality of Unix ``grep`` in parallel, printing results to screen. This command is provided as a convenience to those in the statistical genetics community who often search enormous text files like VCFs. Hail uses `Java regular expression patterns <https://docs.oracle.com/javase/8/docs/api/java/util/regex/Pattern.html>`__. The `RegExr sandbox <http://regexr.com/>`__ may be helpful.

        :param str regex: The regular expression to match.

        :param path: The files to search.
        :type path: str or list of str

        :param int max_count: The maximum number of matches to return.
        """

        self._jhc.grep(regex, jindexed_seq_args(path), max_count)

    @handle_py4j
    @record_method
    @typecheck_method(path=oneof(strlike, listof(strlike)),
                      tolerance=numeric,
                      sample_file=nullable(strlike),
                      min_partitions=nullable(integral),
                      reference_genome=nullable(GenomeReference),
                      contig_recoding=nullable(dictof(strlike, strlike)))
    def import_bgen(self, path, tolerance=0.2, sample_file=None, min_partitions=None, reference_genome=None, contig_recoding=None):
        """Import .bgen file(s) as variant dataset.
        
        .. warning::
        
            A BGEN file must have a ``.idx`` file which can be generated by :py:meth:`~hail.HailContext.index_bgen`

        **Examples**

        Importing a BGEN file as a VDS.

        >>> vds = hc.import_bgen("data/example3.bgen", sample_file="data/example3.sample")

        **Notes**

        Hail supports importing data in the BGEN file format. For more information on the BGEN file format,
        see `here <http://www.well.ox.ac.uk/~gav/bgen_format/bgen_format.html>`__. Note that only v1.1 and v1.2 BGEN files
        are supported at this time. For v1.2 BGEN files, only **unphased** and **diploid** genotype probabilities are allowed and the
        genotype probability blocks must be either compressed with zlib or uncompressed.

        Before importing, ensure that:

          - The sample file has the same number of samples as the BGEN file.
          - No duplicate sample IDs are present.

        To load multiple files at the same time, use :ref:`Hadoop Glob Patterns <sec-hadoop-glob>`.

        .. _gpfilters:

        **Genotype probability (``gp``) representation**:

        The following modifications are made to genotype probabilities in BGEN v1.1 files:

          - Since genotype probabilities are understood to define a probability distribution, :py:meth:`~hail.HailContext.import_bgen` automatically sets to missing those genotypes for which the sum of the probabilities is a distance greater than the ``tolerance`` parameter from 1.0.  The default tolerance is 0.2, so a genotype with sum .79 or 1.21 is filtered out, whereas a genotype with sum .8 or 1.2 remains.

          - :py:meth:`~hail.HailContext.import_bgen` normalizes all probabilities to sum to 1.0. Therefore, an input distribution of (0.98, 0.0, 0.0) will be stored as (1.0, 0.0, 0.0) in Hail.

        **Annotations**

        :py:meth:`~hail.HailContext.import_bgen` adds the following variant annotations:

         - **va.varid** (*String*) -- 2nd column of .gen file if chromosome present, otherwise 1st column.

         - **va.rsid** (*String*) -- 3rd column of .gen file if chromosome present, otherwise 2nd column.

        :param path: .bgen files to import.
        :type path: str or list of str

        :param float tolerance: If the sum of the probabilities for a
            genotype differ from 1.0 by more than the tolerance, set
            the genotype to missing. Only applicable if the BGEN files are v1.1.

        :param sample_file: Sample file.
        :type sample_file: str or None

        :param min_partitions: Number of partitions.
        :type min_partitions: int or None

        :param reference_genome: Reference genome to use. Default is :class:`~.HailContext.default_reference`.
        :type reference_genome: :class:`.GenomeReference`

        :param contig_recoding: Dict of old contig name to new contig name. The new contig name must be in the reference genome given by ``reference_genome``.
        :type contig_recoding: dict of str to str (or None)

        :return: Variant dataset imported from .bgen file.
        :rtype: :class:`.VariantDataset`
        """

        rg = reference_genome if reference_genome else self.default_reference

        if contig_recoding:
            contig_recoding = TDict(TString(), TString())._convert_to_j(contig_recoding)

        jvds = self._jhc.importBgens(jindexed_seq_args(path), joption(sample_file),
                                     tolerance, joption(min_partitions), rg._jrep,
                                     joption(contig_recoding))
        return VariantDataset(self, jvds)

    @handle_py4j
    @record_method
    @typecheck_method(path=oneof(strlike, listof(strlike)),
                      sample_file=nullable(strlike),
                      tolerance=numeric,
                      min_partitions=nullable(integral),
                      chromosome=nullable(strlike),
                      reference_genome=nullable(GenomeReference),
                      contig_recoding=nullable(dictof(strlike, strlike)))
    def import_gen(self, path, sample_file=None, tolerance=0.2, min_partitions=None,
                   chromosome=None, reference_genome=None, contig_recoding=None):
        """Import .gen file(s) as variant dataset.

        **Examples**

        Read a .gen file and a .sample file and write to a .vds file:

        >>> (hc.import_gen('data/example.gen', sample_file='data/example.sample')
        ...    .write('output/gen_example1.vds'))

        Load multiple files at the same time with :ref:`Hadoop glob patterns <sec-hadoop-glob>`:

        >>> (hc.import_gen('data/example.chr*.gen', sample_file='data/example.sample')
        ...    .write('output/gen_example2.vds'))

        **Notes**

        For more information on the .gen file format, see `here <http://www.stats.ox.ac.uk/%7Emarchini/software/gwas/file_format.html#mozTocId40300>`__.

        To ensure that the .gen file(s) and .sample file are correctly prepared for import:

        - If there are only 5 columns before the start of the genotype probability data (chromosome field is missing), you must specify the chromosome using the ``chromosome`` parameter

        - No duplicate sample IDs are allowed

        The first column in the .sample file is used as the sample ID ``s``.

        Also, see section in :py:meth:`~hail.HailContext.import_bgen` linked :ref:`here <gpfilters>` for information about Hail's genotype probability representation.

        **Annotations**

        :py:meth:`~hail.HailContext.import_gen` adds the following variant annotations:

         - **va.varid** (*String*) -- 2nd column of .gen file if chromosome present, otherwise 1st column.

         - **va.rsid** (*String*) -- 3rd column of .gen file if chromosome present, otherwise 2nd column.

        :param path: .gen files to import.
        :type path: str or list of str

        :param str sample_file: The sample file.

        :param float tolerance: If the sum of the genotype probabilities for a genotype differ from 1.0 by more than the tolerance, set the genotype to missing.

        :param min_partitions: Number of partitions.
        :type min_partitions: int or None

        :param chromosome: Chromosome if not listed in the .gen file.
        :type chromosome: str or None

        :param reference_genome: Reference genome to use. Default is :class:`~.HailContext.default_reference`.
        :type reference_genome: :class:`.GenomeReference`

        :param contig_recoding: Dict of old contig name to new contig name. The new contig name must be in the reference genome given by ``reference_genome``.
        :type contig_recoding: dict of str to str (or None).

        :return: Variant dataset imported from .gen and .sample files.
        :rtype: :class:`.VariantDataset`
        """

        rg = reference_genome if reference_genome else self.default_reference

        if contig_recoding:
            contig_recoding = TDict(TString(), TString())._convert_to_j(contig_recoding)

        jvds = self._jhc.importGens(jindexed_seq_args(path), sample_file, joption(chromosome), joption(min_partitions),
                                    tolerance, rg._jrep, joption(contig_recoding))
        return VariantDataset(self, jvds)

    @handle_py4j
    @record_method
    @typecheck_method(paths=oneof(strlike, listof(strlike)),
                      key=oneof(strlike, listof(strlike)),
                      min_partitions=nullable(int),
                      impute=bool,
                      no_header=bool,
                      comment=nullable(strlike),
                      delimiter=strlike,
                      missing=strlike,
                      types=dictof(strlike, Type),
                      quote=nullable(char),
                      reference_genome=nullable(GenomeReference))
    def import_table(self, paths, key=[], min_partitions=None, impute=False, no_header=False,
                     comment=None, delimiter="\t", missing="NA", types={}, quote=None, reference_genome=None):
        """Import delimited text file (text table) as key table.

        The resulting key table will have no key columns, use :py:meth:`.KeyTable.key_by`
        to specify keys.
        
        **Example**
    
        Given this file

        .. code-block:: text

            $ cat data/samples1.tsv
            Sample	Height	Status  Age
            PT-1234	154.1	ADHD	24
            PT-1236	160.9	Control	19
            PT-1238	NA	ADHD	89
            PT-1239	170.3	Control	55

        The interesting thing about this table is that column ``Height`` is a floating-point number, 
        and column ``Age`` is an integer. We can either provide have Hail impute these types from 
        the file, or pass them ourselves:
        
        Pass the types ourselves:
        
        >>> table = hc.import_table('data/samples1.tsv', types={'Height': TFloat64(), 'Age': TInt32()})
        
        Note that string columns like ``Sample`` and ``Status`` do not need to be typed, because ``String``
        is the default type.
        
        Use type imputation (a bit easier, but requires reading the file twice):
        
        >>> table = hc.import_table('data/samples1.tsv', impute=True)

        **Detailed examples**

        Let's import annotations from a CSV file with missing data and special characters:

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

        >>> table = hc.import_table('data/samples2.tsv', delimiter=',', missing='.')

        Let's import annotations from a file with no header and sample IDs that need to be transformed. 
        Suppose the vds sample IDs are of the form ``NA#####``. This file has no header line, and the 
        sample ID is hidden in a field with other information.

        .. code-block: text

            $ cat data/samples3.tsv
            1kg_NA12345   female
            1kg_NA12346   male
            1kg_NA12348   female
            pgc_NA23415   male
            pgc_NA23418   male

        To import:

        >>> annotations = (hc.import_table('data/samples3.tsv', no_header=True)
        ...                   .annotate('sample = f0.split("_")[1]')
        ...                   .key_by('sample'))
        
        **Notes**
        
        The ``impute`` option tells Hail to scan the file an extra time to gather
        information about possible field types. While this is a bit slower for large files, (the 
        file is parsed twice), the convenience is often worth this cost.
        
        The ``delimiter`` parameter is a field separator regex. This regex follows the 
         `Java regex standard <http://docs.oracle.com/javase/7/docs/api/java/util/regex/Pattern.html>`_.
        
        .. note::
        
            Use ``delimiter='\\s+'`` to specify whitespace delimited files.
            
        The ``comment`` is an optional parameter which causes Hail to skip any line that starts in the
        given pattern. Passing ``comment='#'`` will skip any line beginning in a pound sign, for example.
        
        The ``missing`` parameter defines the representation of missing data in the table. 
        
        .. note::
        
            The ``comment`` and ``missing`` parameters are **NOT** regexes.

        The ``no_header`` option indicates that the file has no header line. If this option is passed, 
        then the column names will be ``f0``, ``f1``, ... ``fN`` (0-indexed). 
        
        The ``types`` option allows the user to pass the types of columns in the table. This is a 
        dict keyed by ``str``, with :py:class:`~hail.expr.Type` values. See the examples above for
        a standard usage. Additionally, this option can be used to override type imputation. For example,
        if a column in a file refers to chromosome and does not contain any sex chromosomes, it will be
        imputed as an integer, while most Hail methods expect chromosome to be passed as a string. Using
        the ``impute=True`` mode and passing ``types={'Chromosome': TString()}`` will solve this problem.
        
        The ``min_partitions`` option can be used to increase the number of partitions (level of sharding)
        of an imported table. The default partition size depends on file system and a number of other 
        factors (including the ``min_block_size`` of the hail context), but usually is between 32M and 128M.
        
        :param paths: Files to import.
        :type paths: str or list of str

        :param key: Key column(s).
        :type key: str or list of str

        :param min_partitions: Minimum number of partitions.
        :type min_partitions: int or None

        :param bool no_header: File has no header and the N columns are named ``f0``, ``f1``, ... ``fN`` (0-indexed)
        
        :param bool impute: Impute column types from the file
        
        :param comment: Skip lines beginning with the given pattern
        :type comment: str or None
        
        :param str delimiter: Field delimiter regex
        
        :param str missing: Specify identifier to be treated as missing
        
        :param types: Define types of fields in annotations files   
        :type types: dict with str keys and :py:class:`.Type` values

        :param quote: Quote character
        :type quote: str or None

        :param reference_genome: Reference genome to use when imputing Variant or Locus columns. Default is :class:`~.HailContext.default_reference`.
        :type reference_genome: :class:`.GenomeReference`

        :return: Key table constructed from text table.
        :rtype: :class:`.KeyTable`
        """

        key = wrap_to_list(key)
        paths = wrap_to_list(paths)
        jtypes = {k: v._jtype for k, v in types.items()}
        rg = reference_genome if reference_genome else self.default_reference

        jkt = self._jhc.importTable(paths, key, min_partitions, jtypes, comment, delimiter, missing,
                                    no_header, impute, quote, rg._jrep)
        return KeyTable(self, jkt)

    @handle_py4j
    @record_method
    @typecheck_method(bed=strlike,
                      bim=strlike,
                      fam=strlike,
                      min_partitions=nullable(integral),
                      delimiter=strlike,
                      missing=strlike,
                      quant_pheno=bool,
                      a2_reference=bool,
                      reference_genome=nullable(GenomeReference),
                      contig_recoding=nullable(dictof(strlike, strlike)))
    def import_plink(self, bed, bim, fam, min_partitions=None, delimiter='\\\\s+',
                     missing='NA', quant_pheno=False, a2_reference=True,
                     reference_genome=None, contig_recoding={'23': 'X', '24': 'Y', '25': 'X', '26': 'MT'}):
        """Import PLINK binary file (BED, BIM, FAM) as variant dataset.

        **Examples**

        Import data from a PLINK binary file:

        >>> vds = hc.import_plink(bed="data/test.bed",
        ...                       bim="data/test.bim",
        ...                       fam="data/test.fam")

        **Notes**

        Only binary SNP-major mode files can be read into Hail. To convert your file from individual-major mode to SNP-major mode, use PLINK to read in your fileset and use the ``--make-bed`` option.

        The centiMorgan position is not currently used in Hail (Column 3 in BIM file).

        The ID (``s``) used by Hail is the individual ID (column 2 in FAM file).

        .. warning::

            No duplicate individual IDs are allowed.

        **Annotations**

        :py:meth:`~hail.HailContext.import_plink` adds the following annotations:

         - **va.rsid** (*String*) -- Column 2 in the BIM file.
         - **sa.famID** (*String*) -- Column 1 in the FAM file. Set to missing if ID equals "0".
         - **sa.patID** (*String*) -- Column 3 in the FAM file. Set to missing if ID equals "0".
         - **sa.matID** (*String*) -- Column 4 in the FAM file. Set to missing if ID equals "0".
         - **sa.isFemale** (*String*) -- Column 5 in the FAM file. Set to missing if value equals "-9", "0", or "N/A".
           Set to true if value equals "2". Set to false if value equals "1".
         - **sa.isCase** (*String*) -- Column 6 in the FAM file. Only present if ``quantpheno`` equals False.
           Set to missing if value equals "-9", "0", "N/A", or the value specified by ``missing``.
           Set to true if value equals "2". Set to false if value equals "1".
         - **sa.qPheno** (*String*) -- Column 6 in the FAM file. Only present if ``quantpheno`` equals True.
           Set to missing if value equals ``missing``.

        :param str bed: PLINK BED file.

        :param str bim: PLINK BIM file.

        :param str fam: PLINK FAM file.

        :param min_partitions: Number of partitions.
        :type min_partitions: int or None

        :param str missing: The string used to denote missing values **only** for the phenotype field. This is in addition to "-9", "0", and "N/A" for case-control phenotypes.

        :param str delimiter: FAM file field delimiter regex.

        :param bool quant_pheno: If True, FAM phenotype is interpreted as quantitative.

        :param bool a2_reference: If True, A2 is treated as the reference allele. If False, A1 is treated as the reference allele.
        
        :param reference_genome: Reference genome to use. Default is :class:`~.HailContext.default_reference`.
        :type reference_genome: :class:`.GenomeReference`
        
        :param contig_recoding: Dict of old contig name to new contig name. The new contig name must be in the reference genome given by ``reference_genome``.
        :type contig_recoding: dict of str to str (or None).        

        :return: Variant dataset imported from PLINK binary file.
        :rtype: :class:`.VariantDataset`
        """

        rg = reference_genome if reference_genome else self.default_reference

        if contig_recoding:
            contig_recoding = TDict(TString(), TString())._convert_to_j(contig_recoding)

        jvds = self._jhc.importPlink(bed, bim, fam, joption(min_partitions), delimiter,
                                     missing, quant_pheno, a2_reference, rg._jrep,
                                     joption(contig_recoding))

        return VariantDataset(self, jvds)

    @handle_py4j
    @record_method
    @typecheck_method(path=strlike,
                      drop_samples=bool,
                      drop_variants=bool)
    def read(self, path, drop_samples=False, drop_variants=False):
        """Read .vds file as a variant dataset.

        :param str path: VDS file to read.

        :param bool drop_samples: If True, create sites-only variant
          dataset.  Don't load sample ids, sample annotations
          or gneotypes.

        :param bool drop_variants: If True, create samples-only variant
          dataset (no variants or genotypes).

        :return: Variant dataset read from disk.
        :rtype: :class:`.VariantDataset`

        """

        return VariantDataset(
            self,
            self._jhc.read(path, drop_samples, drop_variants))

    @handle_py4j
    @typecheck_method(path=oneof(strlike, listof(strlike)))
    def get_vcf_metadata(self, path):
        """Extract metadata from VCF header.

        **Examples**

        >>> metadata = hc.get_vcf_metadata('data/example2.vcf.bgz')

        **Notes**

        This method parses the VCF header to extract the `ID`, `Number`, `Type`, and `Description` fields
        from FORMAT and INFO lines as well as `ID` and `Description` for FILTER lines. For example, given the
        following header lines:

        .. code-block:: text

            ##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
            ##FILTER=<ID=LowQual,Description="Low quality">
            ##INFO=<ID=MQ,Number=1,Type=Float,Description="RMS Mapping Quality">

        The resulting Python dictionary returned would be

        .. code-block:: python

            metadata = {'format': {'DP': {'Number': '1', 'Type': 'Integer', 'Description': 'Read Depth'}},
                        'filter': {'LowQual': {'Description': 'Low quality'}},
                        'info': {'MQ': {'Number': '1', 'Type': 'Float', 'Description': 'RMS Mapping Quality'}}}

        which can be used with :py:class:`~hail.VariantDataset.export_vcf` to fill in the relevant fields in the header.

        :param path: VCF file(s) to read. If more than one file is given, the first file is used.
        :type path: str or list of str

        :rtype: (dict of str to (dict of str to (dict of str to str)))
        """

        typ = TDict(TString(), TDict(TString(), TDict(TString(), TString())))
        return typ._convert_to_py(self._jhc.parseVCFMetadata(jindexed_seq_args(path)))

    @handle_py4j
    @record_method
    @typecheck_method(path=oneof(strlike, listof(strlike)),
                      force=bool,
                      force_bgz=bool,
                      header_file=nullable(strlike),
                      min_partitions=nullable(integral),
                      drop_samples=bool,
                      call_fields=oneof(strlike, listof(strlike)),
                      reference_genome=nullable(GenomeReference),
                      contig_recoding=nullable(dictof(strlike, strlike)))
    def import_vcf(self, path, force=False, force_bgz=False, header_file=None, min_partitions=None,
                   drop_samples=False, call_fields=[], reference_genome=None, contig_recoding=None):
        """Import VCF file(s) as variant dataset.

        **Examples**

        >>> vds = hc.import_vcf('data/example2.vcf.bgz')

        **Notes**

        Hail is designed to be maximally compatible with files in the `VCF v4.2 spec <https://samtools.github.io/hts-specs/VCFv4.2.pdf>`__.

        :py:meth:`~hail.HailContext.import_vcf` takes a list of VCF files to load. All files must have the same header and the same set of samples in the same order
        (e.g., a variant dataset split by chromosome). Files can be specified as :ref:`Hadoop glob patterns <sec-hadoop-glob>`.

        Ensure that the VCF file is correctly prepared for import: VCFs should either be uncompressed (*.vcf*) or block compressed
        (*.vcf.bgz*).  If you have a large compressed VCF that ends in *.vcf.gz*, it is likely that the file is actually block-compressed,
        and you should rename the file to ".vcf.bgz" accordingly. If you actually have a standard gzipped file, it is possible to import
        it to Hail using the ``force`` optional parameter. However, this is not recommended -- all parsing will have to take place on one node because
        gzip decompression is not parallelizable. In this case, import could take significantly longer.

        If ``generic`` equals False (default), Hail makes certain assumptions about the genotype fields, see :class:`Representation <hail.representation.Genotype>`. On import, Hail filters
        (sets to no-call) any genotype that violates these assumptions. Hail interprets the format fields: GT, AD, OD, DP, GQ, PL; all others are
        silently dropped.

        If ``generic`` equals True, the genotype schema is a :py:class:`~hail.type.TStruct` with field names equal to the IDs of the FORMAT fields.
        The ``GT`` field is automatically read in as a :py:class:`~hail.type.TCall` type. To specify additional fields to import as a
        :py:class:`~hail.type.TCall` type, use the ``call_fields`` parameter. All other fields are imported as the type specified in the FORMAT header field.

        An example genotype schema after importing a VCF with ``generic=True`` is

        .. code-block:: text

            Struct {
                GT: Call,
                AD: Array[Int],
                DP: Int,
                GQ: Int,
                PL: Array[Int]
            }

        .. warning::

            - The variant dataset generated with ``generic=True`` will have significantly slower performance.

            - Not all :py:class:`.VariantDataset` methods will work with a generic genotype schema.

        :py:meth:`~hail.HailContext.import_vcf` does not perform deduplication - if the provided VCF(s) contain multiple records with the same chrom, pos, ref, alt, all
        these records will be imported and will not be collapsed into a single variant.

        Since Hail's genotype representation does not yet support ploidy other than 2,
        this method imports haploid genotypes as diploid. If ``generic=False``, Hail fills in missing indices
        in PL / PP arrays with 1000 to support the standard VCF / VDS "genotype schema.

        Below are two example haploid genotypes and diploid equivalents that Hail sees.

        .. code-block:: text

            Haploid:     1:0,6:7:70:70,0
            Imported as: 1/1:0,6:7:70:70,1000,0

            Haploid:     2:0,0,9:9:24:24,40,0
            Imported as: 2/2:0,0,9:9:24:24,1000,40,1000:1000:0


        .. note::
            
            Using the **FILTER** field:
            
            The information in the FILTER field of a VCF is contained in the ``va.filters`` annotation.
            This annotation is a ``Set`` and can be queried for filter membership with expressions 
            like ``va.filters.contains("VQSRTranche99.5...")``. Variants that are flagged as "PASS" 
            will have no filters applied; for these variants, ``va.filters.isEmpty()`` is true. Thus, 
            filtering to PASS variants can be done with :py:meth:`.VariantDataset.filter_variants_expr`
            as follows:
            
            >>> pass_vds = vds.filter_variants_expr('va.filters.isEmpty()', keep=True)

        **Annotations**

        - **va.filters** (*Set[String]*) -- Set containing all filters applied to a variant. 
        - **va.rsid** (*String*) -- rsID of the variant.
        - **va.qual** (*Double*) -- Floating-point number in the QUAL field.
        - **va.info** (*Struct*) -- All INFO fields defined in the VCF header
          can be found in the struct ``va.info``. Data types match the type
          specified in the VCF header, and if the declared ``Number`` is not
          1, the result will be stored as an array.

        :param path: VCF file(s) to read.
        :type path: str or list of str

        :param bool force: If True, load .gz files serially. This means that no downstream operations
            can be parallelized, so using this mode is strongly discouraged for VCFs larger than a few MB.

        :param bool force_bgz: If True, load .gz files as blocked gzip files (BGZF)

        :param header_file: File to load VCF header from.  If not specified, the first file in path is used.
        :type header_file: str or None

        :param min_partitions: Number of partitions.
        :type min_partitions: int or None

        :param bool drop_samples: If True, create sites-only variant
            dataset.  Don't load sample ids, sample annotations or
            genotypes.

        :param call_fields: FORMAT fields in VCF to treat as a :py:class:`~hail.type.TCall`. Only applies if ``generic=True``.
        :type call_fields: str or list of str

        :param bool generic: If True, read the genotype with a generic schema.
        
        :param reference_genome: Reference genome to use. Default is :class:`~.HailContext.default_reference`.
        :type reference_genome: :class:`.GenomeReference`

        :param contig_recoding: Dict of old contig name to new contig name. The new contig name must be in the reference genome given by ``reference_genome``.
        :type contig_recoding: dict of str to str (or None).

        :return: Variant dataset imported from VCF file(s)
        :rtype: :py:class:`.VariantDataset`

        """

        rg = reference_genome if reference_genome else self.default_reference

        if contig_recoding:
            contig_recoding = TDict(TString(), TString())._convert_to_j(contig_recoding)
        
        jvds = self._jhc.importVCFs(jindexed_seq_args(path), force, force_bgz, joption(header_file),
                                    joption(min_partitions), drop_samples, jset_args(call_fields), rg._jrep,
                                    joption(contig_recoding))

        return VariantDataset(self, jvds)

    @handle_py4j
    @record_method
    @typecheck_method(path=oneof(strlike, listof(strlike)),
                      min_partitions=nullable(integral),
                      drop_samples=bool,
                      cell_type=nullable(Type),
                      missing=strlike,
                      has_row_id_name=bool)
    def import_matrix(self, path, min_partitions=None, drop_samples=False, cell_type=None, missing="NA", has_row_id_name = False):
        """
        :param path: File(s) to read. Currently, takes 1 header line of column ids and subsequent lines of rowID, data... in TSV form where data can be parsed as an integer.
        :type path: str or list of str

        :param min_partitions: Number of partitions.
        :type min_partitions: int or None

        :param bool drop_samples: I don't know if this is relevant, but it only loads the row IDs. Default: False

        :param str cell_type: Tells function how to parse cell data. Can be Int32, Int64, Float32, Float64, or String. Default: Int64

        :param str missing: notation for cell with missing value. Default: "NA"

        :param str has_row_id_name: whether or not the table header has an entry for the Row IDs. Default: False

        :return: Variant dataset imported from file(s)
        :rtype: :py:class:`.VariantDataset`
        """

        if not cell_type:
            cell_type = TInt64()
        return VariantDataset(self,
                              self._jhc.importMatrices(jindexed_seq_args(path),
                                                       joption(min_partitions),
                                                       drop_samples,
                                                       cell_type._jtype,
                                                       missing,
                                                       has_row_id_name))

    @handle_py4j
    @typecheck_method(path=oneof(strlike, listof(strlike)))
    def index_bgen(self, path):
        """Index .bgen files. :py:meth:`.HailContext.import_bgen` cannot run without these indices.
        
        The index file is generated in the same directory as `path` with the filename of `path` appended by `.idx`.

        **Example**

        >>> hc.index_bgen("data/example3.bgen")

        .. warning::

            While this method parallelizes over a list of BGEN files, each file is
            indexed serially by one core. Indexing several BGEN files on a large
            cluster is a waste of resources, so indexing should generally be done
            as a one-time step separately from large analyses.

        :param path: .bgen files to index.
        :type path: str or list of str
        """

        self._jhc.indexBgen(jindexed_seq_args(path))

    @handle_py4j
    @record_method
    @typecheck_method(populations=integral,
                      samples=integral,
                      variants=integral,
                      num_partitions=nullable(integral),
                      pop_dist=nullable(listof(numeric)),
                      fst=nullable(listof(numeric)),
                      af_dist=oneof(UniformDist, BetaDist, TruncatedBetaDist),
                      seed=integral,
                      reference_genome=nullable(GenomeReference))
    def balding_nichols_model(self, populations, samples, variants, num_partitions=None,
                              pop_dist=None, fst=None, af_dist=UniformDist(0.1, 0.9),
                              seed=0, reference_genome=None):
        """Simulate a variant dataset using the Balding-Nichols model.

        **Examples**

        To generate a VDS with 3 populations, 100 samples in total, and 1000 variants:

        >>> vds = hc.balding_nichols_model(3, 100, 1000)

        To generate a VDS with 4 populations, 2000 samples, 5000 variants, 10 partitions, population distribution [0.1, 0.2, 0.3, 0.4], :math:`F_{ST}` values [.02, .06, .04, .12], ancestral allele frequencies drawn from a truncated beta distribution with a = .01 and b = .05 over the interval [0.05, 1], and random seed 1:

        >>> from hail.stats import TruncatedBetaDist
        >>> vds = hc.balding_nichols_model(4, 40, 150, 10,
        ...                                pop_dist=[0.1, 0.2, 0.3, 0.4],
        ...                                fst=[.02, .06, .04, .12],
        ...                                af_dist=TruncatedBetaDist(a=0.01, b=2.0, minVal=0.05, maxVal=1.0),
        ...                                seed=1)

        **Notes**

        Hail is able to randomly generate a VDS using the Balding-Nichols model.

        - :math:`K` populations are labeled by integers 0, 1, ..., K - 1
        - :math:`N` samples are named by strings 0, 1, ..., N - 1
        - :math:`M` variants are defined as ``1:1:A:C``, ``1:2:A:C``, ..., ``1:M:A:C``
        - The default ancestral frequency distribution :math:`P_0` is uniform on [0.1, 0.9]. Options are UniformDist(minVal, maxVal), BetaDist(a, b), and TruncatedBetaDist(a, b, minVal, maxVal). All three classes are located in hail.stats.
        - The population distribution :math:`\pi` defaults to uniform
        - The :math:`F_{ST}` values default to 0.1
        - The number of partitions defaults to one partition per million genotypes (i.e., samples * variants / 10^6) or 8, whichever is larger

        The Balding-Nichols model models genotypes of individuals from a structured population comprising :math:`K` homogeneous subpopulations
        that have each diverged from a single ancestral population (a `star phylogeny`). We take :math:`N` samples and :math:`M` bi-allelic variants in perfect
        linkage equilibrium. The relative sizes of the subpopulations are given by a probability vector :math:`\pi`; the ancestral allele frequencies are
        drawn independently from a frequency spectrum :math:`P_0`; the subpopulations have diverged with possibly different :math:`F_{ST}` parameters :math:`F_k`
        (here and below, lowercase indices run over a range bounded by the corresponding uppercase parameter, e.g. :math:`k = 1, \ldots, K`).
        For each variant, the subpopulation allele frequencies are drawn a `beta distribution <https://en.wikipedia.org/wiki/Beta_distribution>`__, a useful continuous approximation of
        the effect of genetic drift. We denote the individual subpopulation memberships by :math:`k_n`, the ancestral allele frequences by :math:`p_{0, m}`,
        the subpopulation allele frequencies by :math:`p_{k, m}`, and the genotypes by :math:`g_{n, m}`. The generative model in then given by:

        .. math::
            k_n \,&\sim\, \pi

            p_{0,m}\,&\sim\, P_0

            p_{k,m}\mid p_{0,m}\,&\sim\, \mathrm{Beta}(\mu = p_{0,m},\, \sigma^2 = F_k p_{0,m}(1 - p_{0,m}))

            g_{n,m}\mid k_n, p_{k, m} \,&\sim\, \mathrm{Binomial}(2, p_{k_n, m})

        We have parametrized the beta distribution by its mean and variance; the usual parameters are :math:`a = (1 - p)(1 - F)/F,\; b = p(1-F)/F` with :math:`F = F_k,\; p = p_{0,m}`.

        **Annotations**

        :py:meth:`~hail.HailContext.balding_nichols_model` adds the following global, sample, and variant annotations:

         - **global.nPops** (*Int*) -- Number of populations
         - **global.nSamples** (*Int*) -- Number of samples
         - **global.nVariants** (*Int*) -- Number of variants
         - **global.popDist** (*Array[Double]*) -- Normalized population distribution indexed by population
         - **global.Fst** (*Array[Double]*) -- :math:`F_{ST}` values indexed by population
         - **global.seed** (*Int*) -- Random seed
         - **global.ancestralAFDist** (*Struct*) -- Description of the ancestral allele frequency distribution
         - **sa.pop** (*Int*) -- Population of sample
         - **va.ancestralAF** (*Double*) -- Ancestral allele frequency
         - **va.AF** (*Array[Double]*) -- Allele frequency indexed by population

        :param int populations: Number of populations.

        :param int samples: Number of samples.

        :param int variants: Number of variants.

        :param int num_partitions: Number of partitions.

        :param pop_dist: Unnormalized population distribution
        :type pop_dist: array of float or None

        :param fst: :math:`F_{ST}` values
        :type fst: array of float or None

        :param af_dist: Ancestral allele frequency distribution
        :type af_dist: :class:`.UniformDist` or :class:`.BetaDist` or :class:`.TruncatedBetaDist`

        :param int seed: Random seed.

        :param reference_genome: Reference genome to use. Default is :class:`~.HailContext.default_reference`.
        :type reference_genome: :class:`.GenomeReference`

        :return: Variant dataset simulated using the Balding-Nichols model.
        :rtype: :class:`.VariantDataset`
        """

        if pop_dist is None:
            jvm_pop_dist_opt = joption(pop_dist)
        else:
            jvm_pop_dist_opt = joption(jarray(self._jvm.double, pop_dist))

        if fst is None:
            jvm_fst_opt = joption(fst)
        else:
            jvm_fst_opt = joption(jarray(self._jvm.double, fst))

        rg = reference_genome if reference_genome else self.default_reference
        
        jvds = self._jhc.baldingNicholsModel(populations, samples, variants,
                                             joption(num_partitions),
                                             jvm_pop_dist_opt,
                                             jvm_fst_opt,
                                             af_dist._jrep(),
                                             seed,
                                             rg._jrep)
        return VariantDataset(self, jvds)

    @handle_py4j
    @typecheck_method(expr=strlike)
    def eval_expr_typed(self, expr):
        """Evaluate an expression and return the result as well as its type.

        :param str expr: Expression to evaluate.

        :rtype: (annotation, :class:`.Type`)

        """

        x = self._jhc.eval(expr)
        t = Type._from_java(x._2())
        v = t._convert_to_py(x._1())
        return (v, t)

    @handle_py4j
    @typecheck_method(expr=strlike)
    def eval_expr(self, expr):
        """Evaluate an expression.

        :param str expr: Expression to evaluate.

        :rtype: annotation
        """

        r, t = self.eval_expr_typed(expr)
        return r

    def stop(self):
        """ Shut down the Hail context.

        It is not possible to have multiple Hail contexts running in a
        single Python session, so use this if you need to reconfigure the Hail
        context. Note that this also stops a running Spark context.
        """

        self.sc.stop()
        self.sc = None
        Env._jvm = None
        Env._gateway = None
        Env._hc = None

    @handle_py4j
    @record_method
    @typecheck_method(path=strlike)
    def read_table(self, path):
        """Read a KT file as key table.

        :param str path: KT file to read.

        :return: Key table read from disk.
        :rtype: :class:`.KeyTable`
        """

        jkt = self._jhc.readTable(path)
        return KeyTable(self, jkt)

    def _get_unique_id(self):
        self._counter += 1
        return "__uid_{}".format(self._counter)
