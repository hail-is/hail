from __future__ import print_function # Python 2 and 3 print compatibility

from pyspark.java_gateway import launch_gateway
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

from hail.dataset import VariantDataset
from hail.java import jarray, scala_object, scala_package_object, joption
from hail.keytable import KeyTable
from hail.utils import TextTableConfig
from py4j.protocol import Py4JJavaError

from random import randint

class FatalError(Exception):
    """:class:`.FatalError` is an error thrown by Hail method failures"""

    def __init__(self, message, java_exception):
        self.msg = message
        self.java_exception = java_exception
        super(FatalError)

    def __str__(self):
        return self.msg

class HailContext(object):
    """:class:`.HailContext` is the main entrypoint for Hail
    functionality.

    :param str log: Log file.

    :param bool quiet: Don't write log file.

    :param bool append: Append to existing log file.

    :param long block_size: Minimum size of file splits in MB.

    :param str parquet_compression: Parquet compression codec.

    :param int branching_factor: Branching factor to use in tree aggregate.

    :param str tmp_dir: Temporary directory for file merging.
    """

    def __init__(self, sc=None, appName="Hail", master=None, local='local[*]',
                 log='hail.log', quiet=False, append=False, parquet_compression='uncompressed',
                 block_size=1, branching_factor=50, tmp_dir='/tmp'):
        from pyspark import SparkContext
        SparkContext._ensure_initialized()

        self.gateway = SparkContext._gateway
        self.jvm = SparkContext._jvm

        # hail package
        self.hail = getattr(self.jvm, 'is').hail

        driver = scala_package_object(self.hail.driver)

        if not sc:
            self.jsc = driver.configureAndCreateSparkContext(
                appName, joption(self.jvm, master), local, parquet_compression, block_size)
            self.sc = SparkContext(gateway=self.gateway, jsc=self.jvm.JavaSparkContext(self.jsc))
        else:
            self.sc = sc
            # sc._jsc is a JavaSparkContext
            self.jsc = sc._jsc.sc()

        driver.configureHail(branching_factor, tmp_dir)
        driver.configureLogging(log, quiet, append)

        self.jsql_context = driver.createSQLContext(self.jsc)
        self.sql_context = SQLContext(self.sc, self.jsql_context)

    def _jstate(self, jvds):
        return self.hail.driver.State(
            self.jsc, self.jsql_context, jvds, scala_object(self.jvm.scala.collection.immutable, 'Map').empty())

    def _raise_py4j_exception(self, e):
        msg = scala_package_object(self.hail.utils).getMinimalMessage(e.java_exception)
        raise FatalError(msg, e.java_exception)

    def run_command(self, vds, pargs):
        jargs = jarray(self.gateway, self.jvm.java.lang.String, pargs)
        t = self.hail.driver.ToplevelCommands.lookup(jargs)
        cmd = t._1()
        cmd_args = t._2()
        jstate = self._jstate(vds.jvds if vds != None else None)

        try:
            result = cmd.run(jstate, cmd_args)
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

        return VariantDataset(self, result.vds())

    def grep(self, regex, path, max_count=100):
        """Grep big files, like, really fast.

        **Examples**

        Print all lines containing the string ``hello`` in *file.txt*:

        >>> hc.grep('hello','data/file.txt')

        Print all lines containing digits in *file1.txt* and *file2.txt*:

        >>> hc.grep('\d', ['data/file1.txt','data/file2.txt'])

        **Background**

        :py:meth:`~hail.HailContext.grep` mimics the basic functionality of Unix ``grep`` in parallel, printing results to screen. This command is provided as a convenience to those in the statistical genetics community who often search enormous text files like VCFs. Find background on regular expressions at `RegExr <http://regexr.com/>`_.

        :param str regex: The regular expression to match.

        :param path: The files to search.
        :type path: str or list of str

        :param int max_count: The maximum number of matches to return.

        :return: Nothing.
        """

        pargs = ["grep", regex]
        if isinstance(path, str):
            pargs.append(path)
        else:
            for p in path:
                pargs.append(p)

        pargs.append('--max-count')
        pargs.append(str(max_count))

        self.run_command(None, pargs)

    def import_annotations_table(self, path, variant_expr, code=None, npartitions=None, config=None):
        """Import variants and variant annotations from a delimited text file
        (text table) as a sites-only VariantDataset.

        :param path: The files to import.
        :type path: str or list of str

        :param str variant_expr: Expression to construct a variant
            from a row of the text table.  Must have type Variant.

        :param code: Expression to build the variant annotations.
        :type code: str or None

        :param npartitions: Number of partitions.
        :type npartitions: int or None

        :param config: Configuration options for importing text files
        :type config: :class:`.TextTableConfig` or None

        :rtype: :class:`.VariantDataset`
        """

        pargs = ['importannotations', 'table']
        if isinstance(path, str):
            pargs.append(path)
        else:
            for p in path:
                pargs.append(p)

        pargs.append('--variant-expr')
        pargs.append(variant_expr)

        if code:
            pargs.append('--code')
            pargs.append(code)

        if npartitions:
            pargs.append('--npartition')
            pargs.append(npartitions)

        if not config:
            config = TextTableConfig()

        pargs.extend(config.as_pargs())

        return self.run_command(None, pargs)

    def import_bgen(self, path, tolerance=0.2, sample_file=None, npartitions=None):
        """Import .bgen files as VariantDataset

        :param path: .bgen files to import.
        :type path: str or list of str

        :param float tolerance: If the sum of the dosages for a
            genotype differ from 1.0 by more than the tolerance, set
            the genotype to missing.

        :param sample_file: The sample file.
        :type sample_file: str or None

        :param npartitions: Number of partitions.
        :type npartitions: int or None

        :rtype: :class:`.VariantDataset`
        """

        pargs = ["importbgen"]

        if isinstance(path, str):
            pargs.append(path)
        else:
            for p in path:
                pargs.append(p)

        if sample_file:
            pargs.append('--samplefile')
            pargs.append(sample_file)

        if npartitions:
            pargs.append('--npartition')
            pargs.append(str(npartitions))

        pargs.append('--tolerance')
        pargs.append(str(tolerance))

        return self.run_command(None, pargs)

    def import_gen(self, path, sample_file=None, tolerance=0.02, npartitions=None, chromosome=None):
        """Import .gen files as VariantDataset.

        **Examples**

        Read a .gen file and a .sample file and write to a .vds file::

        >>> (hc.import_gen('data/example.gen', sample_file='data/example.sample')
        >>>  .write('data/example.vds'))

        Load multiple files at the same time with `Hadoop glob patterns <../reference.html#hadoopglob>`_::

        >>> (hc.import_gen('data/example.chr*.gen', sample_file='data/example.sample')
        >>>  .write('data/example.vds'))

        **Notes**

        For more information on the .gen file format, see `here <http://www.stats.ox.ac.uk/%7Emarchini/software/gwas/file_format.html#mozTocId40300>`_.

        To ensure that the .gen file(s) and .sample file are correctly prepared for import:

        - If there are only 5 columns before the start of the dosage data (chromosome field is missing), you must specify the chromosome using the ``chromosome`` parameter

        - No duplicate sample IDs are allowed

        The first column in the .sample file is used as the sample ID ``s.id``.

        .. _dosagefilters:

        **Dosage representation**

        Since dosages are understood as genotype probabilities, :py:meth:`~hail.HailContext.import_gen` automatically sets to missing those genotypes for which the sum of the dosages is a distance greater than the ``tolerance`` paramater from 1.0.  The default tolerance is 0.02, so a genotypes with sum .97 or 1.03 is filtered out, whereas a genotype with sum .98 or 1.02 remains.

        :py:meth:`~hail.HailContext.import_gen` normalizes all dosages to sum to 1.0. Therefore, an input dosage of (0.98, 0.0, 0.0) will be stored as (1.0, 0.0, 0.0) in Hail.

        Even when the dosages sum to 1.0, Hail may store slightly different values than the original GEN file (maximum observed difference is 3E-4).

        **Annotations**

        :py:meth:`~hail.HailContext.import_gen` adds the following variant annotations:

         - **va.varid** (*String*) -- 2nd column of .gen file if chromosome present, otherwise 1st column.

         - **va.rsid** (*String*) -- 3rd column of .gen file if chromosome present, otherwise 2nd column.

        :param path: .gen files to import.
        :type path: str or list of str

        :param sample_file: The sample file.
        :type sample_file: str or None

        :param float tolerance: If the sum of the dosages for a genotype differ from 1.0 by more than the tolerance, set the genotype to missing.

        :param npartitions: Number of partitions.
        :type npartitions: int or None

        :param chromosome: Chromosome if not listed in the .gen file.
        :type chromosome: str or None

        :rtype: :class:`.VariantDataset`
        :return: A VariantDataset imported from a .gen and .sample file.

        """

        pargs = ["importgen"]

        if isinstance(path, str):
            pargs.append(path)
        else:
            for p in path:
                pargs.append(p)

        if sample_file:
            pargs.append('--samplefile')
            pargs.append(sample_file)

        if chromosome:
            pargs.append('--chromosome')
            pargs.append(chromosome)

        if npartitions:
            pargs.append('--npartition')
            pargs.append(str(npartitions))

        if tolerance:
            pargs.append('--tolerance')
            pargs.append(str(tolerance))

        return self.run_command(None, pargs)

    def import_keytable(self, path, key_names, npartitions=None, config=None):
        """Import delimited text file (text table) as KeyTable.

        :param path: files to import.
        :type path: str or list of str

        :param key_names: The name(s) of fields to be considered keys
        :type key_names: str or list of str

        :param npartitions: Number of partitions.
        :type npartitions: int or None

        :param config: Configuration options for importing text files
        :type config: :class:`.TextTableConfig` or None

        :rtype: :class:`.KeyTable`
        """
        path_args = []
        if isinstance(path, str):
            path_args.append(path)
        else:
            for p in path:
                path_args.append(p)

        if not isinstance(key_names, str):
            key_names = ','.join(key_names)

        if not npartitions:
            npartitions = self.sc.defaultMinPartitions

        if not config:
            config = TextTableConfig()

        return KeyTable(self, self.hail.keytable.KeyTable.importTextTable(
            self.jsc, jarray(self.gateway, self.jvm.java.lang.String, path_args), key_names, npartitions, config.to_java(self)))

    def import_plink(self, bed, bim, fam, npartitions=None, delimiter='\\\\s+', missing='NA', quantpheno=False):
        """
        Import PLINK binary file (BED, BIM, FAM) as VariantDataset

        **Examples**

        Import data from a PLINK binary file:

        >>> vds = (hc.import_plink(bed="data/test.bed",
        >>>                        bim="data/test.bim",
        >>>                        fam="data/test.fam"))


        **Implementation Details**

        Only binary SNP-major mode files can be read into Hail. To convert your file from individual-major mode to SNP-major mode, use PLINK to read in your fileset and use the ``--make-bed`` option.

        The centiMorgan position is not currently used in Hail (Column 3 in BIM file).

        The ID (``s.id``) used by Hail is the individual ID (column 2 in FAM file).

        .. warning::

            No duplicate individual IDs are allowed.

        Chromosome names (Column 1) are automatically converted in the following cases:
        
          - 23 => "X"
          - 24 => "Y"
          - 25 => "X"
          - 26 => "MT"

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

        :param npartitions: Number of partitions.
        :type npartitions: int or None

        :param str missing: The string used to denote missing values **only** for the phenotype field. This is in addition to "-9", "0", and "N/A" for case-control phenotypes.

        :param str delimiter: FAM file field delimiter regex.

        :param bool quantpheno: If True, FAM phenotype is interpreted as quantitative.

        :return: A VariantDataset imported from a PLINK binary file.

        :rtype: :class:`.VariantDataset`

        """
        pargs = ["importplink"]

        pargs.append('--bed')
        pargs.append(bed)

        pargs.append('--bim')
        pargs.append(bim)

        pargs.append('--fam')
        pargs.append(fam)

        if npartitions:
            pargs.append('--npartition')
            pargs.append(npartitions)

        if quantpheno:
            pargs.append('--quantpheno')

        pargs.append('--missing')
        pargs.append(missing)

        pargs.append('--delimiter')
        pargs.append(delimiter)

        return self.run_command(None, pargs)

    def read(self, path, sites_only=False):
        """Read .vds files as VariantDataset

        :param path: .vds files to read.
        :type path: str or list of str

        :param bool sites_only: If True, create sites-only
          VariantDataset.  Don't load sample ids, sample annotations
          or gneotypes.

        :rtype: :class:`.VariantDataset`

        When loading multiple .vds files, they must have the same
        sample ids, split status and variant metadata.

        """
        pargs = ["read"]

        if isinstance(path, str):
            pargs.append(path)
        else:
            for p in path:
                pargs.append(p)

        if sites_only:
            pargs.append("--skip-genotypes")
        return self.run_command(None, pargs)

    def import_vcf(self, path, force=False, force_bgz=False, header_file=None, npartitions=None,
                   sites_only=False, store_gq=False, pp_as_pl=False, skip_bad_ad=False):
        """Import .vcf files as VariantDataset

        :param path: .vcf files to read.
        :type path: str or list of str

        :param bool force: If True, load .gz files serially.

        :param bool force_bgz: If True, load .gz files as blocked gzip files (BGZF)

        :param header_file: File to load VCF header from.  If not specified, the first file in path is used.
        :type header_file: str or None

        :param npartitions: Number of partitions.
        :type npartitions: int or None

        :param bool sites_only: If True, create sites-only
            VariantDataset.  Don't load sample ids, sample annotations
            or gneotypes.

        :param bool store_gq: If True, store GQ FORMAT field instead of computing from PL.

        :param bool pp_as_pl: If True, store PP FORMAT field as PL.  EXPERIMENTAL.

        :param bool skip_bad_ad: If True, set AD FORMAT field with
            wrong number of elements to missing, rather than setting
            the entire genotype to missing.

        """

        pargs = ["importvcf"]

        if isinstance(path, str):
            pargs.append(path)
        else:
            for p in path:
                pargs.append(p)

        if force:
            pargs.append('--force')

        if force_bgz:
            pargs.append('--force-bgz')

        if header_file:
            pargs.append('--header-file')
            pargs.append(header_file)

        if npartitions:
            pargs.append('--npartition')
            pargs.append(str(npartitions))

        if pp_as_pl:
            pargs.append('--pp-as-pl')

        if skip_bad_ad:
            pargs.append('--skip-bad-ad')

        if sites_only:
            pargs.append('--skip-genotypes')

        if store_gq:
            pargs.append('--store-gq')

        return self.run_command(None, pargs)

    def index_bgen(self, path):
        """Index .bgen files.  import_bgen cannot run with these indicies.

        :param path: .bgen files to index.
        :type path: str or list of str

        :return: Nothing.
        """

        pargs = ["indexbgen"]

        if isinstance(path, str):
            pargs.append(path)
        else:
            for p in path:
                pargs.append(p)

        self.run_command(None, pargs)

    def balding_nichols_model(self, populations, samples, variants, partitions = None,
                              pop_dist=None,
                              fst=None,
                              root="bn",
                              seed=0):
        """
        Generate a VariantDataset using the Balding-Nichols model.

        **Examples**

        To generate a VDS with 3 populations, 100 samples in total, and 1000 variants:

        >>> vds = hc.balding_nichols_model(3, 100, 1000)

        To generate a VDS with 4 populations, 2000 samples, 5000 variants, 10 partitions, population distribution [0.1, 0.2, 0.3, 0.4], Fst values [.02, .06, .04, .12],  and root "balding", and random seed 1:

        >>> vds = hc.balding_nichols_model(4, 40, 150, 10, pop_dist=[0.1, 0.2, 0.3, 0.4], fst=[.02, .06, .04, .12], root="balding", seed=1)

        **Details**

        Hail is able to randomly generate a VDS using the Balding-Nichols model.

        - `K` populations are labeled by the integers `0`, `1`, ..., `K - 1`
        - `N` samples are named by the strings `"0"`, `"1"`, ..., `"N - 1"`
        - `M` variants are defined as `1:1:A:C`, `1:2:A:C`, ..., `1:M:A:C`
        - The ancestral frequency distribution is uniform on [.1, .9]
        - The population distribution defaults to uniform.
        - The F_st values default to 0.1.
        - The number of partitions defaults to one partition per million genotypes (i.e. samples * variants / 10^6) or 8, whichever is larger.

        Balding-Nichols models genotypes of individuals from a structured population comprising :math:`K` homogeneous subpopulations
        that have each diverged from a single ancestral population (a star phylogeny). We take :math:`N` samples and :math:`M` bi-allelic variants in perfect
        linkage equilibrium. The relative sizes of the subpopulations are given by a probability vector :math:`\pi`; the ancestral allele frequencies are
        drawn independently from a frequency spectrum :math:`P_0`; the subpopulations have diverged with possibly different :math:`F_{ST}` parameters :math:`F_k`
        (here and below, lowercase indices run over a range bounded by the corresponding uppercase parameter, e.g. :math:`k = 1, \ldots, K`).
        A key feature is that we model the subpopulation allele frequencies as drawn from beta distributions, a useful continuous approximation of
        the effect of genetic drift. We denote the individual subpopulation memberships by :math:`k_n`, the ancestral allele frequences :math:`p_{0, m}`,
        the subpopulation allele frequencies :math:`p_{k, m}`, and the genotypes :math:`g_{n, m}`.

        .. math::

          K, N, M &\in \mathbb N

          \pi &\in \mathcal{P}(\{1,\ldots,K\})

          P_0 &\in \mathcal{P}([0.1,0.9])

          F_k &\in [0,1]

        and the following random variables:

        .. math::
            k_n \,&\sim\, \pi

            p_{0,m}\,&\sim\, P_0

            p_{k,m}\mid p_{0,m}\,&\sim\, \mathrm{Beta}(\mu = p_{0,m},\, \sigma^2 = F_k p_{0,m}(1 - p_{0,m}))

            g_{n,m}\mid k_n, p_{k, m} \,&\sim\, \mathrm{Binomial}(2, p_{k_n, m}).

        We have parametrized the Beta distribution by its mean and variance; the usual (natural) parameters are :math:`a = (1 - p)(1 - F)/F,\; b = p(1-F)/F` with :math:`F = F_k,\; p = p_{0,m}`.

        **Annotations**

        Given the default root ``bn``, :py:meth:`~hail.HailContext.baldingnichols` adds the following global, variant, and sample annotations:

         - **global.bn.nPops** (*Int*) -- Number of populations
         - **global.bn.nSamples** (*Int*) -- Number of samples
         - **global.bn.nVariants** (*Int*) -- Number of variants
         - **global.bn.popDist** (*Array[Double]*) -- Normalized population distribution indexed by population
         - **global.bn.Fst** (*Array[Double]*) -- F_st values indexed by population
         - **global.bn.seed** (*Int*) -- Random seed
         - **sa.bn.pop** (*Int*) -- Population of sample
         - **va.bn.ancestralAF** (*Double*) -- Ancestral allele frequency
         - **va.bn.AF** (*Array[Double]*) -- Allele frequency indexed by population

        :param int populations: Number of populations.

        :param int samples: Number of samples.

        :param int variants: Number of variants.

        :param int partitions: Number of partitions.

        :param pop_dist: Unnormalized population distribution
        :type pop_dist: array of float or None

        :param fst: F_st values
        :type fst: array of float or None

        :param str root: Annotation root to follow global, sa and va.

        :param int seed: Random seed.

        :rtype: :class:`.VariantDataset`
        :return: A VariantDataset generated by the Balding-Nichols Model.

        """


        if pop_dist is None:
            jvm_pop_dist_opt = joption(self.jvm, pop_dist)
        else:
            jvm_pop_dist_opt = joption(self.jvm, jarray(self.gateway, self.jvm.double, pop_dist))


        if fst is None:
            jvm_fst_opt = joption(self.jvm, fst)
        else:
            jvm_fst_opt = joption(self.jvm, jarray(self.gateway, self.jvm.double, fst))



        return VariantDataset(self, self.hail.stats.BaldingNicholsModel.apply(self.jsc,  populations, samples, variants,
                            jvm_pop_dist_opt,
                            jvm_fst_opt,
                            seed,
                            joption(self.jvm, partitions), root))

    def stop(self):
        self.sc.stop()
        self.sc = None
