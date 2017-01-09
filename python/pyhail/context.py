from pyspark.java_gateway import launch_gateway
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

from pyhail.dataset import VariantDataset
from pyhail.java import jarray, scala_object, scala_package_object, joption
from pyhail.keytable import KeyTable
from pyhail.utils import TextTableConfig
from py4j.protocol import Py4JJavaError

class FatalError(Exception):
    """:class:`.FatalError` is an error thrown by Hail method failures"""

    def __init__(self, message, java_exception):
        self.msg = message
        self.java_exception = java_exception
        super(FatalError)

    def __str__(self):
        return self.msg

class HailContext(object):
    """:class:`.HailContext` is the main entrypoint for PyHail
    functionality.

    :param str log: Log file.

    :param bool quiet: Don't write log file.

    :param bool append: Append to existing log file.

    :param long block_size: Minimum size of file splits in MB.

    :param str parquet_compression: Parquet compression codec.

    :param int branching_factor: Branching factor to use in tree aggregate.

    :param str tmp_dir: Temporary directory for file merging.
    """

    def __init__(self, appName="PyHail", master=None, local='local[*]',
                 log='hail.log', quiet=False, append=False, parquet_compression='uncompressed',
                 block_size=1, branching_factor=50, tmp_dir='/tmp'):
        from pyspark import SparkContext
        SparkContext._ensure_initialized()

        self.gateway = SparkContext._gateway
        self.jvm = SparkContext._jvm

        self.jsc = scala_package_object(self.jvm.org.broadinstitute.hail.driver).configureAndCreateSparkContext(
            appName, joption(self.jvm, master), local,
            log, quiet, append, parquet_compression,
            block_size, branching_factor, tmp_dir)
        self.sc = SparkContext(gateway=self.gateway, jsc=self.jvm.JavaSparkContext(self.jsc))

        self.jsql_context = scala_package_object(self.jvm.org.broadinstitute.hail.driver).createSQLContext(self.jsc)
        self.sql_context = SQLContext(self.sc, self.jsql_context)

    def _jstate(self, jvds):
        return self.jvm.org.broadinstitute.hail.driver.State(
            self.jsc, self.jsql_context, jvds, scala_object(self.jvm.scala.collection.immutable, 'Map').empty())

    def _raise_py4j_exception(self, e):
        msg = scala_package_object(self.jvm.org.broadinstitute.hail.utils).getMinimalMessage(e.java_exception)
        raise FatalError(msg, e.java_exception)

    def run_command(self, vds, pargs):
        jargs = jarray(self.gateway, self.jvm.java.lang.String, pargs)
        t = self.jvm.org.broadinstitute.hail.driver.ToplevelCommands.lookup(jargs)
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

    def import_gen(self, path, tolerance=0.2, sample_file=None, npartitions=None, chromosome=None):
        """Import .bgen files as VariantDataset

        :param path: .gen files to import.
        :type path: str or list of str

        :param float tolerance: If the sum of the dosages for a
            genotype differ from 1.0 by more than the tolerance, set
            the genotype to missing.

        :param sample_file: The sample file.
        :type sample_file: str or None

        :param npartitions: Number of partitions.
        :type npartitions: int or None

        :param chromosome: Chromosome if not listed in the .gen file.
        :type chromosome: str or None

        :rtype: :class:`.VariantDataset`
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

        return KeyTable(self, self.jvm.org.broadinstitute.hail.keytable.KeyTable.importTextTable(
            self.jsc, jarray(self.gateway, self.jvm.java.lang.String, path_args), key_names, npartitions, config.to_java(self)))

    def import_plink(self, bed, bim, fam, npartitions=None, delimiter='\\\\s+', missing='NA', quantpheno=False):
        """
        Import PLINK binary file (.bed, .bim, .fam) as VariantDataset

        :param str bed: PLINK .bed file.

        :param str bim: PLINK .bim file.

        :param str fam: PLINK .fam file.

        :param npartitions: Number of partitions.
        :type npartitions: int or None

        :param str missing: The string used to denote missing values.

        :param str delimiter: .fam file field delimiter regex.

        :param bool quantpheno: If True, .fam phenotype is interpreted as quantitative.

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

        :param bool sites_only: If True, create sites-only VariantDataset.
            Don't load sample ids, sample annotations or genotypes.

        :param bool store_gq: If True, store GQ FORMAT field instead of computing from PL.

        :param bool pp_as_pl: If True, store PP FORMAT field as PL.  EXPERIMENTAL.

        :param bool skip_bad_ad: If True, set AD FORMAT field with wrong number
            of elements to missing, rather than setting the entire genotype to missing.

        Hail is designed to be maximally compatible with files in the `VCF v4.2 spec <https://samtools.github.io/hts-specs/VCFv4.2.pdf>`_.

        The ``path`` argument specifies the list of files to load.  All files must have the same header and the same set of samples in the same order (e.g., a dataset split by chromosome).  Files can be specified as `Hadoop glob patterns <https://www.hail.is/reference.html#hadoopglob>`_.

        Ensure that the VCF file is correctly prepared for import.  VCFs should be either uncompressed (".vcf") or block-compressed (".vcf.bgz").  If you have a large compressed VCF that ends in ".vcf.gz", it is likely that the file is actually block-compressed, and you should rename the file to ".vcf.bgz" accordingly.  If you actually have a standard gzipped file, it is possible to import it to hail using the ```force`` option.  However, this is not recommended: all parsing will have to take place on one core, because gzip decompression is not parallelizable.  In this case, import will take significantly longer.

        Note that VCF is an inefficient format, and it is much faster to read a VDS than a VCF.  If you are performing several analyses on a VCF, it is advised that you write the VCF to VDS form and operate on that.

        Hail makes certain assumptions about the genotype fields, see `Representation <https://www.hail.is/reference.html#Representation>`_.  On import, Hail filters (sets to no-call) any genotype that violates these assumptions.  Hail interpets the format fields: GT, AD, OD, DP, GQ, PL; all others are silently dropped.

        This function does not perform deduplication - if the provided VCF(s) contain multiple records with the same chrom, pos, ref, alt, all these records will be imported and will not be collapsed into a single variant.

        A dataset imported from VDS contains variant annotations read from the VCF:

        ===================   ===============  ==============
        Annotation name       Type             Description
        ===================   ===============  ==============
        ``va.pass``           ``Boolean``      true if the variant contains `PASS` in the filter field (false if ``.`` or other)
        ``va.filters``        ``Set[String]``  set containing the list of filters applied to a variant.  Accessible using ``va.filters.contains("VQSRTranche99.5...")``, for example
        ``va.rsid``           ``String``       rsid of the variant, if it has one ("." otherwise)
        ``va.qual``           ``Double``       the number in the qual field
        ``va.info.<field>``   ``T``            matches (with proper capitalization) any defined info field.  Data types match the type specified in the vcf header, and if the ``Number`` is "A", "R", or "G", the result will be stored in an array (accessed with ``array\[index\]``).
        ===================   ===============  ==============
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

    def balding_nichols_model(self, populations, samples, variants, npartitions,
                              population_dist=None,
                              fst=None,
                              root="bn",
                              seed=None):
        """
        Generate a VariantDataset using the Balding-Nichols model

        :param int populations: Number of populations.

        :param int samples: Number of samples.

        :param int variants: Number of variants.

        :param int npartitions: Number of partitions.

        :param population_dist: Unnormalized population distributed, comma-separated
        :type population_dist: str or None

        :param fst: F_st values, comma-separated
        :type fst: str or None

        :param str root: Annotation path to follow global, sa and va.

        :param seed: Random seed.
        :type seed: int or None

        :rtype: :class:`.VariantDataset`
        """

        pargs = ['baldingnichols', '-k', str(populations), '-n', str(samples), '-m', str(variants), '--npartitions',
                 str(npartitions),
                 '--root', root]
        if population_dist:
            pargs.append('-d')
            pargs.append(population_dist)
        if fst:
            pargs.append('--fst')
            pargs.append(fst)
        if seed:
            pargs.append('--seed')
            pargs.append(seed)

        return self.run_command(None, pargs)

    def stop(self):
        self.sc.stop()
        self.sc = None
