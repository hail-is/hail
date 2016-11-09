import pyspark

from pyhail.dataset import VariantDataset
from pyhail.java import jarray, scala_object

class HailContext(object):
    """:class:`.HailContext` is the main entrypoint for PyHail
    functionality.

    :param SparkContext sc: The pyspark context.
    """

    def __init__(self, sc):
        self.sc = sc

        self.gateway = sc._gateway
        self.jvm = sc._jvm

        # sc._jsc is JavaObject JavaSparkContext
        self.jsc = sc._jsc.sc()

        self.jsql_context = sc._jvm.SQLContext(self.jsc)

        self.sql_context = pyspark.sql.SQLContext(sc, self.jsql_context)

        self.jsc.hadoopConfiguration().set(
            'io.compression.codecs',
            'org.apache.hadoop.io.compress.DefaultCodec,org.broadinstitute.hail.io.compress.BGzipCodec,org.apache.hadoop.io.compress.GzipCodec')

        logger = sc._jvm.org.apache.log4j
        logger.LogManager.getLogger("org"). setLevel(logger.Level.ERROR)
        logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)

    def _jstate(self, jvds):
        return self.jvm.org.broadinstitute.hail.driver.State(
            self.jsc, self.jsql_context, jvds, scala_object(self.jvm.scala.collection.immutable, 'Map').empty())

    def run_command(self, vds, pargs):
        jargs = jarray(self.gateway, self.jvm.java.lang.String, pargs)
        t = self.jvm.org.broadinstitute.hail.driver.ToplevelCommands.lookup(jargs)
        cmd = t._1()
        cmd_args = t._2()
        result = cmd.run(self._jstate(vds.jvds if vds != None else None),
                         cmd_args)
        return VariantDataset(self, result.vds())

    def fam_summary(self, input, output):
        """Outputs summary of a .fam file.

        :param str input: Input .fam file.

        :param str output: Output summary file.

        :return: Nothing.

        """
        pargs = ["famsummary", "-f", input, "-o", output]
        self.run_command(self, pargs)

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

        self.run_command(self, pargs)

    def import_annotations_table(self, path, variant_expr, code=None, npartitions=None,
                                 # text table options
                                 types=None, missing="NA", delimiter="\\t", comment=None,
                                 header=True, impute=False):
        """Import variants and variant annotaitons from a delimited text file
        (text table) as a sites-only VariantDataset.

        :param path: The files to import.
        :type path: str or list of str

        :param str variant_expr: Expression to construct a variant
            from a row of the text table.  Must have type Variant.

        :param code: Expression to build the variant annotations.
        :type code: str or None

        :param npartitions: Number of partitions.
        :type npartitions: int or None

        :param str types: Type declarations for the fields of the text
            table.

        :param str missing: The string used to denote missing values.

        :param str delimiter: Field delimiter regex.

        :param comment: Skip lines starting with the given regex.
        :type comment: str or None

        :param bool header: If True, the first line is treated as the
            header line.  If False, the columns are named _0, _1, ...,
            _N (0-indexed).

        :param bool impute: If True, impute column types.

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

        if types:
            pargs.append('--types')
            pargs.append(types)

        pargs.append('--missing')
        pargs.append(missing)

        pargs.append('--delimiter')
        pargs.append(delimiter)

        if comment:
            pargs.append('--comment')
            pargs.append(comment)

        if not header:
            pargs.append('--no-header')
        if impute:
            pargs.append('--impute')

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

    def import_plink(self, bed, bim, fam, npartitions=None, delimiter='\\\\s+', missing="NA", quantpheno=False):
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

        pargs = ['baldingnichols', '-k', populations, '-n', samples, '-m', variants, '--npartitions', npartitions,
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
