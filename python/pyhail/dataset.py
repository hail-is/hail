from pyhail.java import scala_package_object
from pyhail.keytable import KeyTable

import pyspark

class VariantDataset(object):
    def __init__(self, hc, jvds):
        self.hc = hc
        self.jvds = jvds

    def aggregate_by_key(self, key_code = None, agg_code = None):
        """Aggregate by user-defined key and aggregation expressions

        :param str key_code: Named expression for which fields are keys

        :param str agg_code: Named aggregation expression.

        :rtype: :class`.KeyTable`
        """
        return KeyTable(self.hc, self.jvds.aggregateByKey(key_cond, agg_condition))

    def aggregate_intervals(self, input, condition, output):
        """Aggregate over intervals and export.

        :param str input: Input interval list file.

        :param str condition: Aggregation expression.

        :param str output: Output file.

        """

        pargs = ['aggregateintervals', '-i', input, '-c', condition, '-o', output]
        return self.hc.run_command(self, pargs)

    def annotate_global_expr(self, condition):
        """Update the global annotations with expression.

        :param str condition: Annotation expression.

        """

        pargs = ['annotateglobal', 'expr', '-c', condition]
        return self.hc.run_command(self, pargs)

    def annotate_global_list(self, input, root, as_set=False):
        """Load text file into global annotations as Array[String] or
        Set[String].

        :param str input: Input text file.

        :param str root: Global annotation path to store text file.

        :param bool as_set: If True, load text file as Set[String],
            otherwise, load as Array[String].

        """

        pargs = ['annotateglobal', 'list', '-i', input, '-r', root]
        if as_set:
            pargs.append('--as-set')
        return self.hc.run_command(self, pargs)

    def annotate_global_table(self, input, root, impute=False):
        """Load delimited text file (text table) into global annotations as
        Array[Struct].

        :param str input: Input text file.

        :param str root: Global annotation path to store text table.

        :param str impute: Impute column types from the file.

        """

        pargs = ['annotateglobal', 'table', '-i', input, '-r', root]
        if impute:
            pargs.append('--impute')
        return self.hc.run_command(self, pargs)

    def annotate_samples_expr(self, condition):
        """Annotate samples with expression.

        :param str condition: Annotation expression.

        """

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
        """Annotate samples with a Boolean indicating presence/absence in a
        list of samples in a text file.

        :param str input: Sample list file.

        :param str root: Sample annotation path to store Boolean.

        """

        pargs = ['annotatesamples', 'table', '-i', input, '-r', root]
        return self.hc.run_command(self, pargs)

    def annotate_samples_table(self, input, sample_expr, root=None, code=None, impute=False):
        """Annotate samples with delimited text file (text table).

        :param str input: Path to delimited text file.

        :param str sample_expr: Expression for sample id (key).

        :param str root: Sample annotation path to store text table.

        :param str code: Annotation expression.

        :param str impute: Impute column types from the file.

        """

        pargs = ['annotatesamples', 'table', '-i', input, '--sample-expr', sample_expr]
        if root:
            pargs.append('--root')
            pargs.append(root)
        if code:
            pargs.append('--code')
            pargs.append(code)
        if impute:
            pargs.append('--impute')
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

        :param str input: Path to .bed file.

        :param str root: Variant annotation path to store annotation.

        :param bool all: If true, store values from all overlapping
            intervals as a set.

        """

        pargs = ['annotatevariants', 'bed', '-i', input, '--root', root]
        if all:
            pargs.append('--all')
        return self.hc.run_command(self, pargs)

    def annotate_variants_expr(self, condition):
        """Annotate variants with expression.

        :param str condition: Annotation expression.

        """
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

    def annotate_variants_loci(self, path, locus_expr, root=None, code=None, impute=False):
        """Annotate variants from an delimited text file (text table) indexed
        by loci.

        :param str input: Path to delimited text file.

        :param str locus_expr: Expression for locus (key).

        :param str root: Variant annotation path to store annotation.

        :param str code: Annotation expression.

        :param str impute: Impute column types from the file.

        """

        pargs = ['annotatevariants', 'loci', '--locus-expr', locus_expr]

        if root:
            pargs.append('--root')
            pargs.append(root)

        if code:
            pargs.append('--code')
            pargs.append(code)

        if impute:
            pargs.append('--impute')

        if isinstance(path, str):
            pargs.append(path)
        else:
            for p in path:
                pargs.append(p)

        return self.hc.run_command(self, pargs)

    def annotate_variants_table(self, path, variant_expr, root=None, code=None, impute=False):
        """Annotate variant with delimited text file (text table).

        :param path: Path to delimited text files.
        :type path: str or list of str

        :param str variant_expr: Expression for Variant (key).

        :param str root: Variant annotation path to store text table.

        :param str code: Annotation expression.

        :param str impute: Impute column types from the file.

        """

        pargs = ['annotatevariants', 'table', '--variant-expr', variant_expr]

        if root:
            pargs.append('--root')
            pargs.append(root)

        if code:
            pargs.append('--code')
            pargs.append(code)

        if impute:
            pargs.append('--impute')

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
        return (VariantDataset(self.hc, result._1),
                VariantDataset(self.hc, result._2))

    def count(self, genotypes=False):
        """Return number of samples, varaints and genotypes.

        :param bool genotypes: If True, return number of called
            genotypes and genotype call rate.

        """

        return (scala_package_object(self.hc.jvm.org.broadinstitute.hail.driver)
                .count(self.jvds, genotypes)
                .toJavaMap())

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
        """Export genotype information (variant- and sample-index) information
        to delimited text file.

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

    def export_plink(self, output):
        """Export as PLINK .bed/.bim/.fam

        :param str output: Output file base.  Will write .bed, .bim and .fam files.

        """

        pargs = ['exportplink', '--output', output]
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

    def filter_samples_expr(self, condition):
        """Filter samples based on expression.

        :param str condition: Expression for filter condition.

        """

        pargs = ['filtersamples', 'expr', '--keep', '-c', condition]
        return self.hc.run_command(self, pargs)

    def filter_samples_list(self, input):
        """Filter samples with a sample list file.

        :param str input: Path to sample list file.

        """

        pargs = ['filtersamples', 'list', '--keep', '-i', input]
        return self.hc.run_command(self, pargs)

    def filter_variants_all(self):
        """Discard all variants, variant annotations and genotypes."""

        pargs = ['filtervariants', 'all']
        return self.hc.run_command(self, pargs)

    def filter_variants_expr(self, condition):
        """Filter samples based on expression.

        :param str condition: Expression for filter condition.

        """

        pargs = ['filtervariants', 'expr', '--keep', '-c', condition]
        return self.hc.run_command(self, pargs)

    def filter_variants_intervals(self, input):
        """Filter variants with an .interval_list file.

        :param str input: Path to .interval_list file.

        """

        pargs = ['filtervariants', 'intervals', '--keep', '-i', input]
        return self.hc.run_command(self, pargs)

    def filter_variants_list(self, input):
        """Filter variants with a list of variants.

        :param str input: Path to variant list file.

        """

        pargs = ['filtervariants', 'list', '-i', input]
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

        :param str output: Output .tsv file for IBD matrix.

        :param maf: Expression for the minor allele frequency.
        :type maf: str or None

        :param bool unbounded: Allows the estimations for Z0, Z1, Z2,
            and PI_HAT to take on biologically-nonsense values
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

    def imputesex(self, maf_threshold=0.0, include_par=False, female_threshold=0.2, male_threshold=0.8, pop_freq=None):
        """Impute sex of samples by calculating inbreeding coefficient on the
        X chromosome.

        :param float maf_threshold: Minimum minor allele frequency threshold.

        :param bool include_par: Include pseudoautosomal regions.

        :param float female_threshold: Samples are called females if F < femaleThreshold

        :param float male_threshold: Samples are called males if F > maleThreshold

        :param Variant annotation for estimate of MAF.  If None, MAF
            will be computed.

        """

        pargs = ['imputesex']
        if maf_threshold:
            pargs.append('--maf-threshold')
            pargs.append(maf_threshold)
        if include_par:
            pargs.append('--include_par')
        if female_threshold:
            pargs.append('--female-threshold')
            pargs.append(female_threshold)
        if male_threshold:
            pargs.append('--male-threshold')
            pargs.append(male_threshold)
        if pop_freq:
            pargs.append('--pop-freq')
            pargs.append(pop_freq)
        return self.hc.run_command(self, pargs)

    def join(self, right):
        """Join datasets, inner join on variants, concatenate samples, variant
        and global annotations from self.

        """

        return VariantDataset(
            self.hc,
            self.hc.jvm.org.broadinstitute.hail.driver.Join.join(self.jvds,
                                                                 right.jvds))

    def linreg(self, y, covariates="", root="va.linreg", minac=1, minaf=0.0):
        """Test each variant for association using the linear regression
        model.

        :param str y: Response sample annotation.

        :param str covariates: Covariant sample annotations, comma separated.

        :param str root: Variant annotation path to store result of linear regression.

        :param float minac: Minimum alternate allele count.

        :param float minaf: Minimum alternate allele frequency.

        """

        pargs = ['linreg', '-y', y, '-c', covariates, '-r', root, '--mac', str(minac), '--maf', str(minaf)]
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

    def pca(self, output, scores, loadings=None, eigenvalues=None, k=10, arrays=False):
        """Run Principal Component Analysis (PCA) on the matrix of genotypes.

        :param str scores: Sample annotation path to store scores.

        :param loadings: Variant annotation path to store site loadings
        :type loadings: str or None

        :param eigenvalues: Global annotation path to store eigenvalues.
        :type eigenvalues: str or None

        """

        pargs = ['pca', '-o', output, '--scores', scores, '-k', k]
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

    def printschema(self, output=None, attributes=False, va=False, sa=False, print_global=False):
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

    def renamesamples(self, input):
        """Rename samples.

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

        return self.jvds.same(other.jvds)

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

    def split_multi(self, propagate_gq=False):
        """Split multi-allelic variants.

        :param bool propagate_gq: Propagate GQ instead of computing from (split) PL.

        """

        pargs = ['splitmulti']
        if propagate_gq:
            pargs.append('--propagate-gq')
        return self.hc.run_command(self, pargs)

    def tdt(self, fam, root):
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
        """Compute per-variant QC metrics."""

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

    def variants_to_pandas(self):
        """Convert variants and variant annotations to Pandas dataframe."""

        return pyspark.sql.DataFrame(self.jvds.variantsDF(self.hc.jsql_context),
                                     self.hc.sql_context).toPandas()
