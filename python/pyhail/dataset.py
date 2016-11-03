from pyhail.java import scala_package_object

class VariantDataset:
    def __init__(self, hc, jstate):
        self.hc = hc
        self.jstate = jstate
    
    def aggregate_intervals(input, condition, output):
        pargs = ["aggregateintervals", "-i", input,
                 "-c", condition, "-o", output]
        return self.hc._run_command(self, pargs)

    def annotate_global_expr(condition):
        pargs = ["annotateglobal", "expr", "-c", condition]
        return self.hc._run_command(self, pargs)

    def annotate_global_list(input, root, as_set = False):
        pargs = ["annotateglobal", "list", "-i", input, "-r", root]
        if as_set:
            pargs.append("--as-set")
        return self.hc._run_command(self, pargs)

    def annotate_global_table(input, root):
        pargs = ["annotateglobal", "table", "-i", input, "-r", root]
        return self.hc._run_command(self, pargs)

    def annotate_samples_expr(condition):
        pargs = ["annotatesamples", "expr", "-c", condition]
        return self.hc._run_command(self, pargs)
    
    def annotate_samples_fam(input, quantpheno = False, delimiter = None, root = None, missing = False):
        pargs = ["annotatesamples", "fam", "-i", input]
        if quantpheno:
            pargs.append("--quantpheno")
        if delimiter:
            pargs.append("--delimiter")
            pargs.append(delimiter)
        if root:
            pargs.append("--root")
            pargs.append(root)
        if missing:
            pargs.append(missing)
        return self.hc._run_command(self, pargs)

    def annotate_samples_list(input, root):
        pargs = ["annotateglobal", "table", "-i", input, "-r", root]
        return self.hc._run_command(self, pargs)

    def annotate_samples_table(input, sample_expr, root = None, code = None):
        pargs = ["annotateglobal", "table", "-i", input, "--sample-expr", sample_expr]
        if root:
            pargs.append("--root")
            pargs.append(root)
        if code:
            pargs.append("--code")
            pargs.append(code)
        return self.hc._run_command(self, pargs)

    def annotate_variants_bed(input, root, all = False):
        pargs = ["annotatevariants", "bed", "-i", input, "--root", root]
        if all:
            pargs.append("--all")
        return self.hc._run_command(self, pargs)

    def annotate_variants_expr(self, condition):
        pargs = ["annotatevariants", "expr", "-c", condition]
        return self.hc._run_command(self, pargs)
    
    def annotate_variants_intervals(input, root, all = False):
        pargs = ["annotatevariants", "intervals", "-i", input, "--root", root]
        if all:
            pargs.append("--all")
        return self.hc._run_command(self, pargs)
    
    def annotate_variants_loci(locus_expr, *args, **kwargs):
        pargs = ["annotatevariants", "loci", "--locus-expr", locus_expr]

        root = kwargs.pop('root', None)
        if root:
            pargs.append("--root")
            pargs.append(root)

        code = kwargs.pop('code', None)
        if code:
            pargs.append("--code")
            pargs.append(code)
            
        return self.hc._run_command(self, pargs)

    def annotate_variants_table(variant_expr, *args, **kwargs):
        pargs = ["annotatevariants", "table", "--variant-expr", variant_expr]
        
        root = kwargs.pop('root', None)
        if root:
            pargs.append("--root")
            pargs.append(root)

        code = kwargs.pop('code', None)
        if code:
            pargs.append("--code")
            pargs.append(code)
            
        return self.hc._run_command(self, pargs)

    def annotate_variants_vcf(*args, **kwpargs):
        args = ["annotatevariants", "vcf"]
        
        root = kwargs.pop('root', None)
        if root:
            pargs.append("--root")
            pargs.append(root)

        code = kwargs.pop('code', None)
        if code:
            pargs.append("--code")
            pargs.append(code)
        
        return self.hc._run_command(self, pargs)
    
    def annotate_variants_vds(input, root = None, code = None, split = False):
        pargs = ["annotatevariants", "vds", "-i", input]
        
        if root:
            pargs.append("--root")
            pargs.append(root)

        if code:
            pargs.append("--code")
            pargs.append(code)

        if split:
            pargs.append("--split")
        
        return self.hc._run_command(self, pargs)
    
    def cache():
        pargs = ["cache"]
        return hc._run_command(self, pargs)
    
    def count(self, genotypes = False):
        return (scala_package_object(self.hc.jvm.org.broadinstitute.hail.driver)
                .count(self.jstate.vds(), genotypes)
                .toJavaMap())
    
    def deduplicate():
        pargs = ["deduplicate"]
        return self.hc._run_command(self, pargs)

    def downsample_variants(keep):
        pargs = ["downsamplevariants", "--keep", str(keep)]
        return self.hc._run_command(self, pargs)

    def export_gen(output):
        pargs = ["exportgen", "--output", output]
        return self.hc._run_command(self, pargs)

    def export_genotypes(output, condition, types = None, print_ref = False, print_missing = False):
        pargs = ["exportgenotypes", "--output", output, "-c", condition]
        if types:
            pargs.append("--types")
            pargs.append(types)
        if print_ref:
            pargs.append("--print-ref")
        if print_missing:
            pargs.append("--print-missing")
        return self.hc._run_command(self, pargs)

    def export_plink(output):
        pargs = ["exportplink", "--output", output]
        return self.hc._run_command(self, pargs)
    
    def export_samples(output, condition, types = None):
        pargs = ["exportsamples", "--output", output, "-c", condition]
        if types:
            pargs.append("--types")
            pargs.append(types)
        return self.hc._run_command(self, pargs)

    def export_variants(output, condition, types = None):
        pargs = ["exportvariants", "--output", output, "-c", condition]
        if types:
            pargs.append("--types")
            pargs.append(types)
        return self.hc._run_command(self, pargs)

    def export_variants_solr(variant_condition,
                             genotype_condition ,
                             address,
                             keyspace,
                             table,
                             export_missing = False,
                             export_ref = False):
        pargs = ["exportvariantscass", "-v", variant_condition, "-g", genotype_condition,
                 "-a", address, "-k", keyspace, "-t", table]
        if export_missing:
            pargs.append("--export-missing")
        if export_ref:
            pargs.append("--export-ref")
        return self.hc._run_command(self, pargs)
    
    def export_variants_solr(variant_condition = '',
                             genotype_condition = '',
                             solr_url = None,
                             solr_cloud_collection = None,
                             zookeeper_host = None,
                             drop = False,
                             num_shards = 1,
                             export_missing = False,
                             export_ref = False,
                             block_size = 100):
        pargs = ["exportvariantssolr", "-v", variant_condition, "-g", genotype_condition, "--block-size", block_size]
        if solr_url:
            pargs.append("-u")
            pargs.append(solr_url)
        if solr_cloud_collection:
            pargs.append("-c")
            pargs.append(solr_cloud_collection)
        if zookeeper_host:
            pargs.append("-z")
            pargs.append(zookeeper_host)
        if drop:
            pargs.append("--drop")
        if num_shards:
            pargs.append("--num-shards")
            pargs.append(num_shards)
        if export_missing:
            pargs.append("--export-missing")
        if export_ref:
            pargs.append("--export-ref")
        return self.hc._run_command(self, pargs)
    
    def export_vcf(output, append_to_header = None, export_pp = False):
        pargs = ["exportvcf", "--output", output]
        if append_to_header:
            pargs.append("-a")
            pargs.append(append_to_header)
        if export_pp:
            pargs.append("--export-pp")
        return self.hc._run_command(self, pargs)
    
    def write(self, destination, no_compress = False):
        pargs = ["write", "-o", desetination]
        if no_compress:
            pargs.append("--no-compress")
        return self.hc._run_command(self, pargs)
    
    def export_variants(self, destination, condition, types = None):
        pargs = ["exportvariants", "-o", destination, "-c", condition]
        if types:
            pargs.append("--types")
            pargs.append(types)
        return self.hc._run_command(self, pargs)
    
    def filter_multi():
        pargs = ["filtermulti"]
        return self.hc._run_command(self, pargs)
    
    def filter_samples_all():
        pargs = ["filtersamples", "all"]
        return self.hc._run_command(self, pargs)
    
    def filter_samples_expr(condition):
        pargs = ["filtersamples", "expr", "-c", condition]
        return self.hc._run_command(self, pargs)
    
    def filter_samples_list(input):
        pargs = ["filtersamples", "list", "-i", input]
        return self.hc._run_command(self, pargs)
    
    def filter_variants_all():
        pargs = ["filtervariants", "all"]
        return self.hc._run_command(self, pargs)
    
    def filter_variants_expr(self, condition):
        pargs = ["filtervariants", "expr", "--keep", "-c", condition]
        return self.hc._run_command(self, pargs)
    
    def filter_variants_intervals(input):
        pargs = ["filtervariants", "intervals", "-i", input]
        return self.hc._run_command(self, pargs)
    
    def filter_variants_list(input):
        pargs = ["filtervariants", "list", "-i", input]
        return self.hc._run_command(self, pargs)
    
    def gqbydp(output, plot = False):
        pargs = ["gqbydp", "-o", output]
        if plot:
            pargs.append("--plot")
        return self.hc._run_command(self, pargs)

    def grm(format, output, id_file = None, N_file = None):
        pargs = ["grm", "-f", format, "-o", output]
        if id_file:
            pargs.append("--id-file")
            pargs.append(id_file)
        if N_file:
            pargs.append("--N-file")
            pargs.append(N_file)
        return self.hc._run_command(self, pargs)

    def hardcalls():
        pargs = ["hardcalls"]
        return self.hc._run_command(self, pargs)

    def ibd(output, maf = None, unbounded = False, min = None, max = None):
        pargs = ["ibd", '-o', output]
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
        return self.hc._run_command(self, pargs)

    def imputesex(maf_threshold = None, include_par = False, female_threshold = None, male_threshold = None, pop_freq = None):
        pargs = ["imputesex"]
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
        return self.hc._run_command(self, pargs)
    
    def join(self, right):
        return self.hc.vds_state(
            self.hc.jvm.org.broadinstitute.hail.driver.Join.join(self.jstate.vds(),
                                                                 right.jstate.vds()))
    
    def linreg(y, covariates = None, root = None):
        pargs = ['linreg', '-y', y]
        if covariates:
            pargs.append('-c')
            pargs.append(covariates)
        if root:
            pargs.append('-r')
            pargs.append(root)
        return self.hc._run_command(self, pargs)
        
    def logreg(test, y, covariates = None, root = None):
        pargs = ['logreg', '-t', test, '-y', y]
        if covariates:
            pargs.append('-c')
            pargs.append(covariates)
        if root:
            pargs.append('-r')
            pargs.append(root)
        return self.hc._run_command(self, pargs)

    def mendel_errors(output, fam):
        pargs = ['mendelerrors', '-o', output, '-f', fam]
        return self.hc._run_command(self, pargs)
    
    def pca(output, k = None, loadings = None, eigenvalues = None):
        pargs = ['pca', '-o', output]
        if k:
            pargs.append('-k')
            pargs.append(k)
        if loadings:
            pargs.append('--loadings')
            pargs.append(loadings)
        if eigenvalues:
            pargs.append('--eigenvalues')
            pargs.append(eigenvalues)
        return self.hc._run_command(self, pargs)
    
    def persist():
        pargs = ['persist']
        return self.hc._run_command(self, pargs)
    
    def printschema(output = None, attributes = False, va = False, sa = False, print_global = False):
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
        return self.hc._run_command(self, pargs)

    def renamesamples(input):
        pargs = ['renamesamples', '-i', input]
        return self.hc._run_command(self, pargs)
    
    def repartition(npartition, no_suffle = False):
        pargs = ['repartition', '--partitions', npartition]
        if no_shuffle:
            pargs.append("--no-shuffle")
        return self.hc._run_command(self, pargs)

    def same(self, other):
        self_vds = self.jstate.vds()
        other_vds = other.jstate.vds()
        return self_vds.same(other_vds)
    
    def sample_qc(output = None):
        pargs = ['sampleqc']
        if output:
            pargs.append('-o')
            pargs.append(output)
        return self.hc._run_command(self, pargs)

    def show_globals(output = None):
        pargs = ['showglobals']
        if output:
            pargs.append('-o')
            pargs.append(output)
        return self.hc._run_command(self, pargs)
    
    def split_multi(propagate_gq = False, no_compress = False):
        pargs = ['splitmulti']
        if propagate_gq:
            pargs.append('--propagate-gq')
        if no_compress:
            pargs.append('--no-compress')
        return self.hc._run_command(self, pargs)

    def variant_qc(output = None):
        pargs = ['variantqc']
        if output:
            pargs.append('-o')
            pargs.append(output)
        return self.hc._run_command(self, pargs)

    def vep(config, block_size = None, root = None, force = False):
        pargs = ['vep', '--config', config]
        if block_size:
            pargs.append('--block-size')
            pargs.append(block_size)
        if root:
            pargs.append('--root')
            pargs.append(root)
        if force:
            pargs.append('--force')
        return self.hc._run_command(self, pargs)

    def variantsToPandas(self):
        return pyspark.sql.DataFrame(self.jstate.vds().variantsDF(self.hc.jsqlContext),
                                     self.hc.sqlContext).toPandas()
