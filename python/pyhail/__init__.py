
import pyspark
import inspect

# FIXME:
#  - break apart __init__
#  - docs
#  - testing
#  - catch and translate exceptions (at least fatal)
#  - only have annotate_variants_vds which takes vds object (not path)
#  - what should export/write return?
#  - add sample, genotype context
#  - add hidden commands
#  - same

# FIXME make sure callers are using correct `up'
def caller_posn(hc, up = 1):
    curframe = inspect.currentframe()
    outer_frames = inspect.getouterframes(curframe, 1)
    
    file = outer_frames[up + 1][1]
    try:
        (lines, lineno) = inspect.getsourcelines(outer_frames[up + 1][0])
    except IOError:
        # FIXME
        print('outer_frames', outer_frames)
        lines = [""]
        lineno = 1
    
    return hc.jvm.org.broadinstitute.hail.expr.PythonPosition(
        file,
        lineno,
        lines[0])

def scala_object(java_package, name):
    return getattr(getattr(java_package, name + '$'),
                   'MODULE$')

def to_ast(hc, value):
    if isinstance(value, AST):
        return value
    elif isinstance(value, ASTFieldRef):
        return AST(hc,
                   hc.jvm.org.broadinstitute.hail.expr.Select(
                       value.posn,
                       value.jast,
                       value.name))
    elif type(value) == bool:
        return AST(hc,
                   hc.jvm.org.broadinstitute.hail.expr.Const(
                       caller_posn(hc),
                       value,
                       scala_object(jc.jvm.org.broadinstitute.hail.expr, 'TBoolean')))
    elif type(value) == str:
        return AST(hc,
                   hc.jvm.org.broadinstitute.hail.expr.Const(
                       caller_posn(hc),
                       value,
                       scala_object(jc.jvm.org.broadinstitute.hail.expr, 'TString')))
    elif type(value) == int:
        return AST(hc,
                   hc.jvm.org.broadinstitute.hail.expr.Const(
                       caller_posn(hc),
                       value,
                       scala_object(hc.jvm.org.broadinstitute.hail.expr, 'TLong')))
    elif type(value) == float:
        return AST(hc,
                   hc.jvm.org.broadinstitute.hail.expr.Const(
                       caller_posn(hc),
                       value,
                       scala_object(hc.jvm.org.broadinstitute.hail.expr, 'TDouble')))
    else:
        raise ValueError('cannot convert python to Hail object: ' + repr(value) + ' of type ' + str(type(value)))

class ExprContext:
    def __init__(self, hc, attrs):
        self.hc = hc
        self.attrs = attrs
    
    def __getattr__(self, name):
        if name in self.attrs:
            return AST(self.hc,
                       self.hc.jvm.org.broadinstitute.hail.expr.SymRef(
                           caller_posn(self.hc),
                           name))
        else:
            # FIXME error message
            raise AttributeError('expression context has no attribute: ' + name)

class AST:
    def __init__(self, hc, jast):
        self.hc = hc
        self.jast = jast
        
    def __getattr__(self, name):
        if name[:2] == '__':
            raise AttributeError
        
        return ASTFieldRef(self.hc, caller_posn(self.hc), self.jast, name)
    
    def __getitem__(self, key):
        return ASTFieldRef(self.hc, caller_posn(self.hc), self.jast, key)
    
    def binop(self, name, other):
        return AST(self.hc,
                   self.hc.jvm.org.broadinstitute.hail.expr.BinaryOp(
                       caller_posn(self.hc, up = 2),
                       self.jast,
                       name,
                       to_ast(self.hc, other)))
    
    def __eq__(self, other):
        return self.binop("==", other)

    def __ne__(self, other):
        return self.binop("!=", other)
    
    def __lt__(self, other):
        return self.binop("<", other)

    def __le__(self, other):
        return self.binop("<=", other)

    def __gt__(self, other):
        return self.binop(">", other)
    
    def __lt__(self, other):
        return self.binop(">=", other)
    
    def __add__(self, other):
        return self.binop("+", other)
    
    def __sub__(self, other):
        return self.binop("-", other)

    def __mul__(self, other):
        return self.binop("*", other)
    
    def __div__(self, other):
        return self.binop("/", other)

class ASTFieldRef:
    def __init__(self, hc, posn, jast, name):
        self.hc = hc
        self.posn = posn
        self.jast = jast
        self.name = name
    
    def __call__(self, *args):
        return AST(
            self.hc,
            self.hc.jvm.org.broadinstitute.hail.expr.ApplyMethod(
                caller_posn(self.hc), # . or (?  This is (
                self.jast,
                self.name,
                self.hc.to_jarray(self.hc.jvm.org.broadinstitute.hail.expr.AST,
                                  [to_ast(self.hc, arg) for arg in args])))

class VariantDataset:
    def __init__(self, hc, sstate):
        self.hc = hc
        self.sstate = sstate
    
    def variant_context(self):
        return ExprContext(self.hc, ['global', 'v', 'va', 'gs'])
    
    def sample_context(self):
        return ExprContext(self.hc, ['global', 's', 'sa', 'gs'])
    
    def genotype_context(self):
        return ExprContext(self.hc, ['global', 'v', 'va', 's', 'sa', 'g'])
    
    def aggregate_intervals(interval_list_path, condition, output):
        pargs = ["aggregateintervals", "-i", interval_list_path,
                 "-c", condition, "-o", output]
        return hc._run_command(self, pargs)

    def annotate_global_expr(condition):
        pargs = ["annotateglobal", "expr", "-c", condition]
        return hc._run_command(self, pargs)

    def annotate_global_list(input, root, as_set = False):
        pargs = ["annotateglobal", "list", "-i", input, "-r", root]
        if as_set:
            pargs.append("--as-set")
        return hc._run_command(self, pargs)

    def annotate_global_table(input, root):
        pargs = ["annotateglobal", "table", "-i", input, "-r", root]
        return hc._run_command(self, pargs)

    def annotate_samples_expr(condition):
        pargs = ["annotatesamples", "expr", "-c", condition]
        return hc._run_command(self, pargs)
    
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
        return hc._run_command(self, pargs)

    def annotate_samples_list(input, root):
        pargs = ["annotateglobal", "table", "-i", input, "-r", root]
        return hc._run_command(self, pargs)

    def annotate_samples_table(input, sample_expr, root = None, code = None):
        pargs = ["annotateglobal", "table", "-i", input, "--sample-expr", sample_expr]
        if root:
            pargs.append("--root")
            pargs.append(root)
        if code:
            pargs.append("--code")
            pargs.append(code)
        return hc._run_command(self, pargs)

    def annotate_variants_bed(input, root, all = False):
        pargs = ["annotatevariants", "bed", "-i", input, "--root", root]
        if all:
            pargs.append("--all")
        return hc._run_command(self, pargs)

    def annotate_variants_expr(condition):
        pargs = ["annotatevariants", "expr", "-c", condition]
        return hc._run_command(self, pargs)
    
    def annotate_variants_intervals(input, root, all = False):
        pargs = ["annotatevariants", "intervals", "-i", input, "--root", root]
        if all:
            pargs.append("--all")
        return hc._run_command(self, pargs)
    
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
            
        return hc._run_command(self, pargs)

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
            
        return hc._run_command(self, pargs)

    def annotate_variants_vcf(*args, **kwargs):
        pargs = ["annotatevariants", "vcf"]
        
        root = kwargs.pop('root', None)
        if root:
            pargs.append("--root")
            pargs.append(root)

        code = kwargs.pop('code', None)
        if code:
            pargs.append("--code")
            pargs.append(code)
            
        return hc._run_command(self, pargs)

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
        
        return hc._run_command(self, pargs)
    
    def cache():
        pargs = ["cache"]
        return hc._run_command(self, pargs)
    
    def count(self, genotypes = False):
        vds = self.sstate.vds()
        return (scala_object(self.hc.jvm.org.broadinstitute.hail.driver, 'package')
                .count(vds, genotypes)
                .toJavaMap())
    
    def deduplicate():
        pargs = ["deduplicate"]
        return hc._run_command(self, pargs)

    def downsample_variants(keep):
        pargs = ["downsamplevariants", "--keep", str(keep)]
        return hc._run_command(self, pargs)

    def export_gen(output):
        pargs = ["exportgen", "--output", output]
        return hc._run_command(self, pargs)

    def export_genotypes(output, condition, types = None, print_ref = False, print_missing = False):
        pargs = ["exportgenotypes", "--output", output, "-c", condition]
        if types:
            pargs.append("--types")
            pargs.append(types)
        if print_ref:
            pargs.append("--print-ref")
        if print_missing:
            pargs.append("--print-missing")
        return hc._run_command(self, pargs)

    def export_plink(output):
        pargs = ["exportplink", "--output", output]
        return hc._run_command(self, pargs)
    
    def export_samples(output, condition, types = None):
        pargs = ["exportsamples", "--output", output, "-c", condition]
        if types:
            pargs.append("--types")
            pargs.append(types)
        return hc._run_command(self, pargs)

    def export_variants(output, condition, types = None):
        pargs = ["exportvariants", "--output", output, "-c", condition]
        if types:
            pargs.append("--types")
            pargs.append(types)
        return hc._run_command(self, pargs)

    # FIXME exportvariants{cass, solr}

    def export_vcf(output, append_to_header = None, export_pp = False):
        pargs = ["exportvcf", "--output", output]
        if append_to_header:
            pargs.append("-a")
            pargs.append(append_to_header)
        if export_pp:
            pargs.append("--export-pp")
        return hc._run_command(self, pargs)
    
    def write(self, destination, no_compress = False):
        pargs = ["write", "-o", desetination]
        if no_compress:
            pargs.append("--no-compress")
        return hc._run_command(self, pargs)
    
    def annotate_variants_expr(self, condition):
        pargs = ["annotatevariants", "expr", "-c", condition]
        return hc._run_command(self, pargs)
    
    def export_variants(self, destination, condition, types = None):
        pargs = ["exportvariants", "-o", destination, "-c", condition]
        if types:
            pargs.append("--types")
            pargs.append(types)
        return hc._run_command(self, pargs)
    
    def filter_multi():
        pargs = ["filtermulti"]
        return hc._run_command(self, pargs)
    
    def filter_samples_all():
        pargs = ["filtersamples", "all"]
        return hc._run_command(self, pargs)
    
    def filter_samples_expr(condition):
        pargs = ["filtersamples", "expr", "-c", condition]
        return hc._run_command(self, pargs)
    
    def filter_samples_list(input):
        pargs = ["filtersamples", "list", "-i", input]
        return hc._run_command(self, pargs)
    
    def filter_variants_all():
        pargs = ["filtervariants", "all"]
        return hc._run_command(self, pargs)
    
    def filter_variants_expr(self, condition):
        pargs = ["filtervariants", "expr", "--keep", "-c", condition]
        return hc._run_command(self, pargs)
    
    def py_filter_variants_expr(self, condition):
        vc = self.variant_context()
        
        vds = self.sstate.vds()
        rich = self.hc.jvm.org.broadinstitute.hail.variant.RichVDS(vds)
        return self.hc.vds_state(
            rich.filterVariantsAST(to_ast(self.hc, condition(vc)).jast, True))
    
    def filter_variants_intervals(input):
        pargs = ["filtervariants", "intervals", "-i", input]
        return hc._run_command(self, pargs)
    
    def filter_variants_list(input):
        pargs = ["filtervariants", "list", "-i", input]
        return hc._run_command(self, pargs)
    
    def gqbydp(output, plot = False):
        pargs = ["gqbydp", "-o", output]
        if plot:
            pargs.append("--plot")
        return hc._run_command(self, pargs)

    def grm(format, output, id_file = None, N_file = None):
        pargs = ["grm", "-f", format, "-o", output]
        if id_file:
            pargs.append("--id-file")
            pargs.append(id_file)
        if N_file:
            pargs.append("--N-file")
            pargs.append(N_file)
        return hc._run_command(self, pargs)

    def hardcalls():
        pargs = ["hardcalls"]
        return hc._run_command(self, pargs)

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
        return hc._run_command(self, pargs)

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
        return hc._run_command(self, pargs)

    def linreg(y, covariates = None, root = None):
        pargs = ['linreg', '-y', y]
        if covariates:
            pargs.append('-c')
            pargs.append(covariates)
        if root:
            pargs.append('-r')
            pargs.append(root)
        return hc._run_command(self, pargs)
        
    def logreg(test, y, covariates = None, root = None):
        pargs = ['logreg', '-t', test, '-y', y]
        if covariates:
            pargs.append('-c')
            pargs.append(covariates)
        if root:
            pargs.append('-r')
            pargs.append(root)
        return hc._run_command(self, pargs)

    def mendel_errors(output, fam):
        pargs = ['mendelerrors', '-o', output, '-f', fam]
        return hc._run_command(self, pargs)
    
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
        return hc._run_command(self, pargs)
    
    def persist():
        pargs = ['persist']
        return hc._run_command(self, pargs)
    
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
        return hc._run_command(self, pargs)

    def renamesamples(input):
        pargs = ['renamesamples', '-i', input]
        return hc._run_command(self, pargs)
    
    def repartition(npartition):
        pargs = ['repartition', '--partitions', npartition]
        return hc._run_command(self, pargs)

    def same(self, other):
        self_vds = self.sstate.vds()
        other_vds = other.sstate.vds()
        return self_vds.same(other_vds)
    
    def sample_qc(output = None):
        pargs = ['sampleqc']
        if output:
            pargs.append('-o')
            pargs.append(output)
        return hc._run_command(self, pargs)

    def show_globals(output = None):
        pargs = ['showglobals']
        if output:
            pargs.append('-o')
            pargs.append(output)
        return hc._run_command(self, pargs)
    
    def split_multi(propagate_gq = False, no_compress = False):
        pargs = ['splitmulti']
        if propagate_gq:
            pargs.append('--propagate-gq')
        if no_compress:
            pargs.append('--no-compress')
        return hc._run_command(self, pargs)

    def variant_qc(output = None):
        pargs = ['variantqc']
        if output:
            pargs.append('-o')
            pargs.append(output)
        return hc._run_command(self, pargs)

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
        return hc._run_command(self, pargs)
    
class HailContext:
    def __init__(self, sc):
        self.gateway = sc._gateway
        self.jvm = sc._jvm
        
        logger = sc._jvm.org.apache.log4j
        logger.LogManager.getLogger("org"). setLevel(logger.Level.ERROR)
        logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
        
        self.ssc = sc._jsc.sc()
        self.ssqlContext = sc._jvm.SQLContext(sc._jsc.sc())
        
        sc._jsc.hadoopConfiguration().set('io.compression.codecs', 'org.apache.hadoop.io.compress.DefaultCodec,org.broadinstitute.hail.io.compress.BGzipCodec,org.apache.hadoop.io.compress.GzipCodec')
    
    def to_jarray(self, jtype, plist):
        n = len(plist)
        jarr = self.gateway.new_array(jtype, n)
        for i, s in enumerate(plist):
            jarr[i] = s
        return jarr
    
    def vds_state(self, vds):
        return VariantDataset(self, self.jvm.org.broadinstitute.hail.driver.State(self.ssc, self.ssqlContext, vds))
    
    def initial_state(self):
        return VariantDataset(self, self.jvm.org.broadinstitute.hail.driver.State(self.ssc, self.ssqlContext, None))
    
    def _run_command(self, state, pargs):
        jargs = self.to_jarray(self.jvm.java.lang.String, pargs)
        cmdargs = self.jvm.org.broadinstitute.hail.driver.ToplevelCommands.lookup(jargs)
        cmd = cmdargs._1()
        args = cmdargs._2()
        options = cmd.parseArgs(args)
        result = cmd.run(state.sstate, options)
        return VariantDataset(self, result)
    
    def fam_summary(input, output):
        pargs = ["famsummary", "-f", input, "-o", output]
        return hc._run_command(self, pargs)

    def grep(*args, **kwargs):
        pargs = ["grep"]
        
        max_count = kwargs.pop('max_count', False)
        if max_count:
            pargs.append('--max-count')
            pargs.append(str(max_count))
        
        return hc._run_command(self, pargs)
    
    def import_annotations_table(self, *args, **kwargs):
        pargs = ["importannotationstable"]
        
        variant_expr = kwargs.pop('variant_expr')
        pargs.append('--variant-expr')
        pargs.append(variant_expr)

        code = kwargs.pop('code', False)
        if code:
            pargs.append('--code')
            pargs.append(code)

        npartition = kwargs.pop('npartition', None)
        if npartition:
            pargs.append('--npartition')
            pargs.append(npartition)
        
        for arg in args:
            pargs.append(arg)
        return self._run_command(self.initial_state(), pargs)

    def import_bgen(self, *args, **kwargs):
        pargs = ["importbgen"]

        no_compress = kwargs.pop('no_compress', False)
        if no_compress:
            pargs.append('--no-compress')
        
        samplefile = kwargs.pop('samplefile', None)
        if samplefile:
            pargs.append('--samplefile')
            pargs.append(samplefile)
        
        npartition = kwargs.pop('npartition', None)
        if npartition:
            pargs.append('--npartition')
            pargs.append(npartition)

        tolerance = kwargs.pop('tolerance', None)
        if tolerance:
            pargs.append('--tolerance')
            pargs.append(npartition)
            
        return self._run_command(self.initial_state(), pargs)

    def import_gen(self, *args, **kwargs):
        pargs = ["importgen"]

        no_compress = kwargs.pop('no_compress', False)
        if no_compress:
            pargs.append('--no-compress')
        
        samplefile = kwargs.pop('samplefile', None)
        if samplefile:
            pargs.append('--samplefile')
            pargs.append(samplefile)
        
        chromosome = kwargs.pop('chromosome', None)
        if chromosome:
            pargs.append('--chromosome')
            pargs.append(npartition)

        npartition = kwargs.pop('npartition', None)
        if npartition:
            pargs.append('--npartition')
            pargs.append(npartition)
        
        tolerance = kwargs.pop('tolerance', None)
        if tolerance:
            pargs.append('--tolerance')
            pargs.append(npartition)
            
        return self._run_command(self.initial_state(), pargs)

    def import_plink(self, *args, **kwargs):
        pargs = ["importplink"]

        bfile = kwargs.pop('bfile', False)
        if bfile:
            pargs.append('--bfile')
            pargs.append(bfile)

        bed = kwargs.pop('bed', False)
        if bed:
            pargs.append('--bed')
            pargs.append(bed)

        bim = kwargs.pop('bim', False)
        if bim:
            pargs.append('--bim')
            pargs.append(bim)

        fam = kwargs.pop('fam', False)
        if fam:
            pargs.append('--fam')
            pargs.append(fam)
            
        npartition = kwargs.pop('npartition', None)
        if npartition:
            pargs.append('--npartition')
            pargs.append(npartition)
        
        tolerance = kwargs.pop('tolerance', None)
        if tolerance:
            pargs.append('--tolerance')
            pargs.append(npartition)

        quantpheno = kwargs.pop('quantpheno', False)
        if quantpheno:
            pargs.append('--quantpheno')

        missing = kwargs.pop('missing', None)
        if missing:
            pargs.append('--missing')
            pargs.append(missing)

        delimiter = kwargs.pop('delimiter', None)
        if delimiter:
            pargs.append('--delimiter')
            pargs.append(delimiter)
        
        return self._run_command(self.initial_state(), pargs)
    
    def read(self, vds_path, skip_genotypes = False):
        pargs = ["read"]
        if (skip_genotypes):
            pargs.append("--skip-genotypes")
        pargs.append("-i")
        pargs.append(vds_path)
        return self._run_command(self.initial_state(), pargs)

    def import_vcf(self, *args, **kwargs):
        pargs = ["importvcf"]
        no_compress = kwargs.pop('no_compress', False)
        if no_compress:
            pargs.append('--no-compress')
        
        force = kwargs.pop('force', False)
        if force:
            pargs.append('--force')
            
        force_bgz = kwargs.pop('force_bgz', False)
        if force_bgz:
            pargs.append('--force-bgz')
            
        header_file = kwargs.pop('header_file', None)
        if header_file:
            pargs.append('--header-file')
            pargs.append(header_file)
            
        npartition = kwargs.pop('npartition', None)
        if npartition:
            pargs.append('--npartition')
            pargs.append(str(n_partitions))
            
        pp_as_pl = kwargs.pop('pp_as_pl', False)
        if pp_as_pl:
            pargs.append('--pp-as-pl')
            
        skip_bad_ad = kwargs.pop('skip_bad_ad', False)
        if skip_bad_ad:
            pargs.append('--skip-bad-ad')
            
        skip_genotypes = kwargs.pop('skip_genotypes', False)
        if skip_genotypes:
            pargs.append('--skip-genotypes')

        store_gq = kwargs.pop('skip_genotypes', False)
        if store_gq:
            pargs.append('--store-gq')
            
        for arg in args:
            pargs.append(arg)

        return self._run_command(self.initial_state(), pargs)

    def index_bgen(*args):
        pargs = ["indexbgen"]
        for arg in args:
            pargs.append(arg)
        return self._run_command(self.initial_state(), pargs)
