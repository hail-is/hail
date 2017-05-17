### Doctests extracted from Hail v0.1 (cf0e0137)

# hail.HailContext

import os, shutil
from hail import *
from hail.representation import *
from hail.expr import *
from hail.stats import *

if not os.path.isdir("output/"):
    os.mkdir("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

hc = HailContext(log="output/hail.log", quiet=True)
vds = hc.read('data/example.vds')

vds = hc.balding_nichols_model(3, 100, 1000)

from hail.stats import TruncatedBetaDist

vds = hc.balding_nichols_model(4, 40, 150, 10,
                               pop_dist=[0.1, 0.2, 0.3, 0.4],
                               fst=[.02, .06, .04, .12],
                               af_dist=TruncatedBetaDist(a=0.01, b=2.0, minVal=0.05, maxVal=1.0),
                               seed=1)

hc.grep('hello','data/file.txt')

hc.grep('\d', ['data/file1.txt','data/file2.txt'])

vds = hc.import_bgen("data/example3.bgen", sample_file="data/example3.sample")

(hc.import_gen('data/example.gen', sample_file='data/example.sample')
   .write('output/gen_example1.vds'))

(hc.import_gen('data/example.chr*.gen', sample_file='data/example.sample')
   .write('output/gen_example2.vds'))

vds = hc.import_plink(bed="data/test.bed",
                      bim="data/test.bim",
                      fam="data/test.fam")

table = hc.import_table('data/samples2.tsv', delimiter=',', missing='.')

annotations = (hc.import_table('data/samples3.tsv', no_header=True)
                  .annotate('sample = f0.split("_")[1]')
                  .key_by('sample'))

vds = hc.import_vcf('data/example2.vcf.bgz')

pass_vds = vds.filter_variants_expr('va.filters.isEmpty()', keep=True)

hc.index_bgen("data/example3.bgen")

import shutil, os

hc.stop()

if os.path.isdir("output/"):
    shutil.rmtree("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

# types

import os, shutil
from hail import *
from hail.representation import *
from hail.expr import *
from hail.stats import *

if not os.path.isdir("output/"):
    os.mkdir("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

hc = HailContext(log="output/hail.log", quiet=True)
vds = hc.read('data/example.vds')

vds = hc.read("data/example.vds").annotate_variants_expr('va.genes = ["ACBD", "DCBA"]')

vds_result = vds.annotate_variants_expr('va.AC = gs.map(g => g.oneHotAlleles(v)).sum()')

vds_result = vds.annotate_variants_expr('va.gqHist = gs.map(g => g.gq).hist(0, 100, 20)')

gq_hist = vds.query_genotypes('gs.map(g => g.gq).hist(0, 100, 100)')

vds_result = vds.annotate_variants_expr('va.gqMean = gs.map(g => g.gq).stats().mean')

[singleton_stats] = (vds.sample_qc()
    .query_samples(['samples.map(s => sa.qc.nSingleton).stats()']))

gq_dp = [
'va.homrefGQ = gs.filter(g => g.isHomRef()).map(g => g.gq).stats()',
'va.hetGQ = gs.filter(g => g.isHet()).map(g => g.gq).stats()',
'va.homvarGQ = gs.filter(g => g.isHomVar()).map(g => g.gq).stats()',
'va.homrefDP = gs.filter(g => g.isHomRef()).map(g => g.dp).stats()',
'va.hetDP = gs.filter(g => g.isHet()).map(g => g.dp).stats()',
'va.homvarDP = gs.filter(g => g.isHomVar()).map(g => g.dp).stats()']

vds_result = vds.annotate_variants_expr(gq_dp)

pheno_stats = [
  'va.case_stats = gs.filter(g => sa.pheno.isCase).callStats(g => v)',
  'va.control_stats = gs.filter(g => !sa.pheno.isCase).callStats(g => v)']

vds_result = vds.annotate_variants_expr(pheno_stats)

vds_result = vds.annotate_variants_expr([
  'va.hweCase = gs.filter(g => sa.pheno.isCase).hardyWeinberg()',
  'va.hweControl = gs.filter(g => !sa.pheno.isCase).hardyWeinberg()'])

vds_result = (vds.variant_qc()
    .annotate_samples_expr('sa.inbreeding = gs.inbreeding(g => va.qc.AF)'))

vds_result = (vds.variant_qc()
    .filter_variants_expr('va.qc.AC > 1 && va.qc.AF >= 1e-8 && va.qc.nCalled * 2 - va.qc.AC > 1 && va.qc.AF <= 1 - 1e-8 && v.isAutosomal()')
    .annotate_samples_expr('sa.inbreeding = gs.inbreeding(g => va.qc.AF)'))

(hc.import_gen("data/example.gen", "data/example.sample")
   .annotate_variants_expr('va.infoScore = gs.infoScore()'))

vds_result = (hc.import_gen("data/example.gen", "data/example.sample")
   .annotate_samples_expr("sa.isCase = pcoin(0.5)")
   .annotate_variants_expr(["va.infoScore.case = gs.filter(g => sa.isCase).infoScore()",
                            "va.infoScore.control = gs.filter(g => !sa.isCase).infoScore()"]))

vds_result = vds.annotate_variants_expr('va.hetSamples = gs.filter(g => g.isHet()).map(g => s).collect()')

vds_result = vds.annotate_variants_expr('va.nHets = gs.filter(g => g.isHet()).count()')

[indels_per_chr] = vds.query_variants(['variants.filter(v => v.altAllele().isIndel()).map(v => v.contig).counter()'])

[counter] = vds.query_variants(['variants.flatMap(v => v.altAlleles).counter()'])

from collections import Counter

counter = Counter(counter)

print(counter.most_common(5))

vds_result = vds.annotate_variants_expr("va.hweCase = gs.filter(g => sa.isCase).hardyWeinberg()")

vds_result = vds.annotate_samples_expr('sa.lof_genes = gs.filter(g => va.consequence == "LOF" && g.nNonRefAlleles() > 0).flatMap(g => va.genes.toSet()).collect()')

vds_result = vds.annotate_samples_expr('sa.lof_genes = gs.filter(g => va.consequence == "LOF" && g.nNonRefAlleles() > 0).flatMap(g => va.genes).collect()')

vds_result = vds.filter_variants_expr('gs.fraction(g => g.isCalled()) > 0.90')

exprs = ['sa.SNPmissingness = gs.filter(g => v.altAllele().isSNP()).fraction(g => g.isNotCalled())',
         'sa.indelmissingness = gs.filter(g => v.altAllele().isIndel()).fraction(g => g.isNotCalled())']

vds_result = vds.annotate_samples_expr(exprs)

vds_result = vds.annotate_variants_expr("va.gqStats = gs.map(g => g.gq).stats()")

vds_result = vds.annotate_variants_expr("va.nonRefSamples = gs.filter(g => g.nNonRefAlleles() > 0).map(g => s).take(5)")

samplesMostSingletons = (vds
  .sample_qc()
  .query_samples('samples.takeBy(s => sa.qc.nSingleton, 10)'))

samplesMostSingletons = (vds
  .sample_qc()
  .query_samples('samples.takeBy(s => sa.qc.nSingleton, 10)'))

import shutil, os

hc.stop()

if os.path.isdir("output/"):
    shutil.rmtree("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

# functions

import os, shutil
from hail import *
from hail.representation import *
from hail.expr import *
from hail.stats import *

if not os.path.isdir("output/"):
    os.mkdir("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

hc = HailContext(log="output/hail.log", quiet=True)
vds = hc.read('data/example.vds')

vds = hc.read("data/example.vds").annotate_variants_expr('va.genes = ["ACBD", "DCBA"]')

(vds.annotate_variants_expr(
  'va.fet = let macCase = gs.filter(g => sa.pheno.isCase).map(g => g.nNonRefAlleles()).sum() and '
  'macControl = gs.filter(g => !sa.pheno.isCase).map(g => g.nNonRefAlleles()).sum() and '
  'majCase = gs.filter(g => sa.pheno.isCase).map(g => 2 - g.nNonRefAlleles()).sum() and '
  'majControl = gs.filter(g => !sa.pheno.isCase).map(g => 2 - g.nNonRefAlleles()).sum() in '
  'fet(macCase, macControl, majCase, majControl)'))

(vds.annotate_variants_expr('va.hwe = '
    'let nHomRef = gs.filter(g => g.isHomRef()).count().toInt() and '
    'nHet = gs.filter(g => g.isHet()).count().toInt() and '
    'nHomVar = gs.filter(g => g.isHomVar()).count().toInt() in '
    'hwe(nHomRef, nHet, nHomVar)'))

import shutil, os

hc.stop()

if os.path.isdir("output/"):
    shutil.rmtree("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

# hail.KeyTable

import os, shutil
from hail import *
from hail.representation import *
from hail.expr import *
from hail.stats import *

if not os.path.isdir("output/"):
    os.mkdir("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

hc = HailContext(log="output/hail.log", quiet=True)
vds = hc.read('data/example.vds')

kt1 = hc.import_table('data/kt_example1.tsv', impute=True)

kt2 = hc.import_table('data/kt_example2.tsv', impute=True)

kt_ht_by_sex = kt1.aggregate_by_key("SEX = SEX", "MEAN_HT = HT.stats().mean")

kt_result = kt1.annotate("Y = 5 * X")

kt1.columns

kt1.count()

if kt1.exists("C1 == 5"):
    print("At least one row has C1 equal 5.")

kt3 = hc.import_table('data/kt_example3.tsv', impute=True,
                      types={'c1': TString(), 'c2': TArray(TInt()), 'c3': TArray(TArray(TInt()))})

kt3.explode('c2')

kt3.explode(['c2', 'c3', 'c3'])

(kt1.rename({'HT' : 'Height'})
    .export("output/kt1_renamed.tsv"))

kt_result = kt1.filter("C1 == 5")

kt_result = kt1.filter("C1 == 10", keep=False)

kt_result = kt3.flatten()

if kt1.forall("C1 == 5"):
    print("All rows have C1 equal 5.")

bed = KeyTable.import_bed('data/file1.bed')

vds_result = vds.annotate_variants_table(bed, root='va.cnvRegion')

bed = KeyTable.import_bed('data/file2.bed')

vds_result = vds.annotate_variants_table(bed, root='va.cnvID')

fam_kt = KeyTable.import_fam('data/myStudy.fam')

fam_kt = KeyTable.import_fam('data/myStudy.fam', quantitative=True)

intervals = KeyTable.import_interval_list('data/capture_intervals.txt')

kt_result = kt1.key_by('ID').join(kt2.key_by('ID'))

kt1.key

kt_result = kt1.key_by(['C2', 'C3'])

kt_result = kt1.key_by('C2')

kt_result = kt1.key_by([])

kt1.num_columns

mean_value = kt1.query('C1.stats().mean')

[hist, counter] = kt1.query(['HT.hist(50, 80, 10)', 'SEX.counter()'])

fraction_tall_male = kt1.query('HT.filter(x => SEX == "M").fraction(x => x > 70)')

ids = kt1.query('ID.filter(x => C2 < C3).collect()')

mean_value, t = kt1.query_typed('C1.stats().mean')

[hist, counter], [t1, t2] = kt1.query_typed(['HT.hist(50, 80, 10)', 'SEX.counter()'])

kt2.rename(['newColumn1', 'newColumn2', 'newColumn3'])

kt2.rename({'A' : 'C1'})

if kt1.same(kt2):
    print("KeyTables are the same!")

print(kt1.schema)

kt_result = kt1.select(['C1'])

kt_result = kt1.select(['C3', 'C1', 'C2'])

kt_result = kt1.select([])

kt1.write('output/kt1.kt')

import shutil, os

hc.stop()

if os.path.isdir("output/"):
    shutil.rmtree("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

# representation/hail.representation.Interval

import os, shutil
from hail import *
from hail.representation import *
from hail.expr import *
from hail.stats import *

if not os.path.isdir("output/"):
    os.mkdir("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

hc = HailContext(log="output/hail.log", quiet=True)
vds = hc.read('data/example.vds')

interval1 = Interval.parse('X:100005-X:150020')
interval2 = Interval.parse('16:29500000-30200000')

interval1.overlaps(interval2)

interval1.contains(interval2.start) or interval2.contains(interval1.start)

interval_1 = Interval.parse('X:100005-X:150020')

interval_2 = Interval.parse('16:29500000-30200000')

interval_3 = Interval.parse('16:29.5M-30.2M')  # same as interval_2

interval_4 = Interval.parse('16:30000000-END')

interval_5 = Interval.parse('16:30M-END')  # same as interval_4

interval_6 = Interval.parse('1-22')  # autosomes

interval_7 = Interval.parse('X')  # all of chromosome X

import shutil, os

hc.stop()

if os.path.isdir("output/"):
    shutil.rmtree("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

# utils/index

import os, shutil
from hail import *
from hail.representation import *
from hail.expr import *
from hail.stats import *

if not os.path.isdir("output/"):
    os.mkdir("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

hc = HailContext(log="output/hail.log", quiet=True)
vds = hc.read('data/example.vds')

import shutil, os

hc.stop()

if os.path.isdir("output/"):
    shutil.rmtree("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

# representation/hail.representation.Call

import os, shutil
from hail import *
from hail.representation import *
from hail.expr import *
from hail.stats import *

if not os.path.isdir("output/"):
    os.mkdir("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

hc = HailContext(log="output/hail.log", quiet=True)
vds = hc.read('data/example.vds')

num_alleles = 2
hom_ref = Call(0)
het = Call(1)
hom_var = Call(2)

hom_ref.one_hot_alleles(num_alleles) == [2, 0]
het.one_hot_alleles(num_alleles) == [1, 1]
hom_var.one_hot_alleles(num_alleles) == [0, 2]

num_genotypes = 3
hom_ref = Call(0)
het = Call(1)
hom_var = Call(2)

hom_ref.one_hot_genotype(num_genotypes) == [1, 0, 0]
het.one_hot_genotype(num_genotypes) == [0, 1, 0]
hom_var.one_hot_genotype(num_genotypes) == [0, 0, 1]

import shutil, os

hc.stop()

if os.path.isdir("output/"):
    shutil.rmtree("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

# representation/hail.representation.Variant

import os, shutil
from hail import *
from hail.representation import *
from hail.expr import *
from hail.stats import *

if not os.path.isdir("output/"):
    os.mkdir("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

hc = HailContext(log="output/hail.log", quiet=True)
vds = hc.read('data/example.vds')

v_biallelic = Variant.parse('16:20012:A:TT')
v_multiallelic = Variant.parse('16:12311:T:C,TTT,A')

v_multiallelic.ref == v_multiallelic.allele(0)

v_biallelic.alt == v_biallelic.allele(1)

v_biallelic = Variant.parse('16:20012:A:TT')

v_multiallelic = Variant.parse('16:12311:T:C,TTT,A')

import shutil, os

hc.stop()

if os.path.isdir("output/"):
    shutil.rmtree("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

# representation/hail.representation.Struct

import os, shutil
from hail import *
from hail.representation import *
from hail.expr import *
from hail.stats import *

if not os.path.isdir("output/"):
    os.mkdir("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

hc = HailContext(log="output/hail.log", quiet=True)
vds = hc.read('data/example.vds')

bar = Struct({'foo': 5, '1kg': 10})

bar.foo

bar['foo']

getattr(bar, '1kg')

bar['1kg']

import shutil, os

hc.stop()

if os.path.isdir("output/"):
    shutil.rmtree("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

# getting_started

import os, shutil
from hail import *
from hail.representation import *
from hail.expr import *
from hail.stats import *

if not os.path.isdir("output/"):
    os.mkdir("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

hc = HailContext(log="output/hail.log", quiet=True)
vds = hc.read('data/example.vds')

hc.stop()

from hail import *

hc = HailContext()

hc.import_vcf('../../../src/test/resources/sample.vcf').write('sample.vds')

vds = (hc.read('sample.vds')
    .split_multi()
    .sample_qc()
    .variant_qc())

vds.export_variants('variantqc.tsv', 'Variant = v, va.qc.*')

vds.write('sample.qc.vds')

vds.count()

vds.summarize().report()

print('sample annotation schema:')

print(vds.sample_schema)

print('\nvariant annotation schema:')

print(vds.variant_schema)

(vds.filter_variants_expr('v.altAllele().isSNP() && va.qc.gqMean >= 20')
    .filter_samples_expr('sa.qc.callRate >= 0.97 && sa.qc.dpMean >= 15')
    .filter_genotypes('let ab = g.ad[1] / g.ad.sum() in '
                      '((g.isHomRef() && ab <= 0.1) || '
                      ' (g.isHet() && ab >= 0.25 && ab <= 0.75) || '
                      ' (g.isHomVar() && ab >= 0.9))')
    .write('sample.filtered.vds'))

import shutil, os

hc.stop()

if os.path.isdir("output/"):
    shutil.rmtree("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

# representation/hail.representation.Pedigree

import os, shutil
from hail import *
from hail.representation import *
from hail.expr import *
from hail.stats import *

if not os.path.isdir("output/"):
    os.mkdir("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

hc = HailContext(log="output/hail.log", quiet=True)
vds = hc.read('data/example.vds')

ped = Pedigree.read('data/test.fam')

ped = Pedigree.read('data/test.fam')

ped.write('out.fam')

import shutil, os

hc.stop()

if os.path.isdir("output/"):
    shutil.rmtree("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

# representation/hail.representation.Genotype

import os, shutil
from hail import *
from hail.representation import *
from hail.expr import *
from hail.stats import *

if not os.path.isdir("output/"):
    os.mkdir("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

hc = HailContext(log="output/hail.log", quiet=True)
vds = hc.read('data/example.vds')

g = Genotype(0, ad=[9,1], dp=11, gq=20, pl=[0,100,1000])

g.ad[0] / sum(g.ad)

g.dp - sum(g.ad)

num_alleles = 2
hom_ref = Genotype(0)
het = Genotype(1)
hom_var = Genotype(2)

hom_ref.one_hot_alleles(num_alleles) == [2, 0]
het.one_hot_alleles(num_alleles) == [1, 1]
hom_var.one_hot_alleles(num_alleles) == [0, 2]

num_genotypes = 3
hom_ref = Genotype(0)
het = Genotype(1)
hom_var = Genotype(2)

hom_ref.one_hot_genotype(num_genotypes) == [1, 0, 0]
het.one_hot_genotype(num_genotypes) == [0, 1, 0]
hom_var.one_hot_genotype(num_genotypes) == [0, 0, 1]

import shutil, os

hc.stop()

if os.path.isdir("output/"):
    shutil.rmtree("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

# hail.VariantDataset

import os, shutil
from hail import *
from hail.representation import *
from hail.expr import *
from hail.stats import *

if not os.path.isdir("output/"):
    os.mkdir("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)

hc = HailContext(log="output/hail.log", quiet=True)
vds = hc.read('data/example.vds')

vds = hc.read("data/example.vds")

kt_result = (vds
    .aggregate_by_key(['Sample = s', 'Gene = va.gene'],
                       'nHet = g.filter(g => g.isHet() && va.consequence == "LOF").count()')
    .export("test.tsv"))

vds_result = vds.annotate_alleles_expr('va.nNonRefSamples = gs.filter(g => g.isCalledNonRef()).count()')

vds_result = vds.annotate_genotypes_expr('g = {GT: g.gt, CASE_HET: sa.pheno.isCase && g.isHet()}')

vds_gta = (hc.import_vcf('data/example3.vcf.bgz', generic=True, call_fields=['GTA'])
                .annotate_genotypes_expr('g = g.GTA.toGenotype()'))

vds_result = vds.annotate_global('global.populations',
                                    ['EAS', 'AFR', 'EUR', 'SAS', 'AMR'],
                                    TArray(TString()))

vds = vds.annotate_global_expr('global.pops = ["FIN", "AFR", "EAS", "NFE"]')

vds = vds.annotate_global_expr('global.pops = ["FIN", "AFR", "EAS"]')

vds = vds.annotate_global_expr('global.pops = ["FIN", "AFR", "EAS", "NFE"]')

vds = vds.annotate_global_expr('global.pops = drop(global, pops)')

vds_result = (vds.annotate_samples_expr('sa.gqHetStats = gs.filter(g => g.isHet()).map(g => g.gq).stats()')
    .export_samples('output/samples.txt', 'sample = s, het_gq_mean = sa.gqHetStats.mean'))

variant_annotations_table = hc.import_table('data/consequence.tsv', impute=True).key_by('Variant')

vds_result = (vds.annotate_variants_table(variant_annotations_table, root='va.consequence')
    .annotate_variants_expr('va.isSingleton = gs.map(g => g.nNonRefAlleles()).sum() == 1')
    .annotate_samples_expr('sa.LOF_genes = gs.filter(g => va.isSingleton && g.isHet() && va.consequence == "LOF").map(g => va.gene).collect()'))

vds_result = vds.annotate_samples_expr('sa.newpheno = if (sa.pheno.cohortName == "cohort1") sa.pheno.bloodPressure else NA: Double')

annotations = hc.import_table('data/samples1.tsv').key_by('Sample')

vds_result = vds.annotate_samples_table(annotations, root='sa.phenotypes')

annotations = hc.import_table('data/samples2.tsv', delimiter=',', missing='.').key_by('PT-ID')

vds_result = vds.annotate_samples_table(annotations, root='sa.batch')

annotations = (hc.import_table('data/samples3.tsv', no_header=True)
                 .annotate('sample = f0.split("_")[1]')
                 .key_by('sample'))

vds_result = vds.annotate_samples_table(annotations,
                            expr='sa.sex = table.f1, sa.batch = table.f0.split("_")[0]')

vds_result = vds.annotate_variants_expr('va.gqHetStats = gs.filter(g => g.isHet()).map(g => g.gq).stats()')

vds_result = vds.annotate_variants_expr('va.nonRefSamples = gs.filter(g => g.isCalledNonRef()).map(g => s).collect()')

vds_result = vds.annotate_variants_expr('va.rsid = str(v)')

table = hc.import_table('data/variant-lof.tsv', impute=True).key_by('v')

vds_result = vds.annotate_variants_table(table, root='va.lof')

kt = hc.import_table('data/locus-table.tsv', impute=True).key_by('Locus')

vds_result = vds.annotate_variants_table(table, root='va.scores')

table = hc.import_table('data/locus-metadata.tsv', impute=True).key_by(['gene', 'type'])

vds_result = (vds.annotate_variants_table(table,
      root='va.foo',
      vds_key=['va.gene', 'if (va.score > 10) "Type1" else "Type2"']))

intervals = KeyTable.import_interval_list('data/exons2.interval_list')

vds_result = vds.annotate_variants_table(intervals, root='va.exon')

intervals = KeyTable.import_interval_list('data/exons2.interval_list')

vds_result = vds.annotate_variants_table(intervals, root='va.exons', product=True)

intervals = KeyTable.import_bed('data/file2.bed')

vds_result = vds.annotate_variants_table(intervals, root='va.bed')

vds1 = vds.annotate_variants_expr('va = drop(va, anno1)')

vds2 = (hc.read("data/example2.vds")
          .annotate_variants_expr('va = select(va, anno1, toKeep1, toKeep2, toKeep3)'))

vds_result = vds1.annotate_variants_vds(vds2, expr='va.annot = vds.anno1')

vds_result = vds1.annotate_variants_vds(vds2, expr='va = merge(va, vds)')

vds_result = vds1.annotate_variants_vds(vds2, expr='va.annotations = select(vds, toKeep1, toKeep2, toKeep3)')

vds_result = vds1.annotate_variants_vds(vds2, expr='va.annotations.toKeep1 = vds.toKeep1, ' +
                                      'va.annotations.toKeep2 = vds.toKeep2, ' +
                                      'va.annotations.toKeep3 = vds.toKeep3')

comparison_vds = hc.read('data/example2.vds')

summary, samples, variants = vds.concordance(comparison_vds)

summary, samples, variants = vds.concordance(hc.read('data/example2.vds'))

left_homref_right_homvar = summary[2][4]

left_het_right_missing = summary[3][1]

left_het_right_something_else = sum(summary[3][:]) - summary[3][3]

total_concordant = summary[2][2] + summary[3][3] + summary[4][4]

total_discordant = sum([sum(s[2:]) for s in summary[2:]]) - total_concordant

samples, variants = vds.count()

vds_result = vds.drop_variants()

vds3 = hc.import_bgen("data/example3.bgen", sample_file="data/example3.sample")

(vds3.filter_variants_expr("gs.infoScore().score >= 0.9")
     .export_gen("output/infoscore_filtered"))

vds.export_genotypes('output/genotypes.tsv', 'SAMPLE=s, VARIANT=v, GQ=g.gq, DP=g.dp, ANNO1=va.anno1, ANNO2=va.anno2')

vds.export_genotypes('output/genotypes.tsv', 's, v, g.gq, g.dp, va.anno1, va.anno2')

vds.split_multi().export_plink('output/plink')

vds.split_multi().export_plink('output/plink')

(vds.sample_qc()
    .export_samples('output/samples.tsv', 'SAMPLE = s, CALL_RATE = sa.qc.callRate, NHET = sa.qc.nHet'))

(vds.sample_qc()
    .export_samples('output/samples.tsv', 's, sa.qc.rTiTv'))

vds.export_variants('output/file.tsv',
       'VARIANT = v, PASS = va.pass, FILTERS = va.filters, MISSINGNESS = 1 - va.qc.callRate')

vds.export_variants('output/file.tsv', 'v, va.pass, va.qc.AF')

vds.export_variants('output/file.tsv', 'variant = v, va.qc.*')

vds.export_variants('output/file.tsv', 'variant = v, QC = va.qc.*')

vds.export_vcf('output/example.vcf.bgz')

vds.export_vcf('output/example_out.vcf')

(vds.filter_genotypes('g.gq >= 20')
    .variant_qc()
    .annotate_variants_expr('va.info.AC = va.qc.AC')
    .export_vcf('output/example.vcf.bgz'))

vds_result = vds.filter_alleles('va.info.AC[aIndex - 1] == 0',
    annotation='va.info.AC = aIndices[1:].map(i => va.info.AC[i - 1])',
    keep=False)

vds_result = vds.filter_genotypes('let ab = g.ad[1] / g.ad.sum() in ' +
                     '((g.isHomRef() && ab <= 0.1) || ' +
                     '(g.isHet() && ab >= 0.25 && ab <= 0.75) || ' +
                     '(g.isHomVar() && ab >= 0.9))')

vds_result = vds.filter_intervals(Interval.parse('17:38449840-38530994'))

vds_result = vds.filter_intervals(Interval(Locus('17', 38449840), Locus('17', 38530994)))

intervals = map(Interval.parse, ['1:50M-75M', '2:START-400000', '3-22'])

intervals = [Interval.parse(x) for x in ['1:50M-75M', '2:START-400000', '3-22']]

vds_result = vds.filter_intervals(intervals)

interval = Interval.parse('15:100000-101000')

vds_filtered = vds.filter_variants_expr('v.contig == "15" && v.start >= 100000 && v.start < 200000')

vds_filtered = vds.filter_intervals(Interval.parse('15:100000-200000'))

vds_result = vds.filter_samples_expr("sa.isCase")

vds_result = vds.filter_samples_expr('"^NA" ~ s' , keep=False)

(vds.sample_qc()
    .filter_samples_expr('sa.qc.callRate >= 0.99 && sa.qc.dpMean >= 10')
    .write("output/filter_samples.vds"))

to_remove = ['NA12878', 'NA12891', 'NA12892']

vds_result = vds.filter_samples_list(to_remove, keep=False)

to_remove = [s.strip() for s in open('data/exclude_samples.txt')]

vds_result = vds.filter_samples_list(to_remove, keep=False)

table = hc.import_table('data/samples1.tsv').key_by('Sample')

vds_filtered = vds.filter_samples_table(table, keep=True)

vds_result = vds.filter_variants_expr('va.gene == "CHD8"')

vds_result = vds.filter_variants_expr('v.contig == "1"', keep=False)

vds_filtered = vds.filter_variants_list([Variant.parse('20:10626633:G:GC'), 
                                         Variant.parse('20:10019093:A:G')], keep=True)

kt = hc.import_table('data/sample_variants.txt', key='Variant', impute=True)

filtered_vds = vds.filter_variants_table(kt, keep=True)

kt = hc.import_table('data/locus-table.tsv', impute=True).key_by('Locus')

filtered_vds = vds.filter_variants_table(kt, keep=True)

kt = KeyTable.import_bed('data/file2.bed')

filtered_vds = vds.filter_variants_table(kt, keep=False)

print(vds.genotype_schema)

gs = vds.genotypes_table()

print(vds.global_schema)

km = vds.grm()

vds.ibd()

vds.ibd(maf='va.panel_maf', min=0.2, max=0.9)

pruned_vds = vds.ibd_prune(0.5)

imputed_sex_vds = (vds.impute_sex()
    .annotate_samples_expr('sa.sexcheck = sa.pheno.isFemale == sa.imputesex.isFemale')
    .filter_samples_expr('sa.sexcheck || isMissing(sa.sexcheck)'))

vds_result = (vds.variant_qc()
                 .filter_variants_expr("va.qc.AF >= 0.05 && va.qc.AF <= 0.95")
                 .ld_prune()
                 .export_variants("output/ldpruned.variants", "v"))

vds_result = vds.linreg('sa.pheno.height', covariates=['sa.pheno.age', 'sa.pheno.isFemale'])

vds_result = vds.linreg('sa.pheno.height', covariates=['sa.pheno.age', 'sa.pheno.isFemale', 'sa.cov.PC1'])

vds_result = vds.linreg('if (sa.pheno.isFemale) sa.pheno.age else (2 * sa.pheno.age + 10)', covariates=[])

linreg_kt, sample_kt = (hc.read('data/example_burden.vds')
    .linreg_burden(key_name='gene',
                   variant_keys='va.genes',
                   single_key=False,
                   agg_expr='gs.map(g => g.gt).max()',
                   y='sa.burden.pheno',
                   covariates=['sa.burden.cov1', 'sa.burden.cov2']))

linreg_kt, sample_kt = (hc.read('data/example_burden.vds')
    .linreg_burden(key_name='gene',
                   variant_keys='va.gene',
                   single_key=True,
                   agg_expr='gs.map(g => va.weight * g.gt).sum()',
                   y='sa.burden.pheno',
                   covariates=['sa.burden.cov1', 'sa.burden.cov2']))

assoc_vds = hc.read("data/example_lmmreg.vds")

kinship_matrix = assoc_vds.filter_variants_expr('va.useInKinship').rrm()

lmm_vds = assoc_vds.lmmreg(kinship_matrix, 'sa.pheno', ['sa.cov1', 'sa.cov2'])

lmm_vds.export_variants('output/lmmreg.tsv.bgz', 'variant = v, va.lmmreg.*')

lmmreg_results = lmm_vds.globals['lmmreg']

vds_result = vds.logreg('wald', 'sa.pheno.isCase', covariates=['sa.pheno.age', 'sa.pheno.isFemale'])

logreg_kt, sample_kt = (hc.read('data/example_burden.vds')
    .logreg_burden(key_name='gene',
                   variant_keys='va.genes',
                   single_key=False,
                   agg_expr='gs.map(g => g.gt).max()',
                   test='wald',
                   y='sa.burden.pheno',
                   covariates=['sa.burden.cov1', 'sa.burden.cov2']))

logreg_kt, sample_kt = (hc.read('data/example_burden.vds')
    .logreg_burden(key_name='gene',
                   variant_keys='va.gene',
                   single_key=True,
                   agg_expr='gs.map(g => va.weight * g.gt).sum()',
                   test='score',
                   y='sa.burden.pheno',
                   covariates=['sa.burden.cov1', 'sa.burden.cov2']))

kt = vds.make_table('v = v', ['gt = g.gt', 'gq = g.gq'])

kt = vds.make_table('v = v', ['gt = g.gt', 'gq = g.gq'])

ped = Pedigree.read('data/trios.fam')

all, per_fam, per_sample, per_variant = vds.mendel_errors(ped)

all.export('output/all_mendel_errors.tsv')

annotated_vds = vds.annotate_samples_table(per_sample, root="sa.mendel")

annotated_vds = vds.annotate_variants_table(per_variant, root="va.mendel")

vds_result = vds.pca('sa.scores')

vds_result = vds.pca('sa.scores', 'va.loadings', 'global.evals', 5, as_array=True)

vds_result = vds.persist()

gq_hist = vds.query_genotypes('gs.map(g => g.gq).hist(0, 100, 100)')

call_rate = vds.query_genotypes('gs.fraction(g => g.isCalled)')

[gq_hist, dp_hist] = vds.query_genotypes(['gs.map(g => g.gq).hist(0, 100, 100)',
                                                    'gs.map(g => g.dp).hist(0, 60, 60)'])

gq_hist, t = vds.query_genotypes_typed('gs.map(g => g.gq).hist(0, 100, 100)')

[gq_hist, dp_hist], [t1, t2] = vds.query_genotypes_typed(['gs.map(g => g.gq).hist(0, 100, 100)',
                                                          'gs.map(g => g.dp).hist(0, 60, 60)'])

result1 = vds.query_genotypes('gs.count()')

result2 = vds.query_genotypes('gs.filter(g => v.altAllele.isSNP() && g.isHet).count()')

exprs = ['gs.count()', 'gs.filter(g => v.altAllele.isSNP() && g.isHet).count()']

[geno_count, snp_hets] = vds.query_genotypes(exprs)

low_callrate_samples = vds.query_samples('samples.filter(s => sa.qc.callRate < 0.95).collect()')

low_callrate_samples, t = vds.query_samples_typed(
   'samples.filter(s => sa.qc.callRate < 0.95).collect()')

lof_variant_count = vds.query_variants('variants.filter(v => va.consequence == "LOF").count()')

[lof_variant_count, missense_count] = vds.query_variants([
    'variants.filter(v => va.consequence == "LOF").count()',
    'variants.filter(v => va.consequence == "Missense").count()'])

exprs = ['variants.count()', 'variants.filter(v => v.altAllele.isSNP()).count()']

[num_variants, num_snps] = vds.query_variants(exprs)

result1 = vds.query_variants('variants.count()')

result2 = vds.query_variants('variants.filter(v => v.altAllele.isSNP()).count()')

lof_variant_count, t = vds.query_variants_typed(
    'variants.filter(v => va.consequence == "LOF").count()')

[lof_variant_count, missense_count], [t1, t2] = vds.query_variants_typed([
    'variants.filter(v => va.consequence == "LOF").count()',
    'variants.filter(v => va.consequence == "Missense").count()'])

vds_result = vds.rename_samples({'ID1': 'id1', 'ID2': 'id2'})

vds_result = vds.repartition(500)

kinship_matrix = vds.rrm()

vds.same(vds)

print(vds.sample_schema)

small_vds = vds.sample_variants(0.01)

annotated_vds = vds.annotate_variants_expr([
'va.info.AC_HC = gs.filter(g => g.dp >= 10 && g.gq >= 20).callStats(g => v).AC[1:]',
'va.filters = if((v.altAllele.isSNP && (va.info.QD < 2.0 || va.info.FS < 60 || va.info.MQ < 40 || ' +
'va.info.MQRankSum < -12.5 || va.info.ReadPosRankSum < -8.0)) || ' +
'(va.info.QD < 2.0 || va.info.FS < 200.0 || va.info.ReadPosRankSum < 20.0)) va.filters.add("HardFilter") else va.filters'])

vds.split_multi().write('output/split.vds')

vds_result = (vds.split_multi()
    .filter_variants_expr('va.info.AC[va.aIndex - 1] < 10', keep = False))

(vds.split_multi()
    .annotate_variants_expr('va.info.AC = va.info.AC[va.aIndex - 1]')
    .export_vcf('output/export.vcf'))

s = vds.summarize()

print(s.contigs)

print('call rate is %.2f' % s.call_rate)

s.report()

pedigree = Pedigree.read('data/trios.fam')

(vds.tdt(pedigree)
    .export_variants("output/tdt_results.tsv", "Variant = v, va.tdt.*"))

vds_result = vds.variant_qc()

print(vds.variant_schema)

vds.write("output/sample.vds")

import shutil, os

hc.stop()

if os.path.isdir("output/"):
    shutil.rmtree("output/")

vds_files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in vds_files:
    if os.path.isdir(f):
        shutil.rmtree(f)
