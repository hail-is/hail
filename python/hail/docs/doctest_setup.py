import os, shutil
from hail import *
from hail2 import *
from hail.stats import *

if not os.path.isdir("output/"):
    os.mkdir("output/")

files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds", "data/ld_matrix"]
for f in files:
    if os.path.isdir(f):
        shutil.rmtree(f)

init(log="output/hail.log", quiet=True)

from hail.utils.java import Env
hc = Env.hc()

(hc.import_vcf('data/sample.vcf.bgz')
    .sample_variants(0.03)
    .annotate_variants_expr('va.useInKinship = pcoin(0.9), va.panel_maf = 0.1, va.anno1 = 5, va.anno2 = 0, va.consequence = "LOF", va.gene = "A", va.score = 5.0')
    .annotate_variants_expr('va.aIndex = 1') # as if split_multi was called
    .variant_qc()
    .sample_qc()
    .annotate_samples_expr('sa.isCase = true, sa.pheno.isCase = pcoin(0.5), sa.pheno.isFemale = pcoin(0.5), sa.pheno.age=rnorm(65, 10), sa.cov.PC1 = rnorm(0,1), sa.pheno.height = rnorm(70, 10), sa.cov1 = rnorm(0, 1), sa.cov2 = rnorm(0,1), sa.pheno.bloodPressure = rnorm(120,20), sa.pheno.cohortName = "cohort1"')
    .write('data/example.vds', overwrite=True))

(hc.import_vcf("data/sample.vcf.bgz")
    .sample_variants(0.015).annotate_variants_expr('va.anno1 = 5, va.toKeep1 = true, va.toKeep2 = false, va.toKeep3 = true')
    .split_multi_hts()
    .write("data/example2.vds", overwrite=True))

(hc.import_vcf("data/sample.vcf.bgz")
    .write("data/example2.multi.generic.vds", overwrite=True))

(hc.import_vcf('data/example_burden.vcf')
    .annotate_samples_table(hc.import_table('data/example_burden.tsv', 'Sample', impute=True), root='sa.burden')
    .annotate_variants_expr('va.weight = v.start.toFloat64()')
    .variant_qc()
    .annotate_variants_table(KeyTable.import_interval_list('data/genes.interval_list'), root='va.genes', product=True)
    .annotate_variants_table(KeyTable.import_interval_list('data/gene.interval_list'), root='va.gene', product=False)
    .write('data/example_burden.vds', overwrite=True))

vds = hc.read('data/example.vds')

multiallelic_generic_vds = hc.read('data/example2.multi.generic.vds')

vds.split_multi_hts().ld_matrix().write("data/ld_matrix")

table1 = methods.import_table('data/kt_example1.tsv', impute=True, key='ID')
lmmreg_ds = methods.variant_qc(methods.split_multi_hts(methods.import_vcf('data/sample.vcf.bgz')))
lmmreg_tsv = hc.import_table('data/example_lmmreg.tsv', 'Sample', impute=True)
lmmreg_ds = lmmreg_ds.annotate_cols(**lmmreg_tsv[lmmreg_ds['s']])
lmmreg_ds = lmmreg_ds.annotate_rows(useInKinship = lmmreg_ds.variant_qc.AF > 0.05)
lmmreg_ds.write('data/example_lmmreg.vds', overwrite=True)

table1 = hc.import_table('data/kt_example1.tsv', impute=True, key='ID')
table1 = table1.annotate_globals(global_field_1 = 5, global_field_2 = 10)

dataset = (vds.annotate_samples_expr('sa = merge(drop(sa, qc), {sample_qc: sa.qc})')
    .annotate_variants_expr('va = merge(drop(va, qc), {variant_qc: va.qc})').to_hail2())

dataset = dataset.annotate_rows(gene=['TTN'])
dataset = dataset.annotate_cols(cohorts=['1kg'], pop='EAS')
