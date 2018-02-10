import os, shutil
import hail as hl
import hail.expr.aggregators as agg
from hail.stats import *

if not os.path.isdir("output/"):
    os.mkdir("output/")

files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in files:
    if os.path.isdir(f):
        shutil.rmtree(f)

hl.init(log="output/hail.log", quiet=True)

ds = hl.import_vcf('data/sample.vcf.bgz')
ds = ds.sample_rows(0.03)
ds = ds.annotate_rows(useInKinship = hl.rand_bool(0.9), panel_maf = 0.1, anno1 = 5, anno2 = 0, consequence = "LOF", gene = "A", score = 5.0)
ds = ds.annotate_rows(aIndex = 1)
ds = hl.sample_qc(hl.variant_qc(ds))
ds = ds.annotate_cols(isCase = True,
                      pheno = hl.Struct(isCase = hl.rand_bool(0.5),
                                     isFemale = hl.rand_bool(0.5),
                                     age = hl.rand_norm(65, 10),
                                     height = hl.rand_norm(70, 10),
                                     bloodPressure = hl.rand_norm(120, 20),
                                     cohortName = "cohort1"),
                      cov = hl.Struct(PC1 = hl.rand_norm(0, 1)), cov1 = hl.rand_norm(0, 1), cov2 = hl.rand_norm(0, 1))
ds.write('data/example.vds', overwrite = True)

ds = hl.import_vcf('data/sample.vcf.bgz')
ds = ds.sample_rows(0.015)
ds = ds.annotate_rows(anno1 = 5, toKeep1 = True, toKeep2 = False, toKeep3 = True)
ds = hl.split_multi_hts(ds)
ds.write('data/example2.vds', overwrite=True)

ds = hl.import_vcf('data/sample.vcf.bgz')
ds.write('data/example2.multi.generic.vds', overwrite=True)

ds = hl.import_vcf('data/sample.vcf.bgz')
ds = hl.split_multi_hts(ds)
ds = hl.variant_qc(ds)
kt = hl.import_table('data/example_lmmreg.tsv', key='Sample', impute=True)
ds = ds.annotate_cols(**kt[ds.s])
ds = ds.annotate_rows(useInKinship = ds.variant_qc.AF > 0.05)
ds.write('data/example_lmmreg.vds', overwrite=True)

ds = hl.import_vcf('data/example_burden.vcf')
kt = hl.import_table('data/example_burden.tsv', key='Sample', impute=True)
ds = ds.annotate_cols(burden = kt[ds.s])
ds = ds.annotate_rows(weight = ds.locus.position.to_float64())
ds = hl.variant_qc(ds)
# geneskt = hl.import_interval_list('data/genes.interval_list')
genekt = hl.import_interval_list('data/gene.interval_list')
# ds = ds.annotate_rows(genes = ???)
ds = ds.annotate_rows(gene = genekt[ds.locus])
ds.write('data/example_burden.vds', overwrite=True)

ds = hl.read_matrix_table('data/example.vds')
ds.write('/tmp/example.vds', overwrite=True)

multiallelic_generic_ds = hl.read_matrix_table('data/example2.multi.generic.vds')

lmmreg_ds = hl.variant_qc(hl.split_multi_hts(hl.import_vcf('data/sample.vcf.bgz')))
lmmreg_tsv = hl.import_table('data/example_lmmreg.tsv', 'Sample', impute=True)
lmmreg_ds = lmmreg_ds.annotate_cols(**lmmreg_tsv[lmmreg_ds['s']])
lmmreg_ds = lmmreg_ds.annotate_rows(useInKinship = lmmreg_ds.variant_qc.AF > 0.05)
lmmreg_ds.write('data/example_lmmreg.vds', overwrite=True)

table1 = hl.import_table('data/kt_example1.tsv', impute=True, key='ID')
table1 = table1.annotate_globals(global_field_1 = 5, global_field_2 = 10)

ds = ds.annotate_rows(gene=['TTN'])
ds = ds.annotate_cols(cohorts=['1kg'], pop='EAS')

dataset = ds
