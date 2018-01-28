import os, shutil
from hail import *
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

ds = methods.import_vcf('data/sample.vcf.bgz')
ds = ds.sample_rows(0.03)
ds = ds.annotate_rows(useInKinship = functions.rand_bool(0.9), panel_maf = 0.1, anno1 = 5, anno2 = 0, consequence = "LOF", gene = "A", score = 5.0)
ds = ds.annotate_rows(aIndex = 1)
ds = methods.sample_qc(methods.variant_qc(ds))
ds = ds.annotate_cols(isCase = True,
                      pheno = Struct(isCase = functions.rand_bool(0.5),
                                     isFemale = functions.rand_bool(0.5),
                                     age = functions.rand_norm(65, 10),
                                     height = functions.rand_norm(70, 10),
                                     bloodPressure = functions.rand_norm(120, 20),
                                     cohortName = "cohort1"),
                      cov = Struct(PC1 = functions.rand_norm(0, 1)), cov1 = functions.rand_norm(0, 1), cov2 = functions.rand_norm(0, 1))
ds.write('data/example.vds', overwrite = True)

ds = methods.import_vcf('data/sample.vcf.bgz')
ds = ds.sample_rows(0.015)
ds = ds.annotate_rows(anno1 = 5, toKeep1 = True, toKeep2 = False, toKeep3 = True)
ds = methods.split_multi_hts(ds)
ds.write('data/example2.vds', overwrite=True)

ds = methods.import_vcf('data/sample.vcf.bgz')
ds.write('data/example2.multi.generic.vds', overwrite=True)

ds = methods.import_vcf('data/sample.vcf.bgz')
ds = methods.split_multi_hts(ds)
ds = methods.variant_qc(ds)
kt = methods.import_table('data/example_lmmreg.tsv', key='Sample', impute=True)
ds = ds.annotate_cols(**kt[ds.s])
ds = ds.annotate_rows(useInKinship = ds.variant_qc.AF > 0.05)
ds.write('data/example_lmmreg.vds', overwrite=True)

ds = methods.import_vcf('data/example_burden.vcf')
kt = methods.import_table('data/example_burden.tsv', key='Sample', impute=True)
ds = ds.annotate_cols(burden = kt[ds.s])
ds = ds.annotate_rows(weight = ds.v.start.to_float64())
ds = methods.variant_qc(ds)
# geneskt = methods.import_interval_list('data/genes.interval_list')
# genekt = methods.import_interval_list('data/gene.interval_list')
# ds = ds.annotate_rows(genes = ???)
# ds = ds.annotate_rows(gene = genekt[ds.v.locus()])
ds.write('data/example_burden.vds', overwrite=True)

ds = methods.read_matrix('data/example.vds')

methods.ld_matrix(methods.split_multi_hts(ds)).write("data/ld_matrix")

multiallelic_generic_ds = methods.read_matrix('data/example2.multi.generic.vds')

lmmreg_ds = methods.variant_qc(methods.split_multi_hts(methods.import_vcf('data/sample.vcf.bgz')))
lmmreg_tsv = methods.import_table('data/example_lmmreg.tsv', 'Sample', impute=True)
lmmreg_ds = lmmreg_ds.annotate_cols(**lmmreg_tsv[lmmreg_ds['s']])
lmmreg_ds = lmmreg_ds.annotate_rows(useInKinship = lmmreg_ds.variant_qc.AF > 0.05)
lmmreg_ds.write('data/example_lmmreg.vds', overwrite=True)

table1 = methods.import_table('data/kt_example1.tsv', impute=True, key='ID')
table1 = table1.annotate_globals(global_field_1 = 5, global_field_2 = 10)

ds = ds.annotate_rows(gene=['TTN'])
ds = ds.annotate_cols(cohorts=['1kg'], pop='EAS')

dataset = ds
