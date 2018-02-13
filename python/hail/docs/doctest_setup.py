import os, shutil
import hail as hl
import hail.expr.aggregators as agg
from hail.stats import *
from hail.utils.java import warn

if not os.path.isdir("output/"):
    os.mkdir("output/")

files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in files:
    if os.path.isdir(f):
        shutil.rmtree(f)

ds = hl.import_vcf('data/sample.vcf.bgz')
ds = ds.sample_rows(0.03)
ds = ds.annotate_rows(useInKinship=hl.rand_bool(0.9), panel_maf=0.1, anno1=5, anno2=0, consequence="LOF", gene="A",
                      score=5.0)
ds = ds.annotate_rows(aIndex=1)
ds = hl.sample_qc(hl.variant_qc(ds))
ds = ds.annotate_cols(isCase=True,
                      pheno=hl.Struct(isCase=hl.rand_bool(0.5),
                                      isFemale=hl.rand_bool(0.5),
                                      age=hl.rand_norm(65, 10),
                                      height=hl.rand_norm(70, 10),
                                      bloodPressure=hl.rand_norm(120, 20),
                                      cohortName="cohort1"),
                      cov=hl.Struct(PC1=hl.rand_norm(0, 1)), cov1=hl.rand_norm(0, 1), cov2=hl.rand_norm(0, 1))

ds = ds.annotate_rows(gene=['TTN'])
ds = ds.annotate_cols(cohorts=['1kg'], pop='EAS')
ds.write('data/example.vds', overwrite=True)

table1 = hl.import_table('data/kt_example1.tsv', impute=True, key='ID')
table1 = table1.annotate_globals(global_field_1=5, global_field_2=10)

ds = hl.read_matrix_table('data/example.vds')

dataset = ds
