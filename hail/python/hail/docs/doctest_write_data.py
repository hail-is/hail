import os, shutil
import hail as hl

if not os.path.isdir("output/"):
    os.mkdir("output/")

files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
for f in files:
    if os.path.isdir(f):
        shutil.rmtree(f)

ds = hl.import_vcf('data/sample.vcf.bgz')
ds = ds.sample_rows(0.03)
ds = ds.annotate_rows(use_as_marker=hl.rand_bool(0.5),
                      panel_maf=0.1,
                      anno1=5,
                      anno2=0,
                      consequence="LOF",
                      gene="A",
                      score=5.0)
ds = ds.annotate_rows(a_index=1)
ds = hl.sample_qc(hl.variant_qc(ds))
ds = ds.annotate_cols(is_case=True,
                      pheno=hl.struct(is_case=hl.rand_bool(0.5),
                                      is_female=hl.rand_bool(0.5),
                                      age=hl.rand_norm(65, 10),
                                      height=hl.rand_norm(70, 10),
                                      blood_pressure=hl.rand_norm(120, 20),
                                      cohort_name="cohort1"),
                      cov=hl.struct(PC1=hl.rand_norm(0, 1)),
                      cov1=hl.rand_norm(0, 1),
                      cov2=hl.rand_norm(0, 1),
                      cohort="SIGMA")
ds = ds.annotate_globals(global_field_1=5,
                         global_field_2=10,
                         pli={'SCN1A': 0.999, 'SONIC': 0.014},
                         populations=['AFR', 'EAS', 'EUR', 'SAS', 'AMR', 'HIS'])
ds = ds.annotate_rows(gene=['TTN'])
ds = ds.annotate_cols(cohorts=['1kg'], pop='EAS')
ds.write('data/example.vds', overwrite=True)

lmmreg_ds = hl.variant_qc(hl.split_multi_hts(hl.import_vcf('data/sample.vcf.bgz')))
lmmreg_tsv = hl.import_table('data/example_lmmreg.tsv', 'Sample', impute=True)
lmmreg_ds = lmmreg_ds.annotate_cols(**lmmreg_tsv[lmmreg_ds['s']])
lmmreg_ds = lmmreg_ds.annotate_rows(use_in_kinship = lmmreg_ds.variant_qc.AF[1] > 0.05)
lmmreg_ds.write('data/example_lmmreg.vds', overwrite=True)

burden_ds = hl.import_vcf('data/example_burden.vcf')
burden_kt = hl.import_table('data/example_burden.tsv', key='Sample', impute=True)
burden_ds = burden_ds.annotate_cols(burden = burden_kt[burden_ds.s])
burden_ds = burden_ds.annotate_rows(weight = hl.float64(burden_ds.locus.position))
burden_ds = hl.variant_qc(burden_ds)
genekt = hl.import_locus_intervals('data/gene.interval_list')
burden_ds = burden_ds.annotate_rows(gene=genekt[burden_ds.locus])
burden_ds.write('data/example_burden.vds', overwrite=True)
