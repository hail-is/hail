import hail as hl

t = hl.import_table('/Users/bfranco/downloads/release-2.1.1-constraint-gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz',impute=True, force_bgz=True)

t = t.key_by(t.gene)

t.write('gs://hail-common/datasets/1/gnomad_v2.1.1_lof_metrics_by_gene.mt', overwrite=True)
