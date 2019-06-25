import hail as hl

t = hl.import_table('/Users/bfranco/downloads/release-2.1.1-constraint-gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz',impute=True, force_bgz=True)

t.write('gs://hail-datasets-hail-data/gnomad_v2.1.1_lof_metrics_by_gene.mt', overwrite=True)
