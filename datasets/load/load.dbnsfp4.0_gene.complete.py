import hail as hl

t = hl.import_table('gs://hail-common/datasets-raw/dbnsfp/dbNSFP4.0_gene.complete.bgz',force=True, impute=True, missing=".")

t = t.key_by(t["Gene_name"])

t.write('gs://hail-common/datasets/1/dbnsfp/dbNSFP4.0_gene.complete.bgz.ht', overwrite=True)
