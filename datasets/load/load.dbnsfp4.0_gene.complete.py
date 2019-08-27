import hail as hl

t = hl.import_table('gs://hail-common/datasets-raw/dbnsfp/dbNSFP4.0_gene.complete.bgz',force=True, impute=True, missing=".")

t = t.annotate(Gene_old_names = t.Gene_old_names.split(";"))
t = t.annotate(Gene_other_names = t.Gene_other_names.split(";"))
t = t.annotate(CCDS_id = t.CCDS_id.split(";"))
t = t.annotate(Refseq_id = t.Refseq_id.split(";"))
t = t.annotate(GO_biological_process = t.GO_biological_process.split(";"))
t = t.annotate(GO_cellular_component = t.GO_cellular_component.split(";"))
t = t.annotate(GO_molecular_function = t.GO_molecular_function.split(";"))

t = t.key_by(t["Gene_name"])

t.write('gs://hail-common/datasets/1/dbnsfp/dbNSFP4.0_gene.complete.bgz.ht', overwrite=True)
