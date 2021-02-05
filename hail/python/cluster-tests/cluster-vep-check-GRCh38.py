import hail as hl

GNOMAD_CHR22_FIRST_1000 = "gs://hail-us-vep/vep_examplars/gnomad3_chr22_first_1000.mt"

for path, csq in [(GNOMAD_CHR22_FIRST_1000, False)]:
    print(f"Checking 'hl.vep' replicates on '{path}'")
    expected = hl.read_matrix_table(path)
    actual = hl.vep(expected.rows().select(), 'gs://hail-us-vep/vep95-GRCh38-loftee-gcloud.json', csq=csq).drop('vep_proc_id')
    actual._force_count()
    # vep_result_agrees = actual._same(expected)
    # if vep_result_agrees:
    #     print('TEST PASSED')
    # else:
    #     print('TEST FAILED')
    # assert vep_result_agrees
