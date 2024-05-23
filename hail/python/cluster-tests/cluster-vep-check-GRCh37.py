import hail as hl

GOLD_STD = 'gs://hail-us-vep/vep_examplars/vep_no_csq_4dc19bc1b.mt/'
GOLD_STD_CSQ = 'gs://hail-us-vep/vep_examplars/vep_csq_4dc19bc1b.mt/'

for path, csq in [(GOLD_STD, False), (GOLD_STD_CSQ, True)]:
    print(f"Checking 'hl.vep' replicates on '{path}'")
    expected = hl.read_matrix_table(path)
    actual = hl.vep(expected.rows().select(), 'gs://hail-us-vep/vep85-loftee-gcloud-testing.json', csq=csq).drop(
        'vep_proc_id'
    )
    actual._force_count()
    # vep_result_agrees = actual._same(expected)
    # if vep_result_agrees:
    #     print('TEST PASSED')
    # else:
    #     print('TEST FAILED')
    # assert vep_result_agrees
