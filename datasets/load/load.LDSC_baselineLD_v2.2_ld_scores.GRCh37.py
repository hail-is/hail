
import hail as hl

root = 'gs://hail-datasets-raw-data/LDSC/baselineLD_v2.2'

mt = hl.import_matrix_table(f'{root}/ld_scores.GRCh37.tsv.bgz',
    row_fields={'CHR': hl.tstr, 'SNP': hl.tstr, 'BP': hl.tint}, entry_type=hl.tstr)

mt = mt.annotate_entries(x=hl.float(mt['x']))
mt = mt.annotate_rows(
    locus=hl.locus(mt['CHR'], mt['BP'], 'GRCh37'))
mt = mt.key_rows_by('locus')
mt = mt.select_rows('SNP')

M = hl.import_table(
    f'{root}/M.GRCh37.tsv.bgz', key='annotation')
M_5_50 = hl.import_table(
    f'{root}/M_5_50.GRCh37.tsv.bgz', key='annotation')

mt = mt.rename({'col_id': 'annotation'})
mt = mt.annotate_cols(
    M_5_50=hl.int(hl.float(M_5_50[mt.annotation].M_5_50)),
    M=hl.int(hl.float(M[mt.annotation].M)))

n_rows, n_cols = mt.count()
n_partitions = mt.n_partitions()

mt = mt.annotate_globals(
    metadata=hl.struct(
        name='LDSC_baselineLD_v2.2_ld_scores',
        reference_genome='GRCh37',
        n_rows=n_rows,
        n_cols=n_cols,
        n_partitions=n_partitions))

mt.write('gs://hail-datasets-hail-data/LDSC_baselineLD_v2.2_ld_scores.GRCh37.mt', overwrite=True)
