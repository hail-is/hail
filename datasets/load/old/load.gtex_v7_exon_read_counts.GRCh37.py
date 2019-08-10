
import hail as hl

ht_samples = hl.import_table('gs://hail-datasets/raw-data/gtex/v7/annotations/GTEx_v7_Annotations.SampleAttributesDS.txt', 
                             key='SAMPID',
                             missing='')

float_cols = ['SMRIN',
              'SME2MPRT',
              'SMNTRART',
              'SMMAPRT',
              'SMEXNCRT',
              'SM550NRM',
              'SMUNMPRT',
              'SM350NRM',
              'SMMNCPB',
              'SME1MMRT',
              'SMNTERRT',
              'SMMNCV',
              'SMGAPPCT',
              'SMNTRNRT',
              'SMMPUNRT',
              'SMEXPEFF',
              'SME2MMRT',
              'SMBSMMRT',
              'SME1PCTS',
              'SMRRNART',
              'SME1MPRT',
              'SMDPMPRT',
              'SME2PCTS']

int_cols = ['SMTSISCH',
            'SMATSSCR',
            'SMTSPAX',
            'SMCHMPRS',
            'SMNUMGPS',
            'SMGNSDTC',
            'SMRDLGTH',
            'SMSFLGTH',
            'SMESTLBS',
            'SMMPPD',
            'SMRRNANM',
            'SMVQCFL',
            'SMTRSCPT',
            'SMMPPDPR',
            'SMCGLGTH',
            'SMUNPDRD',
            'SMMPPDUN',
            'SME2ANTI',
            'SMALTALG',
            'SME2SNSE',
            'SMMFLGTH',
            'SMSPLTRD',
            'SME1ANTI',
            'SME1SNSE',
            'SMNUM5CD']

ht_samples = ht_samples.annotate(**{x: hl.float(ht_samples[x]) for x in float_cols})
ht_samples = ht_samples.annotate(**{x: hl.int(ht_samples[x].replace('.0$', '')) for x in int_cols})

ht_exons = hl.import_table('gs://hail-datasets/raw-data/gtex/v7/reference/gencode.v19.genes.v7.patched_contigs.exons.txt',
                           types={'start_pos': hl.tint, 'end_pos': hl.tint}, missing='.', min_partitions=12)
ht_exons = ht_exons.annotate(interval=hl.interval(hl.locus(ht_exons.chr, ht_exons.start_pos, 'GRCh37'),
                                                  hl.locus(ht_exons.chr, ht_exons.end_pos + 1, 'GRCh37')))
ht_exons = ht_exons.select(ht_exons.exon_id, ht_exons.interval, ht_exons.strand)
ht_exons = ht_exons.key_by(ht_exons.exon_id)

mt = hl.import_matrix_table('gs://hail-datasets/raw-data/gtex/v7/rna-seq/processed/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_exon_reads.tsv.bgz',
                            row_fields={'exon_id': hl.tstr}, row_key='exon_id', missing='', entry_type=hl.tfloat)

mt = mt.annotate_cols(sample_id=mt.col_id)
mt = mt.key_cols_by(mt.sample_id)

mt = mt.annotate_entries(read_count=hl.int(mt.x))
mt = mt.drop(mt.col_id, mt.x)

mt = mt.annotate_cols(**ht_samples[mt.sample_id])
mt = mt.annotate_rows(**ht_exons[mt.exon_id])

mt.describe()
mt.write('gs://hail-datasets/hail-data/gtex_v7_exon_read_counts.GRCh37.mt', overwrite=True)
