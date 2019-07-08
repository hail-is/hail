
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

ht_transcripts = hl.import_table('gs://hail-datasets/raw-data/gtex/v7/reference/gencode.v19.transcripts.patched_contigs.gtf',
                                 comment='#', no_header=True, types={'f3': hl.tint, 'f4': hl.tint}, missing='.', min_partitions=12)

ht_transcripts = ht_transcripts.rename({'f0': 'contig',
                                        'f1': 'annotation_source',
                                        'f2': 'feature_type',
                                        'f3': 'start',
                                        'f4': 'end',
                                        'f5': 'score',
                                        'f6': 'strand',
                                        'f7': 'phase',
                                        'f8': 'attributes'})

ht_transcripts = ht_transcripts.filter(ht_transcripts.feature_type == 'transcript')
ht_transcripts = ht_transcripts.annotate(interval=hl.interval(hl.locus(ht_transcripts.contig, ht_transcripts.start, 'GRCh37'), hl.locus(ht_transcripts.contig, ht_transcripts.end + 1, 'GRCh37')))
ht_transcripts = ht_transcripts.annotate(attributes=hl.dict(hl.map(lambda x: (x.split(' ')[0], x.split(' ')[1].replace('"', '').replace(';$', '')), ht_transcripts.attributes.split('; '))))
attribute_cols = list(ht_transcripts.aggregate(hl.set(hl.flatten(hl.agg.collect(ht_transcripts.attributes.keys())))))
ht_transcripts = ht_transcripts.annotate(**{x: hl.or_missing(ht_transcripts.attributes.contains(x), ht_transcripts.attributes[x]) for x in attribute_cols})
ht_transcripts = ht_transcripts.select(*(['transcript_id', 'transcript_name', 'transcript_type', 'strand', 'transcript_status', 'havana_transcript', 'ccdsid', 'ont', 'gene_name', 'interval', 'gene_type', 'annotation_source', 'havana_gene', 'gene_status', 'tag']))
ht_transcripts = ht_transcripts.rename({'havana_transcript': 'havana_transcript_id', 'havana_gene': 'havana_gene_id'})
ht_transcripts = ht_transcripts.key_by(ht_transcripts.transcript_id)

mt = hl.import_matrix_table('gs://hail-datasets/raw-data/gtex/v7/rna-seq/processed/GTEx_Analysis_2016-01-15_v7_RSEMv1.2.22_transcript_expected_count.tsv.bgz',
                            row_fields={'transcript_id': hl.tstr, 'gene_id': hl.tstr}, row_key='transcript_id', missing='', entry_type=hl.tfloat)

mt = mt.annotate_cols(sample_id=mt.col_id)
mt = mt.key_cols_by(mt.sample_id)

mt = mt.annotate_entries(read_count=hl.int(mt.x))
mt = mt.drop(mt.col_id, mt.x)

mt = mt.annotate_cols(**ht_samples[mt.sample_id])
mt = mt.annotate_rows(**ht_transcripts[mt.transcript_id])

mt.describe()
mt.write('gs://hail-datasets/hail-data/gtex_v7_transcript_read_counts.GRCh37.mt', overwrite=True)
