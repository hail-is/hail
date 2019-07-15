
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

ht_genes = hl.import_table('gs://hail-datasets/raw-data/gtex/v7/reference/gencode.v19.genes.v7.patched_contigs.gtf',
                           comment='#', no_header=True, types={'f3': hl.tint, 'f4': hl.tint}, missing='.', min_partitions=12)

ht_genes = ht_genes.rename({'f0': 'contig',
                            'f1': 'annotation_source',
                            'f2': 'feature_type',
                            'f3': 'start',
                            'f4': 'end',
                            'f5': 'score',
                            'f6': 'strand',
                            'f7': 'phase',
                            'f8': 'attributes'})

ht_genes = ht_genes.filter(ht_genes.feature_type == 'gene')
ht_genes = ht_genes.annotate(interval=hl.interval(hl.locus(ht_genes.contig, ht_genes.start, 'GRCh37'), hl.locus(ht_genes.contig, ht_genes.end + 1, 'GRCh37')))
ht_genes = ht_genes.annotate(attributes=hl.dict(hl.map(lambda x: (x.split(' ')[0], x.split(' ')[1].replace('"', '').replace(';$', '')), ht_genes.attributes.split('; '))))
attribute_cols = list(ht_genes.aggregate(hl.set(hl.flatten(hl.agg.collect(ht_genes.attributes.keys())))))
ht_genes = ht_genes.annotate(**{x: hl.or_missing(ht_genes.attributes.contains(x), ht_genes.attributes[x]) for x in attribute_cols})
ht_genes = ht_genes.select(*(['gene_id', 'interval', 'gene_type', 'strand', 'annotation_source', 'havana_gene', 'gene_status', 'tag']))
ht_genes = ht_genes.rename({'havana_gene': 'havana_gene_id'})
ht_genes = ht_genes.key_by(ht_genes.gene_id)

mt = hl.import_matrix_table('gs://hail-datasets/raw-data/gtex/v7/rna-seq/processed/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_reads.tsv.bgz',
                            row_fields={'Name': hl.tstr, 'Description': hl.tstr}, row_key='Name', missing=' ', entry_type=hl.tfloat)

mt = mt.annotate_cols(sample_id=mt.col_id)
mt = mt.key_cols_by(mt.sample_id)

mt = mt.annotate_rows(gene_id=mt.Name, gene_name=mt.Description)
mt = mt.key_rows_by(mt.gene_id)

mt = mt.annotate_entries(read_count=hl.int(mt.x))
mt = mt.drop(mt.col_id, mt.Name, mt.Description, mt.x)

mt = mt.annotate_cols(**ht_samples[mt.sample_id])
mt = mt.annotate_rows(**ht_genes[mt.gene_id])

mt.describe()
mt.write('gs://hail-datasets/hail-data/gtex_v7_gene_read_counts.GRCh37.mt', overwrite=True)
