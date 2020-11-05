.. _GTEx_RNA_seq_gene_TPMs:

GTEx_RNA_seq_gene_TPMs
======================

*  **Versions:** v7
*  **Reference genome builds:** GRCh37
*  **Type:** :class:`hail.MatrixTable`

Schema (v7, GRCh37)
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        'metadata': struct {
            name: str,
            version: str,
            reference_genome: str,
            n_rows: int32,
            n_cols: int32,
            n_partitions: int32
        }
    ----------------------------------------
    Column fields:
        's': str
        'subject_id': str
        'SMATSSCR': float64
        'SMCENTER': str
        'SMPTHNTS': str
        'SMRIN': float64
        'SMTS': str
        'SMTSD': str
        'SMUBRID': str
        'SMTSISCH': float64
        'SMTSPAX': float64
        'SMNABTCH': str
        'SMNABTCHT': str
        'SMNABTCHD': str
        'SMGEBTCH': str
        'SMGEBTCHD': str
        'SMGEBTCHT': str
        'SMAFRZE': str
        'SMGTC': str
        'SME2MPRT': float64
        'SMCHMPRS': float64
        'SMNTRART': float64
        'SMNUMGPS': str
        'SMMAPRT': float64
        'SMEXNCRT': float64
        'SM550NRM': str
        'SMGNSDTC': float64
        'SMUNMPRT': float64
        'SM350NRM': str
        'SMRDLGTH': float64
        'SMMNCPB': str
        'SME1MMRT': float64
        'SMSFLGTH': float64
        'SMESTLBS': float64
        'SMMPPD': float64
        'SMNTERRT': float64
        'SMRRNANM': float64
        'SMRDTTL': float64
        'SMVQCFL': float64
        'SMMNCV': str
        'SMTRSCPT': float64
        'SMMPPDPR': float64
        'SMCGLGTH': str
        'SMGAPPCT': str
        'SMUNPDRD': float64
        'SMNTRNRT': float64
        'SMMPUNRT': float64
        'SMEXPEFF': float64
        'SMMPPDUN': float64
        'SME2MMRT': float64
        'SME2ANTI': float64
        'SMALTALG': float64
        'SME2SNSE': float64
        'SMMFLGTH': float64
        'SME1ANTI': float64
        'SMSPLTRD': float64
        'SMBSMMRT': float64
        'SME1SNSE': float64
        'SME1PCTS': float64
        'SMRRNART': float64
        'SME1MPRT': float64
        'SMNUM5CD': str
        'SMDPMPRT': float64
        'SME2PCTS': float64
        'is_female': bool
        'age_range': str
        'death_classification_hardy_scale': str
    ----------------------------------------
    Row fields:
        'gene_id': str
        'gene_symbol': str
        'gene_interval': interval<locus<GRCh37>>
        'source': str
        'havana_gene_id': str
        'gene_type': str
        'gene_status': str
        'level': str
        'score': float64
        'strand': str
        'frame': int32
        'tag': str
    ----------------------------------------
    Entry fields:
        'TPM': float64
    ----------------------------------------
    Column key: ['s']
    Row key: ['gene_id']
    ----------------------------------------

