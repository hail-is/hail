.. _gnomad_hgdp_1kg_subset_sparse:

gnomad_hgdp_1kg_subset_sparse
=============================

*  **Versions:** 3.1.2
*  **Reference genome builds:** GRCh38
*  **Type:** :class:`hail.MatrixTable`

Schema (3.1.2, GRCh38)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        None
    ----------------------------------------
    Column fields:
        's': str
    ----------------------------------------
    Row fields:
        'locus': locus<GRCh38>
        'alleles': array<str>
        'rsid': str
    ----------------------------------------
    Entry fields:
        'DP': int32
        'END': int32
        'GQ': int32
        'LA': array<int32>
        'LAD': array<int32>
        'LGT': call
        'LPGT': call
        'LPL': array<int32>
        'MIN_DP': int32
        'PID': str
        'RGQ': int32
        'SB': array<int32>
        'gvcf_info': struct {
            ClippingRankSum: float64,
            BaseQRankSum: float64,
            MQ: float64,
            MQRankSum: float64,
            MQ_DP: int32,
            QUALapprox: int32,
            RAW_MQ: float64,
            ReadPosRankSum: float64,
            VarDP: int32
        }
    ----------------------------------------
    Column key: ['s']
    Row key: ['locus', 'alleles']
    ----------------------------------------
