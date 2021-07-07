.. _dbSNP:

dbSNP
=====

*  **Versions:** 154
*  **Reference genome builds:** GRCh37, GRCh38
*  **Type:** :class:`hail.Table`

Schema (154, GRCh37)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        'metadata': struct {
            name: str,
            version: str,
            reference_genome: str,
            n_rows: int32,
            n_partitions: int32
        }
    ----------------------------------------
    Row fields:
        'locus': locus<GRCh37>
        'alleles': array<str>
        'rsid': str
        'qual': float64
        'filters': set<str>
        'info': struct {
            RS: int32,
            GENEINFO: str,
            PSEUDOGENEINFO: str,
            dbSNPBuildID: int32,
            SAO: int32,
            SSR: int32,
            VC: str,
            PM: bool,
            NSF: bool,
            NSM: bool,
            NSN: bool,
            SYN: bool,
            U3: bool,
            U5: bool,
            ASS: bool,
            DSS: bool,
            INT: bool,
            R3: bool,
            R5: bool,
            GNO: bool,
            PUB: bool,
            FREQ: struct {
                _GENOME_DK: float64,
                _TWINSUK: float64,
                _dbGaP_PopFreq: float64,
                _Siberian: float64,
                _Chileans: float64,
                _FINRISK: float64,
                _HapMap: float64,
                _Estonian: float64,
                _ALSPAC: float64,
                _GoESP: float64,
                _TOPMED: float64,
                _PAGE_STUDY: float64,
                _1000Genomes: float64,
                _Korea1K: float64,
                _ChromosomeY: float64,
                _ExAC: float64,
                _Qatari: float64,
                _GoNL: float64,
                _MGP: float64,
                _GnomAD: float64,
                _Vietnamese: float64,
                _GnomAD_exomes: float64,
                _PharmGKB: float64,
                _KOREAN: float64,
                _Daghestan: float64,
                _HGDP_Stanford: float64,
                _NorthernSweden: float64,
                _SGDP_PRJ: float64
            },
            COMMON: bool,
            CLNHGVS: array<str>,
            CLNVI: array<str>,
            CLNORIGIN: array<str>,
            CLNSIG: array<str>,
            CLNDISDB: array<str>,
            CLNDN: array<str>,
            CLNREVSTAT: array<str>,
            CLNACC: array<str>
        }
        'a_index': int32
        'was_split': bool
        'old_locus': locus<GRCh37>
        'old_alleles': array<str>
    ----------------------------------------
    Key: ['locus', 'alleles']
    ----------------------------------------
