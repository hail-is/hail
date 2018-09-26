.. _dbsnp_build151:

dbsnp_build151
==============

.. code-block:: text

    ----------------------------------------
    Global fields:
    None
    ----------------------------------------
    Row fields:
    'locus': locus<GRCh37> 
    'alleles': array<str> 
    'rsid': str 
    'qual': float64 
    'filters': set<str> 
    'info': struct {
        RS: int32, 
        RSPOS: int32, 
        RV: bool, 
        VP: str, 
        GENEINFO: str, 
        dbSNPBuildID: int32, 
        SAO: int32, 
        SSR: int32, 
        WGT: int32, 
        VC: str, 
        PM: bool, 
        TPA: bool, 
        PMC: bool, 
        S3D: bool, 
        SLO: bool, 
        NSF: bool, 
        NSM: bool, 
        NSN: bool, 
        REF: bool, 
        SYN: bool, 
        U3: bool, 
        U5: bool, 
        ASS: bool, 
        DSS: bool, 
        INT: bool, 
        R3: bool, 
        R5: bool, 
        OTH: bool, 
        CFL: bool, 
        ASP: bool, 
        MUT: bool, 
        VLD: bool, 
        G5A: bool, 
        G5: bool, 
        HD: bool, 
        GNO: bool, 
        KGPhase1: bool, 
        KGPhase3: bool, 
        CDA: bool, 
        LSD: bool, 
        MTP: bool, 
        OM: bool, 
        NOC: bool, 
        WTD: bool, 
        NOV: bool, 
        CAF: str, 
        COMMON: int32, 
        TOPMED: str
    } 
    'a_index': int32 
    'was_split': bool 
    ----------------------------------------
    Key:['locus', 'alleles']
    ----------------------------------------
