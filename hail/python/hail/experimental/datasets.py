
import json
import hail as hl


def load_dataset(dataset_name,
                 reference_genome,
                 config_file='gs://hail-datasets/hail_datasets.config.json'):
    """Load a Hail-formatted genetic dataset.

    Example
    -------

    >>> # Load 1000 Genomes chromosome X MatrixTable with GRCh38 coordinates
    >>> mt_1kg = hl.experimental.load_dataset('1000_genomes_phase3_chrX',
    ...                                       reference_genome='GRCh38')

    >>> # Count rows and columns
    >>> mt_1kg.count()

    .. code-block:: text

        (3474116, 2504)

    >>> # Inspect contents of the MatrixTable
    >>> mt_1kg.describe()

    .. code-block:: text

        ----------------------------------------
        Global fields:
            None
        ----------------------------------------
        Column fields:
            's': str
            'sex': str
            'super_population': str
            'population': str
            'sample_qc': struct {
                call_rate: float64,
                n_called: int64,
                n_not_called: int64,
                n_hom_ref: int64,
                n_het: int64,
                n_hom_var: int64,
                n_non_ref: int64,
                n_singleton: int64,
                n_snp: int64,
                n_insertion: int64,
                n_deletion: int64,
                n_transition: int64,
                n_transversion: int64,
                n_star: int64,
                r_ti_tv: float64,
                r_het_hom_var: float64,
                r_insertion_deletion: float64
            }
        ----------------------------------------
        Row fields:
            'alleles': array<str>
            'rsid': str
            'qual': float64
            'filters': set<str>
            'info': struct {
                CIEND: array<int32>,
                CIPOS: array<int32>,
                CS: str,
                END: int32,
                IMPRECISE: bool,
                MC: array<str>,
                MEINFO: array<str>,
                MEND: int32,
                MLEN: int32,
                MSTART: int32,
                SVLEN: array<int32>,
                SVTYPE: str,
                TSD: str,
                AC: int32,
                AF: float64,
                NS: int32,
                AN: int32,
                EAS_AF: float64,
                EUR_AF: float64,
                AFR_AF: float64,
                AMR_AF: float64,
                SAS_AF: float64,
                DP: int32,
                AA: str,
                OLD_VARIANT: array<str>,
                VT: str,
                EX_TARGET: bool,
                MULTI_ALLELIC: bool
            }
            'was_split': bool
            'cm_position': float64
            'recombination_rate_cm_per_mb': float64
            'variant_qc': struct {
                AC: array<int32>,
                AF: array<float64>,
                AN: int32,
                homozygote_count: array<int32>,
                n_called: int64,
                n_not_called: int64,
                call_rate: float32,
                n_het: int64,
                n_non_ref: int64,
                het_freq_hwe: float64,
                p_value_hwe: float64
            }
            'vep': struct {
                assembly_name: str,
                allele_string: str,
                ancestral: str,
                colocated_variants: array<struct {
                    aa_allele: str,
                    aa_maf: float64,
                    afr_allele: str,
                    afr_maf: float64,
                    allele_string: str,
                    amr_allele: str,
                    amr_maf: float64,
                    clin_sig: array<str>,
                    end: int32,
                    eas_allele: str,
                    eas_maf: float64,
                    ea_allele: str,
                    ea_maf: float64,
                    eur_allele: str,
                    eur_maf: float64,
                    exac_adj_allele: str,
                    exac_adj_maf: float64,
                    exac_allele: str,
                    exac_afr_allele: str,
                    exac_afr_maf: float64,
                    exac_amr_allele: str,
                    exac_amr_maf: float64,
                    exac_eas_allele: str,
                    exac_eas_maf: float64,
                    exac_fin_allele: str,
                    exac_fin_maf: float64,
                    exac_maf: float64,
                    exac_nfe_allele: str,
                    exac_nfe_maf: float64,
                    exac_oth_allele: str,
                    exac_oth_maf: float64,
                    exac_sas_allele: str,
                    exac_sas_maf: float64,
                    id: str,
                    minor_allele: str,
                    minor_allele_freq: float64,
                    phenotype_or_disease: int32,
                    pubmed: array<int32>,
                    sas_allele: str,
                    sas_maf: float64,
                    somatic: int32,
                    start: int32,
                    strand: int32
                }>,
                context: str,
                end: int32,
                id: str,
                input: str,
                intergenic_consequences: array<struct {
                    allele_num: int32,
                    consequence_terms: array<str>,
                    impact: str,
                    minimised: int32,
                    variant_allele: str
                }>,
                most_severe_consequence: str,
                motif_feature_consequences: array<struct {
                    allele_num: int32,
                    consequence_terms: array<str>,
                    high_inf_pos: str,
                    impact: str,
                    minimised: int32,
                    motif_feature_id: str,
                    motif_name: str,
                    motif_pos: int32,
                    motif_score_change: float64,
                    strand: int32,
                    variant_allele: str
                }>,
                regulatory_feature_consequences: array<struct {
                    allele_num: int32,
                    biotype: str,
                    consequence_terms: array<str>,
                    impact: str,
                    minimised: int32,
                    regulatory_feature_id: str,
                    variant_allele: str
                }>,
                seq_region_name: str,
                start: int32,
                strand: int32,
                transcript_consequences: array<struct {
                    allele_num: int32,
                    amino_acids: str,
                    biotype: str,
                    canonical: int32,
                    ccds: str,
                    cdna_start: int32,
                    cdna_end: int32,
                    cds_end: int32,
                    cds_start: int32,
                    codons: str,
                    consequence_terms: array<str>,
                    distance: int32,
                    domains: array<struct {
                        db: str,
                        name: str
                    }>,
                    exon: str,
                    gene_id: str,
                    gene_pheno: int32,
                    gene_symbol: str,
                    gene_symbol_source: str,
                    hgnc_id: str,
                    hgvsc: str,
                    hgvsp: str,
                    hgvs_offset: int32,
                    impact: str,
                    intron: str,
                    lof: str,
                    lof_flags: str,
                    lof_filter: str,
                    lof_info: str,
                    minimised: int32,
                    polyphen_prediction: str,
                    polyphen_score: float64,
                    protein_end: int32,
                    protein_start: int32,
                    protein_id: str,
                    sift_prediction: str,
                    sift_score: float64,
                    strand: int32,
                    swissprot: str,
                    transcript_id: str,
                    trembl: str,
                    uniparc: str,
                    variant_allele: str
                }>,
                variant_class: str
            }
            'locus': locus<GRCh38>
        ----------------------------------------
        Entry fields:
            'GT': call
        ----------------------------------------
        Column key: ['s']
        Row key: ['locus', 'alleles']
        Partition key: ['locus']
        ----------------------------------------

    Parameters
    ----------
    dataset_name : :obj:`str`
        Name of the dataset to load.
    reference_genome : `GRCh37` or `GRCh38`
        Reference genome build.
    config_file : :obj:`str`, optional
        Path of the datasets configuration file.
        Leave as default if running on Google Cloud Platform.

    Returns
    -------
    :class:`.Table` or :class:`.MatrixTable`"""

    with hl.hadoop_open(config_file, 'r') as f:
        config = json.load(f)

    builds = [{'path': x['path'], 'reference_genome': x['reference_genome']}
              for x in config if x['name'] == dataset_name]

    if not builds:
        raise NameError("Dataset '{}' not found.".format(dataset_name))

    path = [x['path'] for x in builds
            if x['reference_genome'] == reference_genome]
    if not path:
        raise ValueError("""Reference genome '{0}' not available for dataset '{1}'. Available reference genomes: '{2}'.""".format(
                         reference_genome,
                         dataset_name,
                         ', '.join([x['reference_genome'] for x in builds])))
    else:
        path = path[0].strip('/')

    if path.endswith('.ht'):
        dataset = hl.read_table(path)
    else:
        assert path.endswith('.mt')
        dataset = hl.read_matrix_table(path)

    return dataset
