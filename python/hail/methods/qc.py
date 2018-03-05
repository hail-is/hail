from hail.typecheck import *
from hail.utils.java import Env
from hail.matrixtable import MatrixTable
from hail.table import Table
from .misc import require_biallelic, require_row_key_variant, require_col_key_str


@typecheck(dataset=MatrixTable, name=str)
def sample_qc(dataset, name='sample_qc'):
    """Compute per-sample metrics useful for quality control.

    .. include:: ../_templates/req_tvariant.rst

    Examples
    --------

    Compute sample QC metrics and remove low-quality samples:

    >>> dataset = hl.sample_qc(dataset, name='sample_qc')
    >>> filtered_dataset = dataset.filter_cols((dataset.sample_qc.dp_mean > 20) & (dataset.sample_qc.r_ti_tv > 1.5))

    Notes
    -----

    This method computes summary statistics per sample from a genetic matrix and stores the results as
    a new column-indexed field in the matrix, named based on the `name` parameter.

    +--------------------------+-------+-+------------------------------------------------------+
    | Name                     | Type    | Description                                          |
    +==========================+=========+======================================================+
    | ``call_rate``            | float64 | Fraction of calls non-missing                        |
    +--------------------------+---------+------------------------------------------------------+
    | ``n_hom_ref``            | int64   | Number of homozygous reference calls                 |
    +--------------------------+---------+------------------------------------------------------+
    | ``n_het``                | int64   | Number of heterozygous calls                         |
    +--------------------------+---------+------------------------------------------------------+
    | ``n_hom_var``            | int64   | Number of homozygous alternate calls                 |
    +--------------------------+---------+------------------------------------------------------+
    | ``n_called``             | int64   | Sum of ``n_hom_ref`` + ``n_het`` + ``n_hom_var``     |
    +--------------------------+---------+------------------------------------------------------+
    | ``n_not_called``         | int64   | Number of missing calls                              |
    +--------------------------+---------+------------------------------------------------------+
    | ``n_snp``                | int64   | Number of SNP alternate alleles                      |
    +--------------------------+---------+------------------------------------------------------+
    | ``n_insertion``          | int64   | Number of insertion alternate alleles                |
    +--------------------------+---------+------------------------------------------------------+
    | ``n_deletion``           | int64   | Number of deletion alternate alleles                 |
    +--------------------------+---------+------------------------------------------------------+
    | ``n_singleton``          | int64   | Number of private alleles                            |
    +--------------------------+---------+------------------------------------------------------+
    | ``n_transition``         | int64   | Number of transition (A-G, C-T) alternate alleles    |
    +--------------------------+---------+------------------------------------------------------+
    | ``n_transversion``       | int64   | Number of transversion alternate alleles             |
    +--------------------------+---------+------------------------------------------------------+
    | ``n_star``               | int64   | Number of star (upstream deletion) alleles           |
    +--------------------------+---------+------------------------------------------------------+
    | ``n_non_ref``            | int64   | Sum of ``n_het`` and ``n_hom_var``                   |
    +--------------------------+---------+------------------------------------------------------+
    | ``r_ti_tv``              | float64 | Transition/Transversion ratio                        |
    +--------------------------+---------+------------------------------------------------------+
    | ``r_het_hom_var``        | float64 | Het/HomVar call ratio                                |
    +--------------------------+---------+------------------------------------------------------+
    | ``r_insertion_deletion`` | float64 | Insertion/Deletion allele ratio                      |
    +--------------------------+---------+------------------------------------------------------+
    | ``dp_mean``              | float64 | Depth mean across all calls                          |
    +--------------------------+---------+------------------------------------------------------+
    | ``dp_stdev``             | float64 | Depth standard deviation across all calls            |
    +--------------------------+---------+------------------------------------------------------+
    | ``gq_mean``              | float64 | The average genotype quality across all calls        |
    +--------------------------+---------+------------------------------------------------------+
    | ``gq_stdev``             | float64 | Genotype quality standard deviation across all calls |
    +--------------------------+---------+------------------------------------------------------+

    Missing values ``NA`` may result from division by zero. The empirical
    standard deviation is computed with zero degrees of freedom.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset.
    name : :obj:`str`
        Name for resulting field.

    Returns
    -------
    :class:`.MatrixTable`
        Dataset with a new column-indexed field `name`.
    """

    return MatrixTable(Env.hail().methods.SampleQC.apply(require_biallelic(dataset, 'sample_qc')._jvds, name))


@typecheck(dataset=MatrixTable, name=str)
def variant_qc(dataset, name='variant_qc'):
    """Compute common variant statistics (quality control metrics).

    .. include:: ../_templates/req_biallelic.rst
    .. include:: ../_templates/req_tvariant.rst

    Examples
    --------

    >>> dataset_result = hl.variant_qc(dataset)

    Notes
    -----
    This method computes 18 variant statistics from the genotype data,
    returning a new struct field `name` with the following metrics:

    +-------------------------+---------+--------------------------------------------------------+
    | Name                    | Type    | Description                                            |
    +=========================+=========+========================================================+
    | ``call_rate``           | float64 | Fraction of samples with called genotypes              |
    +-------------------------+---------+--------------------------------------------------------+
    | ``AF``                  | float64 | Calculated alternate allele frequency (q)              |
    +-------------------------+---------+--------------------------------------------------------+
    | ``AC``                  | int32   | Count of alternate alleles                             |
    +-------------------------+---------+--------------------------------------------------------+
    | ``r_heterozygosity``    | float64 | Proportion of heterozygotes                            |
    +-------------------------+---------+--------------------------------------------------------+
    | ``r_het_hom_var``       | float64 | Ratio of heterozygotes to homozygous alternates        |
    +-------------------------+---------+--------------------------------------------------------+
    | ``r_expected_het_freq`` | float64 | Expected r_heterozygosity based on HWE                 |
    +-------------------------+---------+--------------------------------------------------------+
    | ``p_hwe``               | float64 | p-value from Hardy Weinberg Equilibrium null model     |
    +-------------------------+---------+--------------------------------------------------------+
    | ``n_hom_ref``           | int32   | Number of homozygous reference samples                 |
    +-------------------------+---------+--------------------------------------------------------+
    | ``n_het``               | int32   | Number of heterozygous samples                         |
    +-------------------------+---------+--------------------------------------------------------+
    | ``n_hom_var``           | int32   | Number of homozygous alternate samples                 |
    +-------------------------+---------+--------------------------------------------------------+
    | ``n_called``            | int32   | Sum of ``n_hom_ref``, ``n_het``, and ``n_hom_var``     |
    +-------------------------+---------+--------------------------------------------------------+
    | ``n_not_called``        | int32   | Number of uncalled samples                             |
    +-------------------------+---------+--------------------------------------------------------+
    | ``n_non_ref``           | int32   | Sum of ``n_het`` and ``n_hom_var``                     |
    +-------------------------+---------+--------------------------------------------------------+
    | ``dp_mean``             | float64 | Depth mean across all samples                          |
    +-------------------------+---------+--------------------------------------------------------+
    | ``dp_stdev``            | float64 | Depth standard deviation across all samples            |
    +-------------------------+---------+--------------------------------------------------------+
    | ``gq_mean``             | float64 | The average genotype quality across all samples        |
    +-------------------------+---------+--------------------------------------------------------+
    | ``gq_stdev``            | float64 | Genotype quality standard deviation across all samples |
    +-------------------------+---------+--------------------------------------------------------+

    Missing values ``NA`` may result from division by zero. The empirical
    standard deviation is computed with zero degrees of freedom.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset.
    name : :obj:`str`
        Name for resulting field.

    Returns
    -------
    :class:`.MatrixTable`
        Dataset with a new row-indexed field `name`.
    """

    return MatrixTable(Env.hail().methods.VariantQC.apply(require_biallelic(dataset, 'variant_qc')._jvds, name))


@typecheck(left=MatrixTable,
           right=MatrixTable)
def concordance(left, right):
    """Calculate call concordance with another dataset.

    .. include:: ../_templates/req_tvariant.rst

    .. include:: ../_templates/req_biallelic.rst

    .. testsetup::

        dataset2 = dataset

    Examples
    --------

    Compute concordance between two datasets and output the global concordance
    statistics and two tables with concordance computed per column key and per
    row key:

    >>> global_conc, cols_conc, rows_conc = hl.concordance(dataset, dataset2)

    Notes
    -----

    This method computes the genotype call concordance (from the entry
    field **GT**) between two biallelic variant datasets.  It requires
    unique sample IDs and performs an inner join on samples (only
    samples in both datasets will be considered).

    It performs an ordered zip join of the variants.  That means the
    variants of each dataset are sorted, with duplicate variants
    appearing in some random relative order, and then zipped together.
    When a variant appears a different number of times between the two
    datasets, the dataset with the fewer number of instances is padded
    with "no data".  For example, if a variant is only in one dataset,
    then each genotype is treated as "no data" in the other.

    This method returns a tuple of three objects: a nested list of
    list of int with global concordance summary statistics, a table
    with concordance statistics per column key, and a table with
    concordance statistics per row key.

    **Using the global summary result**

    The global summary is a list of list of int (conceptually a 5 by 5 matrix),
    where the indices have special meaning:

    0. No Data (missing variant)
    1. No Call (missing genotype call)
    2. Hom Ref
    3. Heterozygous
    4. Hom Var

    The first index is the state in the left dataset and the second index is
    the state in the right dataset. Typical uses of the summary list are shown
    below.

    >>> summary, samples, variants = hl.concordance(dataset, dataset2)
    >>> left_homref_right_homvar = summary[2][4]
    >>> left_het_right_missing = summary[3][1]
    >>> left_het_right_something_else = sum(summary[3][:]) - summary[3][3]
    >>> total_concordant = summary[2][2] + summary[3][3] + summary[4][4]
    >>> total_discordant = sum([sum(s[2:]) for s in summary[2:]]) - total_concordant

    **Using the table results**

    Table 1: Concordance statistics by column

    This table contains the column key field of `left`, and the following fields:

        - `n_discordant` (:py:data:`.tint64`) -- Count of discordant calls (see below for
          full definition).
        - `concordance` (:class:`.tarray` of :class:`.tarray` of :py:data:`.tint64`) --
          Array of concordance per state on left and right, matching the structure of
          the global summary defined above.

    Table 2: Concordance statistics by row

    This table contains the row key fields of `left`, and the following fields:

        - `n_discordant` (:py:data:`.tfloat64`) -- Count of discordant calls (see below for
          full definition).
        - `concordance` (:class:`.tarray` of :class:`.tarray` of :py:data:`.tint64`) --
          Array of concordance per state on left and right, matching the structure of the
          global summary defined above.

    In these tables, the column **n_discordant** is provided as a convenience,
    because this is often one of the most useful concordance statistics. This
    value is the number of genotypes which were called (homozygous reference,
    heterozygous, or homozygous variant) in both datasets, but where the call
    did not match between the two.

    The column `concordance` matches the structure of the global summmary,
    which is detailed above. Once again, the first index into this array is the
    state on the left, and the second index is the state on the right. For
    example, ``concordance[1][4]`` is the number of "no call" genotypes on the
    left that were called homozygous variant on the right.

    Parameters
    ----------
    left : :class:`.MatrixTable`
        First dataset to compare.
    right : :class:`.MatrixTable`
        Second dataset to compare.

    Returns
    -------
    (list of list of int, :class:`.Table`, :class:`.Table`)
        The global concordance statistics, a table with concordance statistics
        per column key, and a table with concordance statistics per row key.

    """

    require_col_key_str(left, 'concordance, left')
    require_col_key_str(right, 'concordance, right')
    left = require_biallelic(left, "concordance, left")
    right = require_biallelic(right, "concordance, right")

    r = Env.hail().methods.CalculateConcordance.apply(left._jvds, right._jvds)
    j_global_conc = r._1()
    col_conc = Table(r._2())
    row_conc = Table(r._3())
    global_conc = [[j_global_conc.apply(j).apply(i) for i in range(5)] for j in range(5)]

    return global_conc, col_conc, row_conc


@typecheck(dataset=MatrixTable,
           config=str,
           block_size=int,
           name=str,
           csq=bool)
def vep(dataset, config, block_size=1000, name='vep', csq=False):
    """Annotate variants with VEP.

    .. include:: ../_templates/req_tvariant.rst

    :func:`.vep` runs `Variant Effect Predictor
    <http://www.ensembl.org/info/docs/tools/vep/index.html>`__ with the `LOFTEE
    plugin <https://github.com/konradjk/loftee>`__ on the current dataset and
    adds the result as a row field.

    Examples
    --------

    Add VEP annotations to the dataset:

    >>> result = hl.vep(dataset, "data/vep.properties") # doctest: +SKIP

    Notes
    -----

    **Configuration**

    :func:`.vep` needs a configuration file to tell it
    how to run VEP. The format is a `.properties file
    <https://en.wikipedia.org/wiki/.properties>`__. Roughly, each line defines a
    property as a key-value pair of the form `key = value`. :func:`.vep` supports the
    following properties:

    - **hail.vep.perl** -- Location of Perl. Optional, default: perl.
    - **hail.vep.perl5lib** -- Value for the PERL5LIB environment variable when
      invoking VEP. Optional, by default PERL5LIB is not set.
    - **hail.vep.path** -- Value of the PATH environment variable when invoking
      VEP.  Optional, by default PATH is not set.
    - **hail.vep.location** -- Location of the VEP Perl script.  Required.
    - **hail.vep.cache_dir** -- Location of the VEP cache dir, passed to VEP
      with the ``--dir`` option. Required.
    - **hail.vep.fasta** -- Location of the FASTA file to use to look up the
      reference sequence, passed to VEP with the `--fasta` option. Required.
    - **hail.vep.assembly** -- Genome assembly version to use. Optional,
      default: GRCh37
    - **hail.vep.plugin** -- VEP plugin, passed to VEP with the `--plugin`
      option. Optional. Overrides `hail.vep.lof.human_ancestor` and
      `hail.vep.lof.conservation_file`.
    - **hail.vep.lof.human_ancestor** -- Location of the human ancestor file for
      the LOFTEE plugin. Ignored if `hail.vep.plugin` is set. Required otherwise.
    - **hail.vep.lof.conservation_file** -- Location of the conservation file
      for the LOFTEE plugin. Ignored if `hail.vep.plugin` is set. Required
      otherwise.

    Here is an example ``vep.properties`` configuration file

    .. code-block:: text

        hail.vep.perl = /usr/bin/perl
        hail.vep.path = /usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
        hail.vep.location = /path/to/vep/ensembl-tools-release-81/scripts/variant_effect_predictor/variant_effect_predictor.pl
        hail.vep.cache_dir = /path/to/vep
        hail.vep.lof.human_ancestor = /path/to/loftee_data/human_ancestor.fa.gz
        hail.vep.lof.conservation_file = /path/to/loftee_data/phylocsf.sql

    **VEP Invocation**

    .. code-block:: text

        <hail.vep.perl>
        <hail.vep.location>
        --format vcf
        --json
        --everything
        --allele_number
        --no_stats
        --cache --offline
        --dir <hail.vep.cache_dir>
        --fasta <hail.vep.fasta>
        --minimal
        --assembly <hail.vep.assembly>
        --plugin LoF,\
        human_ancestor_fa:$<hail.vep.lof.human_ancestor>,\
        filter_position:0.05,\
        min_intron_size:15,\
        conservation_file:<hail.vep.lof.conservation_file>
        -o STDOUT

    **Annotations**

    A new row field is added in the location specified by `name` with the
    following schema:

    .. code-block:: text

        struct {
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

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset.
    config : :obj:`str`
        Path to VEP configuration file.
    block_size : :obj:`int`
        Number of rows to process per VEP invocation.
    name : :obj:`str`
        Name for resulting row field.
    csq : :obj:`bool`
        If ``True``, annotates VCF CSQ field as a :py:data:`.tstr`.
        If ``False``, annotates with the full nested struct schema.

    Returns
    -------
    :class:`.MatrixTable`
        Dataset with new row-indexed field `name` containing VEP annotations.
    """

    require_row_key_variant(dataset, 'vep')
    return MatrixTable(Env.hail().methods.VEP.apply(dataset._jvds, config, 'va.`{}`'.format(name), csq, block_size))


@typecheck_method(dataset=MatrixTable,
                  config=str,
                  block_size=int,
                  name=str)
def nirvana(dataset, config, block_size=500000, name='nirvana'):
    """Annotate variants using `Nirvana <https://github.com/Illumina/Nirvana>`_.

    .. include:: ../_templates/experimental.rst

    .. include:: ../_templates/req_tvariant.rst

    :func:`.nirvana` runs `Nirvana
    <https://github.com/Illumina/Nirvana>`_ on the current dataset and adds a
    new row field in the location specified by `name`.

    Examples
    --------

    Add Nirvana annotations to the dataset:

    >>> result = hl.nirvana(dataset, "data/nirvana.properties") # doctest: +SKIP

    Notes
    -----

    ***Configuration***

    :func:`.nirvana` requires a configuration file. The format is a
    `.properties file <https://en.wikipedia.org/wiki/.properties>`__, where each
    line defines a property as a key-value pair of the form ``key = value``.
    :func:`.nirvana` supports the following properties:

    - **hail.nirvana.dotnet** -- Location of dotnet. Optional, default: dotnet.
    - **hail.nirvana.path** -- Value of the PATH environment variable when
      invoking Nirvana. Optional, by default PATH is not set.
    - **hail.nirvana.location** -- Location of Nirvana.dll. Required.
    - **hail.nirvana.reference** -- Location of reference genome. Required.
    - **hail.nirvana.cache** -- Location of cache. Required.
    - **hail.nirvana.supplementaryAnnotationDirectory** -- Location of
      Supplementary Database. Optional, no supplementary database by default.

    Here is an example ``nirvana.properties`` configuration file:

    .. code-block:: text

        hail.nirvana.location = /path/to/dotnet/netcoreapp1.1/Nirvana.dll
        hail.nirvana.reference = /path/to/nirvana/References/Homo_sapiens.GRCh37.Nirvana.dat
        hail.nirvana.cache = /path/to/nirvana/Cache/GRCh37/Ensembl84
        hail.nirvana.supplementaryAnnotationDirectory = /path/to/nirvana/SupplementaryDatabase/GRCh37

    **Annotations**

    A new row field is added in the location specified by `name` with the
    following schema:

    .. code-block:: text

        struct {
            chromosome: str,
            refAllele: str,
            position: int32,
            altAlleles: array<str>,
            cytogeneticBand: str,
            quality: float64,
            filters: array<str>,
            jointSomaticNormalQuality: int32,
            copyNumber: int32,
            strandBias: float64,
            recalibratedQuality: float64,
            variants: array<struct {
                altAllele: str,
                refAllele: str,
                chromosome: str,
                begin: int32,
                end: int32,
                phylopScore: float64,
                isReferenceMinor: bool,
                variantType: str,
                vid: str,
                isRecomposed: bool,
                regulatoryRegions: array<struct {
                    id: str,
                    consequence: set<str>,
                    type: str
                }>,
                clinvar: array<struct {
                    id: str,
                    reviewStatus: str,
                    isAlleleSpecific: bool,
                    alleleOrigins: array<str>,
                    refAllele: str,
                    altAllele: str,
                    phenotypes: array<str>,
                    medGenIds: array<str>,
                    omimIds: array<str>,
                    orphanetIds: array<str>,
                    geneReviewsId: str,
                    significance: str,
                    lastUpdatedDate: str,
                    pubMedIds: array<str>
                }>,
                cosmic: array<struct {
                    id: str,
                    isAlleleSpecific: bool,
                    refAllele: str,
                    altAllele: str,
                    gene: str,
                    sampleCount: int32,
                    studies: array<struct {
                        id: int32,
                        histology: str,
                        primarySite: str
                    }>
                }>,
                dbsnp: struct {
                    ids: array<str>
                },
                evs: struct {
                    coverage: int32,
                    sampleCount: int32,
                    allAf: float64,
                    afrAf: float64,
                    eurAf: float64
                },
                exac: struct {
                    coverage: int32,
                    allAf: float64,
                    allAc: int32,
                    allAn: int32,
                    afrAf: float64,
                    afrAc: int32,
                    afrAn: int32,
                    amrAf: float64,
                    amrAc: int32,
                    amrAn: int32,
                    easAf: float64,
                    easAc: int32,
                    easAn: int32,
                    finAf: float64,
                    finAc: int32,
                    finAn: int32,
                    nfeAf: float64,
                    nfeAc: int32,
                    nfeAn: int32,
                    othAf: float64,
                    othAc: int32,
                    othAn: int32,
                    sasAf: float64,
                    sasAc: int32,
                    sasAn: int32
                },
                globalAllele: struct {
                    globalMinorAllele: str,
                    globalMinorAlleleFrequency: float64
                },
                oneKg: struct {
                    ancestralAllele: str,
                    allAf: float64,
                    allAc: int32,
                    allAn: int32,
                    afrAf: float64,
                    afrAc: int32,
                    afrAn: int32,
                    amrAf: float64,
                    amrAc: int32,
                    amrAn: int32,
                    easAf: float64,
                    easAc: int32,
                    easAn: int32,
                    eurAf: float64,
                    eurAc: int32,
                    eurAn: int32,
                    sasAf: float64,
                    sasAc: int32,
                    sasAn: int32
                },
                transcripts: struct {
                    refSeq: array<struct {
                        transcript: str,
                        bioType: str,
                        aminoAcids: str,
                        cDnaPos: str,
                        codons: str,
                        cdsPos: str,
                        exons: str,
                        introns: str,
                        geneId: str,
                        hgnc: str,
                        consequence: array<str>,
                        hgvsc: str,
                        hgvsp: str,
                        isCanonical: bool,
                        polyPhenScore: float64,
                        polyPhenPrediction: str,
                        proteinId: str,
                        proteinPos: str,
                        siftScore: float64,
                        siftPrediction: str
                    }>,
                    ensembl: array<struct {
                        transcript: str,
                        bioType: str,
                        aminoAcids: str,
                        cDnaPos: str,
                        codons: str,
                        cdsPos: str,
                        exons: str,
                        introns: str,
                        geneId: str,
                        hgnc: str,
                        consequence: array<str>,
                        hgvsc: str,
                        hgvsp: str,
                        isCanonical: bool,
                        polyPhenScore: float64,
                        polyPhenPrediction: str,
                        proteinId: str,
                        proteinPos: str,
                        siftScore: float64,
                        siftPrediction: str
                    }>
                },
                genes: array<struct {
                    name: str,
                    omim: array<struct {
                        mimNumber: int32,
                        hgnc: str,
                        description: str,
                        phenotypes: array<struct {
                            mimNumber: int32,
                            phenotype: str,
                            mapping: str,
                            inheritance: array<str>,
                            comments: str
                        }>
                    }>
                }>
            }>
        }

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset.
    config : :obj:`str`
        Path to Nirvana configuration file.
    block_size : :obj:`int`
        Number of rows to process per Nirvana invocation.
    name : :obj:`str`
        Name for resulting row field.

    Returns
    -------
    :class:`.MatrixTable`
        Dataset with new row-indexed field `name` containing Nirvana annotations.
    """

    require_row_key_variant(dataset, 'nirvana')
    return MatrixTable(Env.hail().methods.Nirvana.apply(dataset._jvds, config, block_size, 'va.`{}`'.format(name)))
