from hail.typecheck import *
from hail.utils.java import Env, handle_py4j
from hail.matrixtable import MatrixTable
from hail.table import Table
from .misc import require_biallelic

@handle_py4j
@typecheck(dataset=MatrixTable, name=strlike)
def sample_qc(dataset, name='sample_qc'):
    """Compute per-sample metrics useful for quality control.

    .. include:: ../_templates/req_tvariant.rst

    Examples
    --------

    Compute sample QC metrics and remove low-quality samples:

    >>> dataset = methods.sample_qc(dataset, name='sample_qc')
    >>> filtered_dataset = dataset.filter_cols((dataset.sample_qc.dpMean > 20) & (dataset.sample_qc.rTiTv > 1.5))

    Notes
    -----

    This method computes summary statistics per sample from a genetic matrix and stores the results as
    a new column-indexed field in the matrix, named based on the ``name`` parameter.

    +------------------------+-------+-+----------------------------------------------------------+
    | Name                   | Type    | Description                                              |
    +========================+=========+==========================================================+
    | ``callRate``           | Float64 | Fraction of calls non-missing                            |
    +------------------------+---------+----------------------------------------------------------+
    | ``nHomRef``            | Int64   | Number of homozygous reference calls                     |
    +------------------------+---------+----------------------------------------------------------+
    | ``nHet``               | Int64   | Number of heterozygous calls                             |
    +------------------------+---------+----------------------------------------------------------+
    | ``nHomVar``            | Int64   | Number of homozygous alternate calls                     |
    +------------------------+---------+----------------------------------------------------------+
    | ``nCalled``            | Int64   | Sum of ``nHomRef`` + ``nHet`` + ``nHomVar``              |
    +------------------------+---------+----------------------------------------------------------+
    | ``nNotCalled``         | Int64   | Number of missing calls                                  |
    +------------------------+---------+----------------------------------------------------------+
    | ``nSNP``               | Int64   | Number of SNP alternate alleles                          |
    +------------------------+---------+----------------------------------------------------------+
    | ``nInsertion``         | Int64   | Number of insertion alternate alleles                    |
    +------------------------+---------+----------------------------------------------------------+
    | ``nDeletion``          | Int64   | Number of deletion alternate alleles                     |
    +------------------------+---------+----------------------------------------------------------+
    | ``nSingleton``         | Int64   | Number of private alleles                                |
    +------------------------+---------+----------------------------------------------------------+
    | ``nTransition``        | Int64   | Number of transition (A-G, C-T) alternate alleles        |
    +------------------------+---------+----------------------------------------------------------+
    | ``nTransversion``      | Int64   | Number of transversion alternate alleles                 |
    +------------------------+---------+----------------------------------------------------------+
    | ``nStar``              | Int64   | Number of star (upstream deletion) alleles               |
    +------------------------+---------+----------------------------------------------------------+
    | ``nNonRef``            | Int64   | Sum of ``nHet`` and ``nHomVar``                          |
    +------------------------+---------+----------------------------------------------------------+
    | ``rTiTv``              | Float64 | Transition/Transversion ratio                            |
    +------------------------+---------+----------------------------------------------------------+
    | ``rHetHomVar``         | Float64 | Het/HomVar call ratio                                    |
    +------------------------+---------+----------------------------------------------------------+
    | ``rInsertionDeletion`` | Float64 | Insertion/Deletion allele ratio                          |
    +------------------------+---------+----------------------------------------------------------+
    | ``dpMean``             | Float64 | Depth mean across all calls                              |
    +------------------------+---------+----------------------------------------------------------+
    | ``dpStDev``            | Float64 | Depth standard deviation across all calls                |
    +------------------------+---------+----------------------------------------------------------+
    | ``gqMean``             | Float64 | The average genotype quality across all calls            |
    +------------------------+---------+----------------------------------------------------------+
    | ``gqStDev``            | Float64 | Genotype quality standard deviation across all calls     |
    +------------------------+---------+----------------------------------------------------------+

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

    return MatrixTable(Env.hail().methods.SampleQC.apply(require_biallelic(dataset, 'sample_qc')._jvds, 'sa.`{}`'.format(name)))

@handle_py4j
@typecheck(dataset=MatrixTable, name=strlike)
def variant_qc(dataset, name='variant_qc'):
    """Compute common variant statistics (quality control metrics).

    .. include:: ../_templates/req_biallelic.rst
    .. include:: ../_templates/req_tvariant.rst

    Examples
    --------

    >>> dataset_result = methods.variant_qc(dataset)

    Notes
    -----
    This method computes 18 variant statistics from the genotype data,
    returning a new struct field `name` with the following metrics:

    +---------------------------+---------+--------------------------------------------------------+
    | Name                      | Type    | Description                                            |
    +===========================+=========+========================================================+
    | ``callRate``              | Float64 | Fraction of samples with called genotypes              |
    +---------------------------+---------+--------------------------------------------------------+
    | ``AF``                    | Float64 | Calculated alternate allele frequency (q)              |
    +---------------------------+---------+--------------------------------------------------------+
    | ``AC``                    | Int32   | Count of alternate alleles                             |
    +---------------------------+---------+--------------------------------------------------------+
    | ``rHeterozygosity``       | Float64 | Proportion of heterozygotes                            |
    +---------------------------+---------+--------------------------------------------------------+
    | ``rHetHomVar``            | Float64 | Ratio of heterozygotes to homozygous alternates        |
    +---------------------------+---------+--------------------------------------------------------+
    | ``rExpectedHetFrequency`` | Float64 | Expected rHeterozygosity based on HWE                  |
    +---------------------------+---------+--------------------------------------------------------+
    | ``pHWE``                  | Float64 | p-value from Hardy Weinberg Equilibrium null model     |
    +---------------------------+---------+--------------------------------------------------------+
    | ``nHomRef``               | Int32   | Number of homozygous reference samples                 |
    +---------------------------+---------+--------------------------------------------------------+
    | ``nHet``                  | Int32   | Number of heterozygous samples                         |
    +---------------------------+---------+--------------------------------------------------------+
    | ``nHomVar``               | Int32   | Number of homozygous alternate samples                 |
    +---------------------------+---------+--------------------------------------------------------+
    | ``nCalled``               | Int32   | Sum of ``nHomRef``, ``nHet``, and ``nHomVar``          |
    +---------------------------+---------+--------------------------------------------------------+
    | ``nNotCalled``            | Int32   | Number of uncalled samples                             |
    +---------------------------+---------+--------------------------------------------------------+
    | ``nNonRef``               | Int32   | Sum of ``nHet`` and ``nHomVar``                        |
    +---------------------------+---------+--------------------------------------------------------+
    | ``rHetHomVar``            | Float64 | Het/HomVar ratio across all samples                    |
    +---------------------------+---------+--------------------------------------------------------+
    | ``dpMean``                | Float64 | Depth mean across all samples                          |
    +---------------------------+---------+--------------------------------------------------------+
    | ``dpStDev``               | Float64 | Depth standard deviation across all samples            |
    +---------------------------+---------+--------------------------------------------------------+
    | ``gqMean``                | Float64 | The average genotype quality across all samples        |
    +---------------------------+---------+--------------------------------------------------------+
    | ``gqStDev``               | Float64 | Genotype quality standard deviation across all samples |
    +---------------------------+---------+--------------------------------------------------------+

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

    return MatrixTable(Env.hail().methods.VariantQC.apply(require_biallelic(dataset, 'variant_qc')._jvds, 'va.`{}`'.format(name)))


@handle_py4j
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

    >>> global_conc, cols_conc, rows_conc = methods.concordance(dataset, dataset2)

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

    >>> summary, samples, variants = methods.concordance(dataset, dataset2)
    >>> left_homref_right_homvar = summary[2][4]
    >>> left_het_right_missing = summary[3][1]
    >>> left_het_right_something_else = sum(summary[3][:]) - summary[3][3]
    >>> total_concordant = summary[2][2] + summary[3][3] + summary[4][4]
    >>> total_discordant = sum([sum(s[2:]) for s in summary[2:]]) - total_concordant

    **Using the table results**

    Table 1: Concordance statistics per column key

        - **s** -- column key.
        - **nDiscordant** (*Long*) -- Count of discordant calls (see below for
          full definition).
        - **concordance** (*Array[Array[Long]]*) -- Array of concordance per
          state on left and right, matching the structure of the global summary
          defined above.

    Table 2: Concordance statistics per row key

        - **v** -- row key.
        - **nDiscordant** (*Long*) -- Count of discordant calls (see below for
          full definition).
        - **concordance** (*Array[Array[Long]]*) -- Array of concordance per
          state on left and right, matching the structure of the global summary
          defined above.

    In these tables, the column **nDiscordant** is provided as a convenience,
    because this is often one of the most useful concordance statistics. This
    value is the number of genotypes which were called (homozygous reference,
    heterozygous, or homozygous variant) in both datasets, but where the call
    did not match between the two.

    The column **concordance** matches the structure of the global summmary,
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

    left = require_biallelic(left, "concordance, left")
    right = require_biallelic(right, "concordance, right")

    r = Env.hail().methods.CalculateConcordance.apply(left._jvds, right._jvds)
    j_global_conc = r._1()
    col_conc = Table(r._2())
    row_conc = Table(r._3())
    global_conc = [[j_global_conc.apply(j).apply(i) for i in xrange(5)] for j in xrange(5)]

    return global_conc, col_conc, row_conc


@handle_py4j
@typecheck(dataset=MatrixTable,
           config=strlike,
           block_size=integral,
           name=strlike,
           csq=bool)
def vep(dataset, config, block_size=1000, name='vep', csq=False):
    """Annotate variants with VEP.

    .. include:: ../_templates/req_tvariant.rst

    :meth:`.vep` runs `Variant Effect Predictor
    <http://www.ensembl.org/info/docs/tools/vep/index.html>`__ with the `LOFTEE
    plugin <https://github.com/konradjk/loftee>`__ on the current dataset and
    adds the result as a row field.

    Examples
    --------

    Add VEP annotations to the dataset:

    >>> result = methods.vep(dataset, "data/vep.properties") # doctest: +SKIP

    Notes
    -----

    **Configuration**

    :meth:`.vep` needs a configuration file to tell it
    how to run VEP. The format is a `.properties file
    <https://en.wikipedia.org/wiki/.properties>`__. Roughly, each line defines a
    property as a key-value pair of the form `key = value`. :meth:`.vep` supports the
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

        Struct{
          assembly_name: String,
          allele_string: String,
          colocated_variants: Array[Struct{
            aa_allele: String,
            aa_maf: Double,
            afr_allele: String,
            afr_maf: Double,
            allele_string: String,
            amr_allele: String,
            amr_maf: Double,
            clin_sig: Array[String],
            end: Int,
            eas_allele: String,
            eas_maf: Double,
            ea_allele: String,,
            ea_maf: Double,
            eur_allele: String,
            eur_maf: Double,
            exac_adj_allele: String,
            exac_adj_maf: Double,
            exac_allele: String,
            exac_afr_allele: String,
            exac_afr_maf: Double,
            exac_amr_allele: String,
            exac_amr_maf: Double,
            exac_eas_allele: String,
            exac_eas_maf: Double,
            exac_fin_allele: String,
            exac_fin_maf: Double,
            exac_maf: Double,
            exac_nfe_allele: String,
            exac_nfe_maf: Double,
            exac_oth_allele: String,
            exac_oth_maf: Double,
            exac_sas_allele: String,
            exac_sas_maf: Double,
            id: String,
            minor_allele: String,
            minor_allele_freq: Double,
            phenotype_or_disease: Int,
            pubmed: Array[Int],
            sas_allele: String,
            sas_maf: Double,
            somatic: Int,
            start: Int,
            strand: Int
          }],
          end: Int,
          id: String,
          input: String,
          intergenic_consequences: Array[Struct{
            allele_num: Int,
            consequence_terms: Array[String],
            impact: String,
            minimised: Int,
            variant_allele: String
          }],
          most_severe_consequence: String,
          motif_feature_consequences: Array[Struct{
            allele_num: Int,
            consequence_terms: Array[String],
            high_inf_pos: String,
            impact: String,
            minimised: Int,
            motif_feature_id: String,
            motif_name: String,
            motif_pos: Int,
            motif_score_change: Double,
            strand: Int,
            variant_allele: String
          }],
          regulatory_feature_consequences: Array[Struct{
            allele_num: Int,
            biotype: String,
            consequence_terms: Array[String],
            impact: String,
            minimised: Int,
            regulatory_feature_id: String,
            variant_allele: String
          }],
          seq_region_name: String,
          start: Int,
          strand: Int,
          transcript_consequences: Array[Struct{
            allele_num: Int,
            amino_acids: String,
            biotype: String,
            canonical: Int,
            ccds: String,
            cdna_start: Int,
            cdna_end: Int,
            cds_end: Int,
            cds_start: Int,
            codons: String,
            consequence_terms: Array[String],
            distance: Int,
            domains: Array[Struct{
              db: String
              name: String
            }],
            exon: String,
            gene_id: String,
            gene_pheno: Int,
            gene_symbol: String,
            gene_symbol_source: String,
            hgnc_id: String,
            hgvsc: String,
            hgvsp: String,
            hgvs_offset: Int,
            impact: String,
            intron: String,
            lof: String,
            lof_flags: String,
            lof_filter: String,
            lof_info: String,
            minimised: Int,
            polyphen_prediction: String,
            polyphen_score: Double,
            protein_end: Int,
            protein_start: Int,
            protein_id: String,
            sift_prediction: String,
            sift_score: Double,
            strand: Int,
            swissprot: String,
            transcript_id: String,
            trembl: String,
            uniparc: String,
            variant_allele: String
          }],
          variant_class: String
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
        If ``True``, annotates VCF CSQ field as a String.
        If ``False``, annotates with the full nested struct schema.

    Returns
    -------
    :class:`.MatrixTable`
        Dataset with new row-indexed field `name` containing VEP annotations.
    """

    return MatrixTable(Env.hail().methods.VEP.apply(dataset._jvds, config, 'va.`{}`'.format(name), csq, block_size))


@handle_py4j
@typecheck_method(dataset=MatrixTable,
                  config=strlike,
                  block_size=integral,
                  name=strlike)
def nirvana(dataset, config, block_size=500000, name='nirvana'):
    """Annotate variants using `Nirvana <https://github.com/Illumina/Nirvana>`_.

    .. include:: ../_templates/experimental.rst

    .. include:: ../_templates/req_tvariant.rst

    :meth:`.nirvana` runs `Nirvana
    <https://github.com/Illumina/Nirvana>`_ on the current dataset and adds a
    new row field in the location specified by `name`.

    Examples
    --------

    Add Nirvana annotations to the dataset:

    >>> result = methods.nirvana(dataset, "data/nirvana.properties") # doctest: +SKIP

    Notes
    -----

    ***Configuration***

    :meth:`.nirvana` requires a configuration file. The format is a
    `.properties file <https://en.wikipedia.org/wiki/.properties>`__, where each
    line defines a property as a key-value pair of the form ``key = value``.
    :meth:`.nirvana` supports the following properties:

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

        Struct{
          chromosome: String,
          refAllele: String,
          position: Int,
          altAlleles: Array[String],
          cytogeneticBand: String,
          quality: Double,
          filters: Array[String],
          jointSomaticNormalQuality: Int,
          copyNumber: Int,
          strandBias: Double,
          recalibratedQuality: Double,
          variants: Array[Struct{
            altAllele: String,
            refAllele: String,
            chromosome: String,
            begin: Int,
            end: Int,
            phylopScore: Double,
            isReferenceMinor: Boolean,
            variantType: String,
            vid: String,
            isRecomposed: Boolean,
            regulatoryRegions: Array[Struct{
              id: String,
              consequence: Set[String],
              type: String
            }],
            clinvar: Array[Struct{
              id: String,
              reviewStatus: String,
              isAlleleSpecific: Boolean,
              alleleOrigins: Array[String],
              refAllele: String,
              altAllele: String,
              phenotypes: Array[String],
              medGenIds: Array[String],
              omimIds: Array[String],
              orphanetIds: Array[String],
              geneReviewsId: String,
              significance: String,
              lastUpdatedDate: String,
              pubMedIds: Array[String]
            }],
            cosmic: Array[Struct{
              id: String,
              isAlleleSpecific: Boolean,
              refAllele: String,
              altAllele: String,
              gene: String,
              sampleCount: Int,
              studies: Array[Struct{
                id: Int,
                histology: String,
                primarySite: String
              }]
            }],
            dbsnp: Struct{"ids: Array(String)},
            evs: Struct{
              coverage: Int,
              sampleCount: Int,
              allAf: Double,
              afrAf: Double,
              eurAf: Double
            },
            exac: Struct{
              coverage: Int,
              allAf: Double,
              allAc: Int,
              allAn: Int,
              afrAf: Double,
              afrAc: Int,
              afrAn: Int,
              amrAf: Double,
              amrAc: Int,
              amrAn: Int,
              easAf: Double,
              easAc: Int,
              easAn: Int,
              finAf: Double,
              finAc: Int,
              finAn: Int,
              nfeAf: Double,
              nfeAc: Int,
              nfeAn: Int,
              othAf: Double,
              othAc: Int,
              othAn: Int,
              sasAf: Double,
              sasAc: Int,
              sasAn: Int
            },
            globalAllele: Struct{
              globalMinorAllele: String,
              globalMinorAlleleFrequency: Double
            },
            oneKg: Struct{
              ancestralAllele: String,
              allAf: Double,
              allAc: Int,
              allAn: Int,
              afrAf: Double,
              afrAc: Int,
              afrAn: Int,
              amrAf: Double,
              amrAc: Int,
              amrAn: Int,
              easAf: Double,
              easAc: Int,
              easAn: Int,
              eurAf: Double,
              eurAc: Int,
              eurAn: Int,
              sasAf: Double,
              sasAc: Int,
              sasAn: Int
            },
            transcripts: Struct{
              refSeq: Array[Struct{
                transcript: String,
                bioType: String,
                aminoAcids: String,
                cDnaPos: String,
                codons: String,
                cdsPos: String,
                exons: String,
                introns: String,
                geneId: String,
                hgnc: String,
                consequence: Array[String],
                hgvsc: String,
                hgvsp: String,
                isCanonical: Boolean,
                polyPhenScore: Double,
                polyPhenPrediction: String,
                proteinId: String,
                proteinPos: String,
                siftScore: Double,
                siftPrediction: String
              }],
              ensembl: Array[Struct{
                transcript: String,
                bioType: String,
                aminoAcids: String,
                cDnaPos: String,
                codons: String,
                cdsPos: String,
                exons: String,
                introns: String,
                geneId: String,
                hgnc: String,
                consequence: Array[String],
                hgvsc: String,
                hgvsp: String,
                isCanonical: Boolean,
                polyPhenScore: Double,
                polyPhenPrediction: String,
                proteinId: String,
                proteinPos: String,
                siftScore: Double,
                siftPrediction: String
              }]
            },
            genes: Array[Struct{
              name: String,
              omim: Array[Struct{
                mimNumber: Int,
                hgnc: String,
                description: String,
                phenotypes: Array[Struct{
                  mimNumber: Int,
                  phenotype: String,
                  mapping: String,
                  inheritance: Array[String],
                  comments: String
                }]
              }]
            }]
          }]
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

    return MatrixTable(Env.hail().methods.Nirvana.apply(dataset._jvds, config, block_size, 'va.`{}`'.format(name)))
