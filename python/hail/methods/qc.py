from hail.typecheck import *
from hail.utils.java import Env, handle_py4j
from hail.api2 import MatrixTable
from .misc import require_biallelic


@handle_py4j
@require_biallelic
@typecheck(dataset=MatrixTable, name=strlike)
def sample_qc(dataset, name='sample_qc'):
    """Compute per-sample metrics useful for quality control.

    .. include:: ../_templates/req_tvariant.rst

    **Examples**

    .. testsetup::

        dataset = vds.annotate_samples_expr('sa = drop(sa, qc)').to_hail2()
        from hail.methods import sample_qc

    Compute sample QC metrics and remove low-quality samples:

    >>> dataset = sample_qc(dataset, name='sample_qc')
    >>> filtered_dataset = dataset.filter_cols((dataset.sample_qc.dpMean > 20) & (dataset.sample_qc.rTiTv > 1.5))

    **Notes**:

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

    return MatrixTable(Env.hail().methods.SampleQC.apply(dataset._jvds, 'sa.`{}`'.format(name)))

@handle_py4j
@require_biallelic
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

    return MatrixTable(Env.hail().methods.VariantQC.apply(dataset._jvds, 'va.`{}`'.format(name)))


@handle_py4j
@typecheck(dataset=MatrixTable,
           config=strlike,
           block_size=integral,
           name=strlike,
           csq=bool)
def vep(dataset, config, block_size=1000, name='vep', csq=False):
    """Annotate variants with VEP.

    .. include:: ../_templates/req_tvariant.rst

    :py:meth:`~hail.methods.vep` runs `Variant Effect Predictor
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

    :py:meth:`~hail.VariantDataset.vep` needs a configuration file to tell it
    how to run VEP. The format is a `.properties file
    <https://en.wikipedia.org/wiki/.properties>`__. Roughly, each line defines a
    property as a key-value pair of the form `key = value`. `vep` supports the
    following properties:

    - **hail.vep.perl** -- Location of Perl. Optional, default: perl.
    - **hail.vep.perl5lib** -- Value for the PERL5LIB environment variable when
      invoking VEP. Optional, by default PERL5LIB is not set.
    - **hail.vep.path** -- Value of the PATH environment variable when invoking
      VEP.  Optional, by default PATH is not set.
    - **hail.vep.location** -- Location of the VEP Perl script.  Required.
    - **hail.vep.cache_dir** -- Location of the VEP cache dir, passed to VEP
      with the `--dir` option. Required.
    - **hail.vep.fasta** -- Location of the FASTA file to use to look up the
      reference sequence, passed to VEP with the `--fasta` option. Required.
    - **hail.vep.plugin** -- VEP plugin, passed to VEP with the `--plugin`
      option. Optional. Overrides `hail.vep.lof.human_ancestor` and
      `hail.vep.lof.conservation_file`.
    - **hail.vep.lof.conservation_file** -- Location of the conservation file
      for the LOFTEE plugin. Ignored if `hail.vep.plugin` is set. Required
      otherwise.

    Here is an example `vep.properties` configuration file

    .. code-block:: text

        hail.vep.perl = /usr/bin/perl
        hail.vep.path = /usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
        hail.vep.location = /path/to/vep/ensembl-tools-release-81/scripts/variant_effect_predictor/variant_effect_predictor.pl
        hail.vep.cache_dir = /path/to/vep
        hail.vep.lof.human_ancestor = /path/to/loftee_data/human_ancestor.fa.gz
        hail.vep.lof.conservation_file = /path/to/loftee_data//phylocsf.sql

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
        --plugin LoF,human_ancestor_fa:$<hail.vep.lof.human_ancestor>,filter_position:0.05,min_intron_size:15,conservation_file:<hail.vep.lof.conservation_file>
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
    block_size: :obj:`int`
        Number of variants to process per VEP invocation.
    name : :obj:`str`
        Name for resulting field.
    csq : :obj:`bool`
        If ``True``, annotates VCF CSQ field as a String.
        If ``False``, annotates with the full nested struct schema

    Returns
    -------
    :class:`.MatrixTable`
        Dataset with new row-indexed field `name` containing VEP annotations.
    """

    return MatrixTable(Env.hail().methods.VEP.apply(dataset._jvds, config, 'va.`{}`'.format(name), csq, block_size))
