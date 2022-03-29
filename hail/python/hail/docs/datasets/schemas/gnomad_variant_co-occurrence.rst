.. _gnomad_variant_co-occurrence:

gnomad_variant_co-occurrence
============================

*  **Versions:** 2.1.1
*  **Reference genome builds:** GRCh37
*  **Type:** :class:`hail.Table`

Schema (2.1.1, GRCh37)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        'max_freq': float64
        'least_consequence': str
        'same_haplotype_em_probability_cutoff': float64
        'different_haplotypes_em_probability_cutoff': float64
        'global_annotation_descriptions': struct {
            max_freq: str,
            least_consequence: str,
            same_haplotype_em_probability_cutoff: str,
            different_haplotypes_em_probability_cutoff: str
        }
        'row_annotation_descriptions': struct {
            locus1: str,
            alleles1: str,
            locus2: str,
            alleles2: str,
            phase_info: struct {
                description: str,
                gt_counts: str,
                em: struct {
                    hap_counts: str,
                    p_chet: str,
                    same_haplotype: str,
                    different_haplotype: str
                }
            }
        }
    ----------------------------------------
    Row fields:
        'locus1': locus<GRCh37>
        'alleles1': array<str>
        'locus2': locus<GRCh37>
        'alleles2': array<str>
        'phase_info': dict<str, struct {
            gt_counts: array<int32>,
            em: struct {
                hap_counts: array<float64>,
                p_chet: float64,
                same_haplotype: bool,
                different_haplotype: bool
            }
        }>
    ----------------------------------------
    Key: ['locus1', 'alleles1', 'locus2', 'alleles2']
    ----------------------------------------
