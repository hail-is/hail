
import hail as hl
from hail.expr.expressions import *
from hail.typecheck import *
from hail.expr.types import *
from hail.ir import *

annotations = [
    'Coding_UCSC',
    'Coding_UCSC.extend.500',
    'Conserved_LindbladToh',
    'Conserved_LindbladToh.extend.500',
    'CTCF_Hoffman',
    'CTCF_Hoffman.extend.500',
    'DGF_ENCODE',
    'DGF_ENCODE.extend.500',
    'DHS_peaks_Trynka',
    'DHS_Trynka',
    'DHS_Trynka.extend.500',
    'Enhancer_Andersson',
    'Enhancer_Andersson.extend.500',
    'Enhancer_Hoffman',
    'Enhancer_Hoffman.extend.500',
    'FetalDHS_Trynka',
    'FetalDHS_Trynka.extend.500',
    'H3K27ac_Hnisz',
    'H3K27ac_Hnisz.extend.500',
    'H3K27ac_PGC2',
    'H3K27ac_PGC2.extend.500',
    'H3K4me1_peaks_Trynka',
    'H3K4me1_Trynka',
    'H3K4me1_Trynka.extend.500',
    'H3K4me3_peaks_Trynka',
    'H3K4me3_Trynka',
    'H3K4me3_Trynka.extend.500',
    'H3K9ac_peaks_Trynka',
    'H3K9ac_Trynka',
    'H3K9ac_Trynka.extend.500',
    'Intron_UCSC',
    'Intron_UCSC.extend.500',
    'PromoterFlanking_Hoffman',
    'PromoterFlanking_Hoffman.extend.500',
    'Promoter_UCSC',
    'Promoter_UCSC.extend.500',
    'Repressed_Hoffman',
    'Repressed_Hoffman.extend.500',
    'SuperEnhancer_Hnisz',
    'SuperEnhancer_Hnisz.extend.500',
    'TFBS_ENCODE',
    'TFBS_ENCODE.extend.500',
    'Transcribed_Hoffman',
    'Transcribed_Hoffman.extend.500',
    'UTR_3_UCSC',
    'UTR_3_UCSC.extend.500',
    'UTR_5_UCSC',
    'UTR_5_UCSC.extend.500',
    'WeakEnhancer_Hoffman',
    'WeakEnhancer_Hoffman.extend.500']

def locus_interval_expr(contig, start, end, includes_start, includes_end,
                        reference_genome, skip_invalid_intervals):
    if reference_genome:
        if skip_invalid_intervals:
            is_valid_locus_interval = (
                (hl.is_valid_contig(contig, reference_genome) &
                 (hl.is_valid_locus(contig, start, reference_genome) |
                  (~hl.bool(includes_start) & (start == 0))) &
                 (hl.is_valid_locus(contig, end, reference_genome) |
                  (~hl.bool(includes_end) & hl.is_valid_locus(contig, end - 1, reference_genome))) &
                 (hl.bool(end > start) | 
                    hl.bool(includes_start) & hl.bool(includes_end) & (start == end))))

            return hl.or_missing(is_valid_locus_interval,
                                 hl.locus_interval(contig, start, end,
                                                   includes_start, includes_end,
                                                   reference_genome))
        else:
            return hl.locus_interval(contig, start, end, includes_start,
                                     includes_end, reference_genome)
    else:
        return hl.interval(hl.struct(contig=contig, position=start),
                           hl.struct(contig=contig, position=end),
                           includes_start,
                           includes_end)

rg = hl.get_reference('GRCh37')
lengths = rg.lengths
for i, a in enumerate(annotations):
    ht = hl.import_table(f'file:///Users/labbott/ldsc-best-practices/testing/baseline_v1.1_bedfiles/{a}.bed',
        no_header=True, impute=True, types={'f0': hl.tstr})
    if len(ht.row) > 3:
        print(a)
        ht.show(5)
        continue
    ht = ht.annotate(interval=hl.locus_interval(
        contig=ht.f0.replace('chr', ''),
        start=ht.f1 + 1,
        end=hl.if_else(end > length, length, end)),
        includes_start=True,
        includes_end=True,
        reference_genome='GRCh37'))
    ht = ht.key_by(ht.interval)
    ht = ht.annotate(annotation=a)
    ht = ht.select(ht.annotation)
    ht.write(f'file:///Users/labbott/ldsc-best-practices/testing/baselineLD_v1.1_annotations/{a}.ht', overwrite=True)
