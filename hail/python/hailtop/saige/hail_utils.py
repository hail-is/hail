from typing import Tuple, Union

import hail as hl


def gt_to_gp(mt: hl.MatrixTable, location: str = 'GP') -> hl.MatrixTable:
    return mt.annotate_entries(**{location:hl.or_missing(
        hl.is_defined(mt.GT),
        hl.map(lambda i: hl.cond(mt.GT.unphased_diploid_gt_index() == i, 1.0, 0.0),
               hl.range(0, hl.triangle(hl.len(mt.alleles)))))})


def impute_missing_gp(mt: hl.MatrixTable,
                      location: str = 'GP',
                      mean_impute: bool = True) -> hl.MatrixTable:
    mt = mt.annotate_entries(_gp=mt[location])
    if mean_impute:
        mt = mt.annotate_rows(_mean_gp=hl.agg.array_agg(lambda x: hl.agg.mean(x), mt._gp))
        gp_expr = mt._mean_gp
    else:
        gp_expr = [1.0, 0.0, 0.0]
    return mt.annotate_entries(**{location: hl.or_else(mt._gp, gp_expr)}).drop('_gp')


def get_interval_str(chrom_length: int, chromosome: Union[str, int], chunk_size: int, start_pos: int) -> str:
    if start_pos + chunk_size > chrom_length:
        end_pos = chrom_length
    else:
        end_pos = start_pos + chunk_size
    return f'{chromosome}:{start_pos}-{end_pos}'


def get_interval_strs(chrom_length: int, chromosome: Union[str, int], chunk_size: int, start_pos: int) -> Tuple[str, str]:
    padded_chunk_size = chunk_size * 3
    short_interval = get_interval_str(chrom_length, chromosome, chunk_size, start_pos)
    long_interval = get_interval_str(chrom_length, chromosome, padded_chunk_size, start_pos)
    return short_interval, long_interval
