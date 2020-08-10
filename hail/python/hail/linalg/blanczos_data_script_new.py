import numpy as np
import hail as hl
from hail import methods
from hail.expr import matrix_table_source
import pandas as pd
from math import sqrt, pi, ceil
from random import randint, choice
import pandas as pd
import time


hl.init(log = 'hail_log.txt')

df = pd.DataFrame(columns=['M', 'N', 'block size', 'K', 'L', 'Q', 'blanczos time', 'hail pca time'])

def loop(i, data_field, m, n, block_size, k, l, q):

    print("loop attempt ", i, ' with m:', m, ', n:', n, ', m*n =', m*n, ', and q:', q)

    try:
        assert l > k
        assert n <= m
    except Exception as e:
        print(e)
        return

    try:

        start = time.time()
        blanczos_u, blanczos_s, blanczos_v = hl._blanczos_pca(data_field, k=10, q_iterations=q, oversampling_param=(l-k), block_size=block_size)
        end = time.time()
        blanczos_time = end - start

        start = time.time()
        eigens, scores, loadings = hl.pca(data_field, k=k)
        end = time.time()
        hail_time = end - start

        df.loc[i] = [m, n, block_size, k, l, q, blanczos_time, hail_time]

    except Exception as e:

        print(e)
        print('failed during blanczos algorithm with ', (m, n))
        return

    return


hl.utils.get_1kg('data/')
hl.import_vcf('data/1kg.vcf.bgz').write('data/1kg.mt', overwrite=True)
small_data = hl.read_matrix_table('data/1kg.mt')

def cleanMissingData(mt):

    call_expr = mt.GT
    mt = call_expr._indices.source
    mt = mt.select_entries(__gt=call_expr.n_alt_alleles())
    mt = mt.annotate_rows(__AC=hl.agg.sum(mt.__gt),
                          __n_called=hl.agg.count_where(hl.is_defined(mt.__gt)))
    mt = mt.filter_rows((mt.__AC > 0) & (mt.__AC < 2 * mt.__n_called))

    n_variants = mt.count_rows()
    if n_variants == 0:
        raise FatalError("hwe_normalized: found 0 variants after filtering out monomorphic sites.")

    mt = mt.annotate_rows(__mean_gt=mt.__AC / mt.__n_called)
    mt = mt.annotate_rows(__hwe_scaled_std_dev=hl.sqrt(mt.__mean_gt * (2 - mt.__mean_gt) * n_variants / 2))
    mt = mt.unfilter_entries()
    mt = matrix_table_source('hwe_normalized_pca/call_expr', hl.or_else((mt.__gt - mt.__mean_gt) / mt.__hwe_scaled_std_dev, 0.0))
    
    return mt

small_data = cleanMissingData(small_data)

m, n = small_data.count()

K = 10
i = 0

for L in [K + 2, 2 * K]:

    for Q in [1, 2, 3]:

        for block_size in [2, 4, 8, 16, 32]:

            # medium data too?

            loop(i, small_data.__gt, m, n, block_size, K, L, Q)
            df.to_csv('gs://aotoole/blanczos_newsmalldata_times.csv')
            i += 1
