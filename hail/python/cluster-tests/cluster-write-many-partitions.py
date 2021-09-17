import hail as hl

ht = hl.utils.range_table(1_000_000, n_partitions=10_000)

# use HDFS so as not to create garbage on GS
ht.write('/tmp/many_partitions.ht')

mt = hl.utils.range_matrix_table(1_000_000, 2, n_partitions=10_000)
mt.write('/tmp/many_partitions.mt')
