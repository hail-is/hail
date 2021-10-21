import hail as hl
from hail.expr import (check_entry_indexed, matrix_table_source)
from hail.expr.expressions import construct_expr
from hail.utils.java import Env
import hail.ir as ir


def whiten(entry_expr, chunk_size, window_size, partition_size, block_size=64):
    if window_size % chunk_size != 0:
        raise ValueError('whiten window_size must be a multiple of the chunk_size')
    if partition_size % chunk_size != 0:
        raise ValueError('whiten partition_size must be a multiple of the chunk_size')
    if partition_size <= chunk_size:
        raise ValueError('whiten requires partition_size be at least 2*chunk_size')
    if partition_size < window_size:
        raise ValueError('whiten requires partition_size be at least window_size')
    check_entry_indexed('mt_to_table_of_ndarray/entry_expr', entry_expr)
    mt = matrix_table_source('mt_to_table_of_ndarray/entry_expr', entry_expr)

    if entry_expr in mt._fields_inverse:
        field = mt._fields_inverse[entry_expr]
    else:
        field = Env.get_uid()
        mt = mt.select_entries(**{field: entry_expr})
    mt = mt.select_cols().select_rows().select_globals()

    mt = mt.select_entries(x=mt[field])

    def get_even_partitioning(ht, total_num_rows):
        ht = ht.select().add_index("_even_partitioning_index")
        num_partitions = -(-total_num_rows // partition_size)  # ceiling division
        idx_in_partition = ht._even_partitioning_index % partition_size
        partition_idx = ht._even_partitioning_index // partition_size
        agg_result = ht.aggregate(hl.struct(
            interval_bounds=hl.agg.filter((idx_in_partition == 0) | (ht._even_partitioning_index == (total_num_rows - 1)), hl.agg.collect(ht.key)),
            trailing_blocks=hl.agg.filter((idx_in_partition == partition_size - window_size) & (partition_idx < num_partitions), hl.agg.collect(ht.key))))
        intervals = [
            hl.utils.Interval(start=agg_result.interval_bounds[i], end=agg_result.interval_bounds[i + 1], includes_start=True, includes_end=False)
            for i in range(num_partitions - 1)
        ]
        # intervals_bounds normally length num_partitions+1, but could be one
        # less if last partition has exactly one row. Using index -1 for end of
        # last interval handles both cases.
        last_interval = hl.utils.Interval(start=agg_result.interval_bounds[num_partitions - 1], end=agg_result.interval_bounds[-1], includes_start=True, includes_end=True)
        intervals.append(last_interval)

        trailing_blocks = [
            hl.utils.Interval(start=agg_result.trailing_blocks[i], end=agg_result.interval_bounds[i + 1], includes_start=True, includes_end=False)
            for i in range(num_partitions - 1)
        ]

        rekey_map = [(agg_result.trailing_blocks[i], agg_result.interval_bounds[i + 1]) for i in range(num_partitions - 1)]

        return intervals, trailing_blocks, rekey_map

    ht = mt.localize_entries(entries_array_field_name="entries", columns_array_field_name="cols")
    ht = ht.select(xs=ht.entries.map(lambda e: e['x']))
    temp_file_name = hl.utils.new_temp_file("mt_to_table_of_ndarray", "ht")
    ht = ht.checkpoint(temp_file_name)

    num_rows = ht.count()
    new_partitioning, trailing_blocks, rekey_map = get_even_partitioning(ht, num_rows)
    new_part_ht = hl.read_table(temp_file_name, _intervals=new_partitioning).select('xs')

    grouped = new_part_ht._group_within_partitions("groups", chunk_size)
    A = grouped.select(ndarray=hl.nd.array(grouped.groups.map(lambda group: group.xs)).T)

    trailing_blocks_ht = hl.read_table(temp_file_name, _intervals=trailing_blocks)
    trailing_blocks_ht = trailing_blocks_ht._group_within_partitions("groups", window_size)
    trailing_blocks_ht = trailing_blocks_ht.select(prev_window=hl.nd.array(trailing_blocks_ht.groups.map(lambda group: group.xs)).T)
    trailing_blocks_ht = trailing_blocks_ht.annotate_globals(rekey_map=hl.dict(rekey_map))
    trailing_blocks_ht = trailing_blocks_ht.key_by(**trailing_blocks_ht.rekey_map[trailing_blocks_ht.key])

    vec_size = hl.eval(ht.take(1, _localize=False)[0].xs.length())

    joined = A.annotate(prev_window=trailing_blocks_ht[A.key].prev_window)

    def map_body(part_stream):
        stream_ir = ir.ToArray(ir.StreamWhiten(ir.ToStream(part_stream._ir), "ndarray", "prev_window", vec_size, window_size, chunk_size, block_size))
        return construct_expr(stream_ir, part_stream.dtype)
    joined = joined._map_partitions(map_body)

    return joined
