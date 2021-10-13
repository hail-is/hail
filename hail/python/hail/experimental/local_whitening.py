import hail as hl
from hail.expr import (check_entry_indexed, matrix_table_source)
from hail.utils.java import Env


def whiten(entry_expr, block_size=16, partition_size=16):
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
        rows_per_partition = block_size * partition_size
        num_partitions = -(-total_num_rows // rows_per_partition)  # ceiling division
        idx_in_partition = ht._even_partitioning_index % rows_per_partition
        partition_idx = ht._even_partitioning_index // rows_per_partition
        filt = ht.filter(
            (idx_in_partition == 0)
            | ((idx_in_partition == rows_per_partition - block_size) & (partition_idx < num_partitions))
            | (ht._even_partitioning_index == (total_num_rows - 1)))
        interval_bounds = filt.select().collect()
        intervals = []
        for i in range(num_partitions - 1):
            intervals.append(hl.utils.Interval(start=interval_bounds[2*i], end=interval_bounds[2*(i+1)], includes_start=True, includes_end=False))
        # intervals_bounds normally length 2*num_partitions, but could be one
        # less if last partition has exactly one row. Using index -1 for end of
        # last interval handles both cases
        last_interval = hl.utils.Interval(start=interval_bounds[2*(num_partitions-1)], end=interval_bounds[-1], includes_start=True, includes_end=True)
        intervals.append(last_interval)

        trailing_blocks = [
            hl.utils.Interval(start=interval_bounds[2*i + 1], end=interval_bounds[2*(i+1)], includes_start=True, includes_end=False)
            for i in range(num_partitions - 1)]

        rekey_map = [(interval_bounds[2*i + 1], interval_bounds[2*(i+1)]) for i in range(num_partitions - 1)]

        return intervals, trailing_blocks, rekey_map

    ht = mt.localize_entries(entries_array_field_name="entries", columns_array_field_name="cols")
    ht = ht.select(xs=ht.entries.map(lambda e: e['x']))
    temp_file_name = hl.utils.new_temp_file("mt_to_table_of_ndarray", "ht")
    ht = ht.checkpoint(temp_file_name)

    num_rows = ht.count()
    new_partitioning, trailing_blocks, rekey_map = get_even_partitioning(ht, num_rows)
    new_part_ht = hl.read_table(temp_file_name, _intervals=new_partitioning)

    grouped = new_part_ht._group_within_partitions("groups", block_size)
    A = grouped.select(ndarray=hl.nd.array(grouped.groups.map(lambda group: group.xs)))

    trailing_blocks_ht = hl.read_table(temp_file_name, _intervals=trailing_blocks)
    trailing_blocks_ht = trailing_blocks_ht._group_within_partitions("groups", block_size)
    trailing_blocks_ht = trailing_blocks_ht.select(prev_window=hl.nd.array(trailing_blocks_ht.groups.map(lambda group: group.xs)))
    trailing_blocks_ht = trailing_blocks_ht.annotate_globals(rekey_map=hl.dict(rekey_map))
    trailing_blocks_ht = trailing_blocks_ht.key_by(**trailing_blocks_ht.rekey_map[trailing_blocks_ht.key])

    return A.join(trailing_blocks_ht, how='left')
