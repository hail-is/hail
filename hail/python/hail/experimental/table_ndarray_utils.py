import hail as hl
from hail.expr import (check_entry_indexed, matrix_table_source)
from hail.utils.java import Env


def mt_to_table_of_ndarray(entry_expr, block_size=16, return_checkpointed_table_also=False):
    check_entry_indexed('mt_to_table_of_ndarray/entry_expr', entry_expr)
    mt = matrix_table_source('mt_to_table_of_ndarray/entry_expr', entry_expr)

    if entry_expr in mt._fields_inverse:
        field = mt._fields_inverse[entry_expr]
    else:
        field = Env.get_uid()
        mt = mt.select_entries(**{field: entry_expr})
    mt = mt.select_cols().select_rows().select_globals()

    mt = mt.select_entries(x=mt[field])

    def get_even_partitioning(ht, partition_size, total_num_rows):
        ht = ht.select().add_index("_even_partitioning_index")
        filt = ht.filter((ht._even_partitioning_index % partition_size == 0) | (ht._even_partitioning_index == (total_num_rows - 1)))
        interval_bounds = filt.select().collect()
        intervals = []
        num_intervals = len(interval_bounds)
        for i in range(num_intervals - 2):
            intervals.append(hl.utils.Interval(start=interval_bounds[i], end=interval_bounds[i + 1], includes_start=True, includes_end=False))
        last_interval = hl.utils.Interval(start=interval_bounds[num_intervals - 2], end=interval_bounds[num_intervals - 1], includes_start=True, includes_end=True)
        intervals.append(last_interval)

        return intervals

    ht = mt.localize_entries(entries_array_field_name="entries", columns_array_field_name="cols")
    ht = ht.select(xs=ht.entries.map(lambda e: e['x']))
    temp_file_name = hl.utils.new_temp_file("mt_to_table_of_ndarray", "ht")
    ht = ht.checkpoint(temp_file_name)
    num_rows = ht.count()
    new_partitioning = get_even_partitioning(ht, block_size, num_rows)
    new_part_ht = hl.read_table(temp_file_name, _intervals=new_partitioning)

    grouped = new_part_ht._group_within_partitions("groups", block_size)
    A = grouped.select(ndarray=hl.nd.array(grouped.groups.map(lambda group: group.xs)))

    if return_checkpointed_table_also:
        return A, ht
    return A
