import hail as hl
from hail.expr import matrix_table_source, raise_unless_entry_indexed
from hail.utils.java import Env


def mt_to_table_of_ndarray(
    entry_expr, block_size=16, *, partition_size=None, window_size=None, return_checkpointed_table_also=False
):
    raise_unless_entry_indexed('mt_to_table_of_ndarray/entry_expr', entry_expr)
    mt = matrix_table_source('mt_to_table_of_ndarray/entry_expr', entry_expr)

    if partition_size is None:
        partition_size = block_size

    if entry_expr in mt._fields_inverse:
        field = mt._fields_inverse[entry_expr]
    else:
        field = Env.get_uid()
        mt = mt.select_entries(**{field: entry_expr})
    mt = mt.select_cols().select_rows().select_globals()

    mt = mt.select_entries(x=mt[field])

    ht = mt.localize_entries(entries_array_field_name="entries", columns_array_field_name="cols")
    ht = ht.select(xs=ht.entries.map(lambda e: e['x']))
    temp_file_name = hl.utils.new_temp_file("mt_to_table_of_ndarray", "ht")
    ht_checkpoint = ht.checkpoint(temp_file_name)
    total_num_rows = ht.count()

    ht = ht_checkpoint.select().add_index("_even_partitioning_index")
    num_partitions = -(-total_num_rows // partition_size)  # ceiling division
    idx_in_partition = ht._even_partitioning_index % partition_size
    partition_idx = ht._even_partitioning_index // partition_size

    if window_size is None:
        agg_result = ht.aggregate(
            hl.struct(
                interval_bounds=hl.agg.filter(
                    (idx_in_partition == 0) | (ht._even_partitioning_index == (total_num_rows - 1)),
                    hl.agg.collect(ht.key),
                )
            )
        )
    else:
        agg_result = ht.aggregate(
            hl.struct(
                interval_bounds=hl.agg.filter(
                    (idx_in_partition == 0) | (ht._even_partitioning_index == (total_num_rows - 1)),
                    hl.agg.collect(ht.key),
                ),
                trailing_blocks=hl.agg.filter(
                    (idx_in_partition == partition_size - window_size) & (partition_idx < num_partitions),
                    hl.agg.collect(ht.key),
                ),
            )
        )

    new_partitioning = [
        hl.utils.Interval(
            start=agg_result.interval_bounds[i],
            end=agg_result.interval_bounds[i + 1],
            includes_start=True,
            includes_end=False,
        )
        for i in range(num_partitions - 1)
    ]
    # intervals_bounds normally length num_partitions+1, but could be one
    # less if last partition has exactly one row. Using index -1 for end of
    # last interval handles both cases.
    last_interval = hl.utils.Interval(
        start=agg_result.interval_bounds[num_partitions - 1],
        end=agg_result.interval_bounds[-1],
        includes_start=True,
        includes_end=True,
    )
    new_partitioning.append(last_interval)

    new_part_ht = hl.read_table(temp_file_name, _intervals=new_partitioning)

    grouped = new_part_ht._group_within_partitions("groups", block_size)
    A = grouped.select(ndarray=hl.nd.array(grouped.groups.map(lambda group: group.xs)))
    temp_file_name2 = hl.utils.new_temp_file("mt_to_table_of_ndarray", "A")
    A = A.checkpoint(temp_file_name2)

    if window_size is None:
        if return_checkpointed_table_also:
            return A, ht_checkpoint
        return A
    else:
        trailing_blocks = [
            hl.utils.Interval(
                start=agg_result.trailing_blocks[i],
                end=agg_result.interval_bounds[i + 1],
                includes_start=True,
                includes_end=False,
            )
            for i in range(num_partitions - 1)
        ]
        if num_partitions > 1:
            rekey_map = hl.dict([
                (agg_result.trailing_blocks[i], agg_result.interval_bounds[i + 1]) for i in range(num_partitions - 1)
            ])
        else:
            rekey_map = hl.empty_dict(ht.key.dtype, ht.key.dtype)

        trailing_blocks_ht = hl.read_table(temp_file_name, _intervals=trailing_blocks)
        trailing_blocks_ht = trailing_blocks_ht._group_within_partitions("groups", window_size)
        trailing_blocks_ht = trailing_blocks_ht.select(
            prev_window=hl.nd.array(trailing_blocks_ht.groups.map(lambda group: group.xs)).T
        )
        trailing_blocks_ht = trailing_blocks_ht.annotate_globals(rekey_map=hl.dict(rekey_map))
        trailing_blocks_ht = trailing_blocks_ht.key_by(**trailing_blocks_ht.rekey_map[trailing_blocks_ht.key])

        if return_checkpointed_table_also:
            return A, trailing_blocks_ht, ht_checkpoint
        return A, trailing_blocks_ht
