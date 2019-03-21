import hail as hl


def full_outer_join_mt(left: hl.MatrixTable, right: hl.MatrixTable) -> hl.MatrixTable:
    """Performs a full outer join on `left` and `right`.

    Replaces row, column, and entry fields with the following:

     - `left_row` / `right_row`: structs of row fields from left and right.
     - `left_col` / `right_col`: structs of column fields from left and right.
     - `left_entry` / `right_entry`: structs of entry fields from left and right.

    Parameters
    ----------
    left : :class:`.MatrixTable`
    right : :class:`.MatrixTable`

    Returns
    -------
    :class:`.MatrixTable`
    """

    if [x.dtype for x in left.row_key.values()] != [x.dtype for x in right.row_key.values()]:
        raise ValueError(f"row key types do not match:\n"
                         f"  left:  {list(left.row_key.values())}\n"
                         f"  right: {list(right.row_key.values())}")

    if [x.dtype for x in left.col_key.values()] != [x.dtype for x in right.col_key.values()]: 
        raise ValueError(f"column key types do not match:\n"
                         f"  left:  {list(left.col_key.values())}\n"
                         f"  right: {list(right.col_key.values())}")

    left = left.select_rows(left_row=left.row)
    left_t = left.localize_entries('left_entries', 'left_cols')
    right = right.select_rows(right_row=right.row)
    right_t = right.localize_entries('right_entries', 'right_cols')

    ht = left_t.join(right_t, how='outer')
    ht = ht.annotate_globals(
        left_keys=hl.group_by(
            lambda t: t[0],
            hl.zip_with_index(
                ht.left_cols.map(lambda x: hl.tuple([x[f] for f in left.col_key])), index_first=False)).map_values(
            lambda elts: elts.map(lambda t: t[1])),
        right_keys=hl.group_by(
            lambda t: t[0],
            hl.zip_with_index(
                ht.right_cols.map(lambda x: hl.tuple([x[f] for f in right.col_key])), index_first=False)).map_values(
            lambda elts: elts.map(lambda t: t[1])))
    ht = ht.annotate_globals(
        key_indices=hl.array(ht.left_keys.key_set().union(ht.right_keys.key_set()))
            .map(lambda k: hl.struct(k=k, left_indices=ht.left_keys.get(k), right_indices=ht.right_keys.get(k)))
            .flatmap(lambda s: hl.case()
                     .when(hl.is_defined(s.left_indices) & hl.is_defined(s.right_indices),
                           hl.range(0, s.left_indices.length()).flatmap(
                               lambda i: hl.range(0, s.right_indices.length()).map(
                                   lambda j: hl.struct(k=s.k, left_index=s.left_indices[i],
                                                       right_index=s.right_indices[j]))))
                     .when(hl.is_defined(s.left_indices),
                           s.left_indices.map(
                               lambda elt: hl.struct(k=s.k, left_index=elt, right_index=hl.null('int32'))))
                     .when(hl.is_defined(s.right_indices),
                           s.right_indices.map(
                               lambda elt: hl.struct(k=s.k, left_index=hl.null('int32'), right_index=elt)))
                     .or_error('assertion error')))
    ht = ht.annotate(__entries=ht.key_indices.map(lambda s: hl.struct(left_entry=ht.left_entries[s.left_index],
                                                                      right_entry=ht.right_entries[s.right_index])))
    ht = ht.annotate_globals(__cols=ht.key_indices.map(
        lambda s: hl.struct(**{f: s.k[i] for i, f in enumerate(left.col_key)},
                            left_col=ht.left_cols[s.left_index],
                            right_col=ht.right_cols[s.right_index])))
    ht = ht.drop('left_entries', 'left_cols', 'left_keys', 'right_entries', 'right_cols', 'right_keys', 'key_indices')
    return ht._unlocalize_entries('__entries', '__cols', list(left.col_key))
