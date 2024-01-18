import hail as hl
from hail.matrixtable import MatrixTable


def full_outer_join_mt(left: MatrixTable, right: MatrixTable) -> MatrixTable:
    """Performs a full outer join on `left` and `right`.

    Replaces row, column, and entry fields with the following:

     - `left_row` / `right_row`: structs of row fields from left and right.
     - `left_col` / `right_col`: structs of column fields from left and right.
     - `left_entry` / `right_entry`: structs of entry fields from left and right.

    Examples
    --------

    The following creates and joins two random datasets with disjoint sample ids
    but non-disjoint variant sets. We use :func:`.or_else` to attempt to find a
    non-missing genotype. If neither genotype is non-missing, then the genotype
    is set to missing. In particular, note that Samples `2` and `3` have missing
    genotypes for loci 1:1 and 1:2 because those loci are not present in `mt2`
    and these samples are not present in `mt1`

    >>> hl.reset_global_randomness()
    >>> mt1 = hl.balding_nichols_model(1, 2, 3)
    >>> mt2 = hl.balding_nichols_model(1, 2, 3)
    >>> mt2 = mt2.key_rows_by(locus=hl.locus(mt2.locus.contig,
    ...                                      mt2.locus.position+2),
    ...                       alleles=mt2.alleles)
    >>> mt2 = mt2.key_cols_by(sample_idx=mt2.sample_idx+2)
    >>> mt1.show()
    +---------------+------------+------+------+
    | locus         | alleles    | 0.GT | 1.GT |
    +---------------+------------+------+------+
    | locus<GRCh37> | array<str> | call | call |
    +---------------+------------+------+------+
    | 1:1           | ["A","C"]  | 0/0  | 0/0  |
    | 1:2           | ["A","C"]  | 0/1  | 0/1  |
    | 1:3           | ["A","C"]  | 0/0  | 0/1  |
    +---------------+------------+------+------+
    <BLANKLINE>
    >>> mt2.show()
    +---------------+------------+------+------+
    | locus         | alleles    | 2.GT | 3.GT |
    +---------------+------------+------+------+
    | locus<GRCh37> | array<str> | call | call |
    +---------------+------------+------+------+
    | 1:3           | ["A","C"]  | 0/1  | 1/1  |
    | 1:4           | ["A","C"]  | 1/1  | 0/1  |
    | 1:5           | ["A","C"]  | 0/0  | 0/0  |
    +---------------+------------+------+------+
    <BLANKLINE>
    >>> mt3 = hl.experimental.full_outer_join_mt(mt1, mt2)
    >>> mt3 = mt3.select_entries(GT=hl.or_else(mt3.left_entry.GT, mt3.right_entry.GT))
    >>> mt3.show()
    +---------------+------------+------+------+------+------+
    | locus         | alleles    | 0.GT | 1.GT | 2.GT | 3.GT |
    +---------------+------------+------+------+------+------+
    | locus<GRCh37> | array<str> | call | call | call | call |
    +---------------+------------+------+------+------+------+
    | 1:1           | ["A","C"]  | 0/0  | 0/0  | NA   | NA   |
    | 1:2           | ["A","C"]  | 0/1  | 0/1  | NA   | NA   |
    | 1:3           | ["A","C"]  | 0/0  | 0/1  | 0/1  | 1/1  |
    | 1:4           | ["A","C"]  | NA   | NA   | 1/1  | 0/1  |
    | 1:5           | ["A","C"]  | NA   | NA   | 0/0  | 0/0  |
    +---------------+------------+------+------+------+------+
    <BLANKLINE>

    Parameters
    ----------
    left : :class:`.MatrixTable`
    right : :class:`.MatrixTable`

    Returns
    -------
    :class:`.MatrixTable`
    """

    if [x.dtype for x in left.row_key.values()] != [x.dtype for x in right.row_key.values()]:
        raise ValueError(
            f"row key types do not match:\n"
            f"  left:  {list(left.row_key.values())}\n"
            f"  right: {list(right.row_key.values())}"
        )

    if [x.dtype for x in left.col_key.values()] != [x.dtype for x in right.col_key.values()]:
        raise ValueError(
            f"column key types do not match:\n"
            f"  left:  {list(left.col_key.values())}\n"
            f"  right: {list(right.col_key.values())}"
        )

    left = left.select_rows(left_row=left.row)
    left_t = left.localize_entries('left_entries', 'left_cols')
    right = right.select_rows(right_row=right.row)
    right_t = right.localize_entries('right_entries', 'right_cols')

    ht = left_t.join(right_t, how='outer')
    ht = ht.annotate_globals(
        left_keys=hl.group_by(
            lambda t: t[0],
            hl.enumerate(ht.left_cols.map(lambda x: hl.tuple([x[f] for f in left.col_key])), index_first=False),
        ).map_values(lambda elts: elts.map(lambda t: t[1])),
        right_keys=hl.group_by(
            lambda t: t[0],
            hl.enumerate(ht.right_cols.map(lambda x: hl.tuple([x[f] for f in right.col_key])), index_first=False),
        ).map_values(lambda elts: elts.map(lambda t: t[1])),
    )
    ht = ht.annotate_globals(
        key_indices=hl.array(ht.left_keys.key_set().union(ht.right_keys.key_set()))
        .map(lambda k: hl.struct(k=k, left_indices=ht.left_keys.get(k), right_indices=ht.right_keys.get(k)))
        .flatmap(
            lambda s: hl.case()
            .when(
                hl.is_defined(s.left_indices) & hl.is_defined(s.right_indices),
                hl.range(0, s.left_indices.length()).flatmap(
                    lambda i: hl.range(0, s.right_indices.length()).map(
                        lambda j: hl.struct(k=s.k, left_index=s.left_indices[i], right_index=s.right_indices[j])
                    )
                ),
            )
            .when(
                hl.is_defined(s.left_indices),
                s.left_indices.map(lambda elt: hl.struct(k=s.k, left_index=elt, right_index=hl.missing('int32'))),
            )
            .when(
                hl.is_defined(s.right_indices),
                s.right_indices.map(lambda elt: hl.struct(k=s.k, left_index=hl.missing('int32'), right_index=elt)),
            )
            .or_error('assertion error')
        )
    )
    ht = ht.annotate(
        __entries=ht.key_indices.map(
            lambda s: hl.struct(left_entry=ht.left_entries[s.left_index], right_entry=ht.right_entries[s.right_index])
        )
    )
    ht = ht.annotate_globals(
        __cols=ht.key_indices.map(
            lambda s: hl.struct(
                **{f: s.k[i] for i, f in enumerate(left.col_key)},
                left_col=ht.left_cols[s.left_index],
                right_col=ht.right_cols[s.right_index],
            )
        )
    )
    ht = ht.drop('left_entries', 'left_cols', 'left_keys', 'right_entries', 'right_cols', 'right_keys', 'key_indices')
    return ht._unlocalize_entries('__entries', '__cols', list(left.col_key))
