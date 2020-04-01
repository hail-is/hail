import hail as hl

def densify(sparse_mt):
    """Convert sparse matrix table to a dense VCF-like representation by expanding reference blocks.

    Parameters
    ----------
    sparse_mt : :class:`.MatrixTable`
        Sparse MatrixTable to densify.  The first row key field must
        be named ``locus`` and have type ``locus``.  Must have an
        ``END`` entry field of type ``int32``.

    Returns
    -------
    :class:`.MatrixTable`
        The densified MatrixTable.  The ``END`` entry field is dropped.

    While computationally expensive, this
    operation is necessary for many downstream analyses, and should be thought of as
    roughly costing as much as reading a matrix table created by importing a dense
    project VCF.
    """
    if list(sparse_mt.row_key)[0] != 'locus' or not isinstance(sparse_mt.locus.dtype, hl.tlocus):
        raise ValueError("first row key field must be named 'locus' and have type 'locus'")
    if 'END' not in sparse_mt.entry or sparse_mt.END.dtype != hl.tint32:
        raise ValueError("'densify' requires 'END' entry field of type 'int32'")
    col_key_fields = list(sparse_mt.col_key)

    contigs = sparse_mt.locus.dtype.reference_genome.contigs
    contig_idx_map = hl.literal({contigs[i]:i for i in range(len(contigs))}, 'dict<str, int32>')
    mt = sparse_mt.annotate_rows(__contig_idx=contig_idx_map[sparse_mt.locus.contig])
    mt = mt.annotate_entries(__contig=mt.__contig_idx)

    t = mt._localize_entries('__entries', '__cols')
    t = t.annotate(
        __entries = hl.rbind(
            hl.scan.array_agg(
                lambda entry: hl.scan._prev_nonnull(hl.or_missing(hl.is_defined(entry.END), entry)),
                t.__entries),
            lambda prev_entries: hl.map(
                lambda i:
                hl.rbind(
                    prev_entries[i], t.__entries[i],
                    lambda prev_entry, entry:
                    hl.cond(
                        (~hl.is_defined(entry) &
                         (prev_entry.END >= t.locus.position) &
                         (prev_entry.__contig == t.__contig_idx)),
                        prev_entry,
                        entry)),
                hl.range(0, hl.len(t.__entries)))))
    mt = t._unlocalize_entries('__entries', '__cols', col_key_fields)
    mt = mt.drop('__contig_idx', '__contig', 'END')
    return mt
