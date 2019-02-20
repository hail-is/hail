import hail as hl


def sparse_split_multi(sparse_mt):
    """Splits multiallelic variants on a sparse MatrixTable.

    Takes a dataset formatted like the output of :func:`.vcf_combiner`. The
    splitting will add `was_split` and `a_index` fields, as :func:`.split_multi`
    does. This function drops the `LA` (local alleles) field, as it re-computes
    entry fields based on the new, split globals alleles.

    Variants are split thus:

    - A row with only one (reference) or two (reference and alternate) alleles
      in addition to the symbolic `<NON_REF>` will not be split.

    - A row with multiple alternate alleles in addition to the symbolic
      `<NON_REF>` allele be split, with one row for each alternate allele not
      including `<NON_REF>`, and each row will contain three alleles: ref, alt,
      and `<NON_REF>`. The reference and alternate allele will be minrepped using
      :func:`.min_rep`.

    The split multi logic handles the following entry fields:

        .. code-block:: text

        LGT: call
        LAD: array<int32>
        DP: int32
        GQ: int32
        LPL: array<int32>
        LPGT: call
        LA: array<int32>
        END: int32

    All fields except for `LA` are optional, and only handled if they exist.

    - `LA` is used to find the corresponding local allele index for the desired
      global `a_index`, and then dropped from the resulting dataset. If `LA`
      does not contain the global `a_index`, the index for the `<NON_REF>`
      allele is used to process the entry fields.

    - `LGT` and `LPGT` are downcoded using the corresponding local `a_index`.
      They are renamed to `GT` and `PGT` respectively, as the resulting call is
      no longer local.

    - `LAD` is used to create an `AD` field consisting of the allele depths
      corresponding to the reference, global `a_index` allele, and `<NON_REF>`
      allele.

    - `DP` is preserved unchanged.

    - `GQ` is recalculated from the updated `PL`, if it exists, but otherwise
      preserved unchanged.

    - `PL` array elements are calculated from the minimum `LPL` value for all
      allele pairs that downcode to the desired one. (This logic is identical to
      the `PL` logic in :func:`.split_mult_hts`, except with the addition of
      a special case where `<NON_REF>` alleles don't downcode to ref but instead
      are used for the (ref, `<NON_REF>`), (alt, `<NON_REF>`), and (`<NON_REF>`,
      `<NON_REF>`) cases.)

    - `END` is untouched.

    Notes
    -----
    This version of split-multi doesn't deal with either duplicate loci (in
    which case the explode could possibly result in out-of-order rows, although
    the actual split_multi function also doesn't handle that case).

    It also checks that min-repping will not change the locus and will error if
    it does. (I believe the VCF combiner checks that this holds true,
    currently.)

    Parameters
    ----------
    sparse_mt : :class:`.MatrixTable`
        Sparse MatrixTable to split.

    Returns
    -------
    :class:`.MatrixTable`
        The split MatrixTable in sparse format.

    """

    hl.methods.misc.require_row_key_variant(sparse_mt, "sparse_split_multi")

    entries = hl.utils.java.Env.get_uid()
    cols = hl.utils.java.Env.get_uid()
    ds = sparse_mt.localize_entries(entries, cols)
    new_id = hl.utils.java.Env.get_uid()

    non_ref = '<NON_REF>'

    def struct_from_min_rep(i):
        return hl.bind(lambda mr:
                       (hl.case()
                        .when(ds.locus == mr.locus,
                              hl.struct(
                                  locus=ds.locus,
                                  alleles=[mr.alleles[0], mr.alleles[1], non_ref],
                                  a_index=i,
                                  was_split=True))
                        .or_error("Found non-left-aligned variant in sparse_split_multi")),
                       hl.min_rep(ds.locus, [ds.alleles[0], ds.alleles[i]]))

    explode_structs = (hl.case()
                       .when(ds.alleles[-1] == non_ref,
                             hl.cond(hl.len(ds.alleles) < 4,
                                   [hl.struct(
                                       locus=ds.locus,
                                       alleles=ds.alleles,
                                       a_index=1,
                                       was_split=False)],
                                     hl._sort_by(
                                         hl.range(1, hl.len(ds.alleles) - 1)
                                             .map(struct_from_min_rep),
                                         lambda l, r: hl._compare(l.alleles, r.alleles) < 0
                                     )))
                       .or_error("'sparse_split_multi': Last allele in sparse representation was not '<NON_REF>'"))

    ds = ds.annotate(**{new_id: explode_structs}).explode(new_id)

    def transform_entries(old_entry):
        def with_non_ref_index(non_ref_index):
            def with_local_a_index(local_a_index):
                def downcodes_to(c, c2):
                    return (((hl.case()
                              .when(c2[0] == 1, c[0] == local_a_index)
                              .when(c2[0] == 2, c[0] == non_ref_index)
                              .default((c[0] != local_a_index) & (c[0] != non_ref_index))) &
                             (hl.case()
                              .when(c2[1] == 1, c[1] == local_a_index)
                              .when(c2[1] == 2, c[1] == non_ref_index)
                              .default((c[1] != local_a_index) & (c[1] != non_ref_index)))) |
                            ((hl.case()
                              .when(c2[0] == 1, c[1] == local_a_index)
                              .when(c2[0] == 2, c[1] == non_ref_index)
                              .default((c[1] != local_a_index) & (c[1] != non_ref_index))) &
                             (hl.case()
                              .when(c2[1] == 1, c[0] == local_a_index)
                              .when(c2[1] == 2, c[0] == non_ref_index)
                              .default((c[0] != local_a_index) & (c[0] != non_ref_index)))))

                new_pl = hl.or_missing(
                    hl.is_defined(old_entry.LPL),
                    (hl.range(0, 6).map(lambda i: hl.min(
                        (hl.range(0, hl.triangle(hl.len(old_entry.LA) + 1))
                         .filter(lambda j: hl.bind(downcodes_to,
                                                   hl.unphased_diploid_gt_index_call(j),
                                                   hl.unphased_diploid_gt_index_call(i)))
                         .map(lambda j: old_entry.LPL[j]))))))

                fields = set(old_entry.keys())

                def with_pl(pl):
                        new_exprs = {}
                        dropped_fields = ['LA']

                        if 'LGT' in fields:
                            new_exprs['GT'] = hl.downcode(old_entry.LGT, local_a_index)
                            dropped_fields.append('LGT')
                        if 'LPGT' in fields:
                            new_exprs['PGT'] = hl.downcode(old_entry.LPGT, local_a_index)
                            dropped_fields.append('LPGT')
                        if 'LAD' in fields:
                            new_exprs['AD'] = hl.or_missing(
                                hl.is_defined(old_entry.LAD),
                                [old_entry.LAD[0], old_entry.LAD[local_a_index], old_entry.LAD[non_ref_index]])
                            dropped_fields.append('LAD')
                        if 'LPL' in fields:
                            new_exprs['PL'] = pl
                            if 'GQ' in fields:
                                new_exprs['GQ'] = hl.gq_from_pl(pl)

                            dropped_fields.append('LPL')

                        return hl.cond(hl.len(ds.alleles) < 3,
                                       old_entry.annotate(**{f[1:]: old_entry[f] for f in ['LGT', 'LPGT', 'LAD', 'LPL'] if f in fields}).drop(*dropped_fields),
                                       old_entry.annotate(**new_exprs).drop(*dropped_fields))

                if 'LPL' in fields:
                    return hl.bind(with_pl, new_pl)
                else:
                    return with_pl(None)

            lai = hl.fold(lambda accum, elt:
                          hl.cond(old_entry.LA[elt] == ds[new_id].a_index,
                                  elt + 1, accum),
                          non_ref_index,
                          hl.range(0, hl.len(old_entry.LA)))
            return hl.bind(with_local_a_index, lai)

        return hl.bind(with_non_ref_index, hl.len(old_entry.LA))

    new_row = ds.row.annotate(**{
        'locus': ds[new_id].locus,
        'alleles': ds[new_id].alleles,
        'a_index': ds[new_id].a_index,
        'was_split': ds[new_id].was_split,
        entries: ds[entries].map(transform_entries)
    }).drop(new_id)

    ds = hl.Table(
        hl.ir.TableKeyBy(
            hl.ir.TableMapRows(
                hl.ir.TableKeyBy(ds._tir, ['locus']),
                new_row._ir),
            ['locus', 'alleles'],
            is_sorted=True))
    return ds._unlocalize_entries(entries, cols, list(sparse_mt.col_key.keys()))
