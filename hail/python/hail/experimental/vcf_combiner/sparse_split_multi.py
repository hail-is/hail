import hail as hl


def sparse_split_multi(sparse_mt, *, filter_changed_loci=False):
    """Splits multiallelic variants on a sparse matrix table.

    Analogous to :func:`.split_multi_hts` (splits entry fields) for sparse
    representations.

    Takes a dataset formatted like the output of :func:`.run_combiner`. The
    splitting will add `was_split` and `a_index` fields, as :func:`.vds.split_multi`
    does. This function drops the `LA` (local alleles) field, as it re-computes
    entry fields based on the new, split globals alleles.

    Variants are split thus:

    - A row with only one (reference) or two (reference and alternate) alleles
      is unchanged, as local and global alleles are the same.

    - A row with multiple alternate alleles  will be split, with one row for
      each alternate allele, and each row will contain two alleles: ref and alt.
      The reference and alternate allele will be minrepped using
      :func:`.min_rep`.

    The split multi logic handles the following entry fields:

    .. code-block:: text

        struct {
            LGT: call
            LAD: array<int32>
            DP: int32
            GQ: int32
            LPL: array<int32>
            RGQ: int32
            LPGT: call
            LA: array<int32>
            END: int32
        }

    All fields except for `LA` are optional, and only handled if they exist.

    - `LA` is used to find the corresponding local allele index for the desired
      global `a_index`, and then dropped from the resulting dataset. If `LA`
      does not contain the global `a_index`, calls will be downcoded to hom ref
      and `PL` will be set to missing.

    - `LGT` and `LPGT` are downcoded using the corresponding local `a_index`.
      They are renamed to `GT` and `PGT` respectively, as the resulting call is
      no longer local.

    - `LAD` is used to create an `AD` field consisting of the allele depths
      corresponding to the reference and global `a_index` alleles.

    - `DP` is preserved unchanged.

    - `GQ` is recalculated from the updated `PL`, if it exists, but otherwise
      preserved unchanged.

    - `PL` array elements are calculated from the minimum `LPL` value for all
      allele pairs that downcode to the desired one. (This logic is identical to
      the `PL` logic in :func:`~.split_multi_hts`.) If a row has an alternate
      allele but it is not present in `LA`, the `PL` field is set to missing.
      The `PL` for `ref/<NON_REF>` in that case can be drawn from `RGQ`.

    - `RGQ` (the reference genotype quality) is preserved unchanged.

    - `END` is untouched.

    Notes
    -----
    This version of split-multi doesn't deal with either duplicate loci (in
    which case the explode could possibly result in out-of-order rows, although
    the actual split_multi function also doesn't handle that case).

    It also checks that min-repping will not change the locus and will error if
    it does.

    Unlike the normal split_multi function. Sparse split multi will not filter
    ``*`` alleles. This is because a row with a bi-allelic spanning deletion
    may contain reference blocks that start at this position for other samples.

    Parameters
    ----------
    sparse_mt : :class:`.MatrixTable`
        Sparse MatrixTable to split.
    filter_changed_loci : :obj:`.bool`
        Rather than erroring if any REF/ALT pair changes locus under :func:`.min_rep`
        filter that variant instead.

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

    def struct_from_min_rep(i):
        return hl.bind(lambda mr:
                       (hl.case()
                        .when(ds.locus == mr.locus,
                              hl.struct(
                                  locus=ds.locus,
                                  alleles=[mr.alleles[0], mr.alleles[1]],
                                  a_index=i,
                                  was_split=True))
                        .when(filter_changed_loci,
                              hl.missing(hl.tstruct(locus=ds.locus.dtype, alleles=hl.tarray(hl.tstr),
                                                    a_index=hl.tint, was_split=hl.tbool)))
                        .or_error(
                            "Found non-left-aligned variant in sparse_split_multi\n"
                            + "old locus: " + hl.str(ds.locus) + "\n"
                            + "old ref  : " + ds.alleles[0] + "\n"
                            + "old alt  : " + ds.alleles[i] + "\n"
                            + "mr locus : " + hl.str(mr.locus) + "\n"
                            + "mr ref   : " + mr.alleles[0] + "\n"
                            + "mr alt   : " + mr.alleles[1])),
                       hl.min_rep(ds.locus, [ds.alleles[0], ds.alleles[i]]))

    explode_structs = hl.if_else(hl.len(ds.alleles) < 3,
                                 [hl.struct(
                                     locus=ds.locus,
                                     alleles=ds.alleles,
                                     a_index=1,
                                     was_split=False)],
                                 hl._sort_by(
                                     hl.if_else(
                                         filter_changed_loci,
                                         hl.range(1, hl.len(ds.alleles)).map(struct_from_min_rep).filter(hl.is_defined),
                                         hl.range(1, hl.len(ds.alleles)).map(struct_from_min_rep)),
                                     lambda l, r: hl._compare(l.alleles, r.alleles) < 0))

    ds = ds.annotate(**{new_id: explode_structs}).explode(new_id)

    def transform_entries(old_entry):
        def with_local_a_index(local_a_index):
            fields = set(old_entry.keys())

            def with_pl(pl):
                new_exprs = {}
                dropped_fields = ['LA']
                if 'LGT' in fields:
                    new_exprs['GT'] = hl.downcode(old_entry.LGT, hl.or_else(local_a_index, hl.len(old_entry.LA)))
                    dropped_fields.append('LGT')
                if 'LPGT' in fields:
                    new_exprs['PGT'] = hl.downcode(old_entry.LPGT, hl.or_else(local_a_index, hl.len(old_entry.LA)))
                    dropped_fields.append('LPGT')
                if 'LAD' in fields:
                    non_ref_ad = hl.or_else(old_entry.LAD[local_a_index], 0)  # zeroed if not in LAD
                    new_exprs['AD'] = hl.or_missing(
                        hl.is_defined(old_entry.LAD),
                        [hl.sum(old_entry.LAD) - non_ref_ad, non_ref_ad])
                    dropped_fields.append('LAD')
                if 'LPL' in fields:
                    new_exprs['PL'] = pl
                    if 'GQ' in fields:
                        new_exprs['GQ'] = hl.or_else(hl.gq_from_pl(pl), old_entry.GQ)

                    dropped_fields.append('LPL')

                return (hl.case()
                        .when(hl.len(ds.alleles) == 1,
                              old_entry.annotate(**{f[1:]: old_entry[f] for f in ['LGT', 'LPGT', 'LAD', 'LPL'] if f in fields}).drop(*dropped_fields))
                        .when(hl.or_else(old_entry.LGT.is_hom_ref(), False),
                              old_entry.annotate(**{f: old_entry[f'L{f}'] if f in ['GT', 'PGT'] else e for f, e in new_exprs.items()}).drop(*dropped_fields))
                        .default(old_entry.annotate(**new_exprs).drop(*dropped_fields)))

            if 'LPL' in fields:
                new_pl = hl.or_missing(
                    hl.is_defined(old_entry.LPL),
                    hl.or_missing(
                        hl.is_defined(local_a_index),
                        hl.range(0, 3).map(lambda i: hl.min(
                            hl.range(0, hl.triangle(hl.len(old_entry.LA)))
                            .filter(lambda j: hl.downcode(hl.unphased_diploid_gt_index_call(j), local_a_index) == hl.unphased_diploid_gt_index_call(i))
                            .map(lambda idx: old_entry.LPL[idx])))))
                return hl.bind(with_pl, new_pl)
            else:
                return with_pl(None)

        lai = hl.fold(lambda accum, elt:
                      hl.if_else(old_entry.LA[elt] == ds[new_id].a_index,
                                 elt, accum),
                      hl.missing(hl.tint32),
                      hl.range(0, hl.len(old_entry.LA)))
        return hl.bind(with_local_a_index, lai)

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
