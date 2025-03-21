import json
import os

import hail as hl
from hail.genetics import ReferenceGenome
from hail.matrixtable import MatrixTable
from hail.typecheck import typecheck_method
from hail.utils.java import info, warning

extra_ref_globals_file = 'extra_reference_globals.json'


def read_vds(
    path,
    *,
    intervals=None,
    n_partitions=None,
    _assert_reference_type=None,
    _assert_variant_type=None,
    _warn_no_ref_block_max_length=True,
    _drop_end=False,
) -> 'VariantDataset':
    """Read in a :class:`.VariantDataset` written with :meth:`.VariantDataset.write`.

    Parameters
    ----------
    path: :obj:`str`

    Returns
    -------
    :class:`.VariantDataset`
    """
    if intervals or not n_partitions:
        reference_data = hl.read_matrix_table(VariantDataset._reference_path(path), _intervals=intervals)
        variant_data = hl.read_matrix_table(VariantDataset._variants_path(path), _intervals=intervals)
    else:
        assert n_partitions is not None
        reference_data = hl.read_matrix_table(VariantDataset._reference_path(path))
        intervals = reference_data._calculate_new_partitions(n_partitions)
        assert len(intervals) > 0
        reference_data = hl.read_matrix_table(VariantDataset._reference_path(path), _intervals=intervals)
        variant_data = hl.read_matrix_table(VariantDataset._variants_path(path), _intervals=intervals)

    # if LEN is missing, add it, _add_len is a no-op if LEN is already present
    reference_data = VariantDataset._add_len(reference_data)
    if _drop_end:
        if 'END' in reference_data.entry:
            reference_data = reference_data.drop('END')
    else:  # if END is missing, add it, _add_end is a no-op if END is already present
        reference_data = VariantDataset._add_end(reference_data)

    vds = VariantDataset(reference_data, variant_data)
    if VariantDataset.ref_block_max_length_field not in vds.reference_data.globals:
        fs = hl.current_backend().fs
        metadata_file = os.path.join(path, extra_ref_globals_file)
        if fs.exists(metadata_file):
            with fs.open(metadata_file, 'r') as f:
                metadata = json.load(f)
                vds.reference_data = vds.reference_data.annotate_globals(**metadata)
        elif _warn_no_ref_block_max_length:
            warning(
                "You are reading a VDS written with an older version of Hail."
                "\n  Hail now supports much faster interval filters on VDS, but you'll need to run either"
                "\n  `hl.vds.truncate_reference_blocks(vds, ...)` and write a copy (see docs) or patch the"
                "\n  existing VDS in place with `hl.vds.store_ref_block_max_length(vds_path)`."
            )

    return vds


def store_ref_block_max_length(vds_path):
    """Patches an existing VDS file to store the max reference block length for faster interval filters.

    This method permits :func:`.vds.filter_intervals` to remove reference data not overlapping a target interval.

    This method is able to patch an existing VDS file in-place, without copying all the data. However,
    if significant downstream interval filtering is anticipated, it may be advantageous to run
    :func:`.vds.truncate_reference_blocks` to truncate long reference blocks and make interval filters
    even faster. However, truncation requires rewriting the entire VDS.


    Examples
    --------
    >>> hl.vds.store_ref_block_max_length('gs://path/to/my.vds')  # doctest: +SKIP

    See Also
    --------
    :func:`.vds.filter_intervals`, :func:`.vds.truncate_reference_blocks`.

    Parameters
    ----------
    vds_path : :obj:`str`
    """
    vds = read_vds(vds_path, _warn_no_ref_block_max_length=False)

    if VariantDataset.ref_block_max_length_field in vds.reference_data.globals:
        warning(f"VDS at {vds_path} already contains a global annotation with the max reference block length")
        return
    rd = vds.reference_data
    fs = hl.current_backend().fs
    ref_block_max_len = rd.aggregate_entries(hl.agg.max(rd.LEN))
    with fs.open(os.path.join(vds_path, extra_ref_globals_file), 'w') as f:
        json.dump({VariantDataset.ref_block_max_length_field: ref_block_max_len}, f)


class VariantDataset:
    """Class for representing cohort-level genomic data.

    This class facilitates a sparse, split representation of genomic data in
    which reference block data and variant data are contained in separate
    :class:`.MatrixTable` objects.

    Parameters
    ----------
    reference_data : :class:`.MatrixTable`
        MatrixTable containing only reference block data.
    variant_data : :class:`.MatrixTable`
        MatrixTable containing only variant data.
    """

    #: Name of global field that indicates max reference block length.
    ref_block_max_length_field = 'ref_block_max_length'

    @staticmethod
    def _reference_path(base: str) -> str:
        return os.path.join(base, 'reference_data')

    @staticmethod
    def _variants_path(base: str) -> str:
        return os.path.join(base, 'variant_data')

    @staticmethod
    def from_merged_representation(
        mt, *, ref_block_indicator_field='END', ref_block_fields=(), infer_ref_block_fields: bool = True, is_split=False
    ):
        """Create a VariantDataset from a sparse MatrixTable containing variant and reference data."""

        if ref_block_indicator_field not in ('END', 'LEN'):
            raise ValueError(
                f'Invalid `ref_block_indicator_field` `{ref_block_indicator_field}` one of `LEN` or `END` expected'
            )

        if ref_block_indicator_field not in mt.entry:
            raise ValueError(
                f'VariantDataset.from_merged_representation: expect field `{ref_block_indicator_field}` in matrix table entry'
            )

        if 'LA' not in mt.entry and not is_split:
            raise ValueError(
                'VariantDataset.from_merged_representation: expect field `LA` in matrix table entry.'
                '\n  If this dataset is already split into biallelics, use `is_split=True` to permit a conversion'
                ' with no `LA` field.'
            )

        if 'GT' not in mt.entry and 'LGT' not in mt.entry:
            raise ValueError(
                'VariantDataset.from_merged_representation: expect field `LGT` or `GT` in matrix table entry'
            )

        n_rows_to_use = 100
        info(f"inferring reference block fields from missingness patterns in first {n_rows_to_use} rows")
        used_ref_block_fields = set(ref_block_fields)
        used_ref_block_fields.add(ref_block_indicator_field)

        if infer_ref_block_fields:
            mt_head = mt.head(n_rows=n_rows_to_use)
            for k, any_present in zip(
                list(mt_head.entry),
                mt_head.aggregate_entries(
                    hl.agg.filter(
                        hl.is_defined(mt_head[ref_block_indicator_field]),
                        tuple(hl.agg.any(hl.is_defined(mt_head[x])) for x in mt_head.entry),
                    )
                ),
            ):
                if any_present:
                    used_ref_block_fields.add(k)

        gt_field = 'LGT' if 'LGT' in mt.entry else 'GT'

        # remove the LA field, which is trivial for reference blocks and does not need to be represented
        if 'LA' in used_ref_block_fields:
            used_ref_block_fields.remove('LA')

        info(
            "Including the following fields in reference block table:"
            + "".join(f"\n  {k!r}" for k in mt.entry if k in used_ref_block_fields)
        )

        rmt = mt.filter_entries(
            hl.case()
            .when(hl.is_missing(mt[ref_block_indicator_field]), False)
            .when(hl.is_defined(mt[ref_block_indicator_field]) & mt[gt_field].is_hom_ref(), True)
            .or_error(
                hl.str(
                    f'cannot create VDS from merged representation - found {ref_block_indicator_field} field with non-reference genotype at '
                )
                + hl.str(mt.locus)
                + hl.str(' / ')
                + hl.str(mt.col_key[0])
            )
        )
        rmt = rmt.select_entries(*(x for x in rmt.entry if x in used_ref_block_fields))
        rmt = rmt.filter_rows(hl.agg.count() > 0)

        rmt = rmt.key_rows_by('locus').select_rows().select_cols()
        if ref_block_indicator_field == 'END':
            rmt = VariantDataset._add_len(rmt)
        else:  # ref_block_indicator_field is 'LEN'
            rmt = VariantDataset._add_end(rmt)

        if is_split:
            rmt = rmt.distinct_by_row()

        vmt = (
            mt.filter_entries(hl.is_missing(mt[ref_block_indicator_field]))
            .drop(ref_block_indicator_field)
            ._key_rows_by_assert_sorted('locus', 'alleles')
        )
        vmt = vmt.filter_rows(hl.agg.count() > 0)

        return VariantDataset(rmt, vmt)

    def __init__(self, reference_data: MatrixTable, variant_data: MatrixTable):
        self.reference_data: MatrixTable = reference_data
        self.variant_data: MatrixTable = variant_data

        self.validate(check_data=False)

    def write(self, path, **kwargs):
        """Write to `path`.

        Any optional parameter from :meth:`.MatrixTable.write` can be passed as
        a keyword paramter to this method.
        """

        # NOTE: Populate LEN and drop END from reference data to align with VCF 4.5.
        # Furthermore, since LEN values are smaller and more likely to be close
        # or the same as neighboring values, we expect that after small integer
        # compression and general purpose data compression that reference data should
        # be smaller using LEN over END
        rd = self.reference_data
        if 'LEN' not in rd.entry:
            rd = VariantDataset._add_len(rd)
        if 'END' in rd.entry:
            rd = rd.drop('END')

        rd.write(VariantDataset._reference_path(path), **kwargs)
        self.variant_data.write(VariantDataset._variants_path(path), **kwargs)

    def checkpoint(self, path, **kwargs) -> 'VariantDataset':
        """Write to `path` and then read from `path`."""
        self.write(path, **kwargs)
        return read_vds(path)

    def n_samples(self) -> int:
        """The number of samples present."""
        return self.reference_data.count_cols()

    @property
    def reference_genome(self) -> ReferenceGenome:
        """Dataset reference genome.

        Returns
        -------
        :class:`.ReferenceGenome`
        """
        return self.reference_data.locus.dtype.reference_genome

    @typecheck_method(check_data=bool)
    def validate(self, *, check_data: bool = True):
        """Eagerly checks necessary representational properties of the VDS."""

        rd = self.reference_data
        vd = self.variant_data

        def error(msg):
            raise ValueError(f'VDS.validate: {msg}')

        rd_row_key = rd.row_key.dtype
        if (
            not isinstance(rd_row_key, hl.tstruct)
            or len(rd_row_key) != 1
            or not rd_row_key.fields[0] == 'locus'
            or not isinstance(rd_row_key.types[0], hl.tlocus)
        ):
            error(f"expect reference data to have a single row key 'locus' of type locus, found {rd_row_key}")

        vd_row_key = vd.row_key.dtype
        if (
            not isinstance(vd_row_key, hl.tstruct)
            or len(vd_row_key) != 2
            or not vd_row_key.fields == ('locus', 'alleles')
            or not isinstance(vd_row_key.types[0], hl.tlocus)
            or vd_row_key.types[1] != hl.tarray(hl.tstr)
        ):
            error(
                f"expect variant data to have a row key {{'locus': locus<rg>, alleles: array<str>}}, found {vd_row_key}"
            )

        rd_col_key = rd.col_key.dtype
        if not isinstance(rd_col_key, hl.tstruct) or len(rd_row_key) != 1 or rd_col_key.types[0] != hl.tstr:
            error(f"expect reference data to have a single col key of type string, found {rd_col_key}")

        vd_col_key = vd.col_key.dtype
        if not isinstance(vd_col_key, hl.tstruct) or len(vd_col_key) != 1 or vd_col_key.types[0] != hl.tstr:
            error(f"expect variant data to have a single col key of type string, found {vd_col_key}")

        end_exists = 'END' in rd.entry
        len_exists = 'LEN' in rd.entry
        if not (end_exists or len_exists):
            error("expect at least one of 'END' or 'LEN' in entry of reference data")
        if end_exists and rd.END.dtype != hl.tint32:
            error("'END' field in entry of reference data must have type tint32")
        if len_exists and rd.LEN.dtype != hl.tint32:
            error("'LEN' field in entry of reference data must have type tint32")

        if check_data:
            # check cols
            ref_cols = rd.col_key.collect()
            var_cols = vd.col_key.collect()
            if len(ref_cols) != len(var_cols):
                error(
                    f"mismatch in number of columns: reference data has {ref_cols} columns, variant data has {var_cols} columns"
                )

            if ref_cols != var_cols:
                first_mismatch = 0
                while ref_cols[first_mismatch] == var_cols[first_mismatch]:
                    first_mismatch += 1
                error(
                    f"mismatch in columns keys: ref={ref_cols[first_mismatch]}, var={var_cols[first_mismatch]} at position {first_mismatch}"
                )

            # check locus distinctness
            n_rd_rows = rd.count_rows()
            n_distinct = rd.distinct_by_row().count_rows()

            if n_distinct != n_rd_rows:
                error(f'reference data loci are not distinct: found {n_rd_rows} rows, but {n_distinct} distinct loci')

            # check END field
            end_exprs = dict(
                missing_end=hl.agg.filter(hl.is_missing(rd.END), hl.agg.take((rd.row_key, rd.col_key), 5)),
                end_before_position=hl.agg.filter(rd.END < rd.locus.position, hl.agg.take((rd.row_key, rd.col_key), 5)),
            )
            if VariantDataset.ref_block_max_length_field in rd.globals:
                rbml = rd[VariantDataset.ref_block_max_length_field]
                end_exprs['blocks_too_long'] = hl.agg.filter(
                    rd.END - rd.locus.position + 1 > rbml, hl.agg.take((rd.row_key, rd.col_key), 5)
                )

            res = rd.aggregate_entries(hl.struct(**end_exprs))

            if res.missing_end:
                error(
                    'found records in reference data with missing END field\n  '
                    + '\n  '.join(str(x) for x in res.missing_end)
                )
            if res.end_before_position:
                error(
                    'found records in reference data with END before locus position\n  '
                    + '\n  '.join(str(x) for x in res.end_before_position)
                )
            blocks_too_long = res.get('blocks_too_long', [])
            if blocks_too_long:
                error(
                    'found records in reference data with blocks larger than `ref_block_max_length`\n  '
                    + '\n  '.join(str(x) for x in blocks_too_long)
                )

    def _same(self, other: 'VariantDataset'):
        return self.reference_data._same(other.reference_data) and self.variant_data._same(other.variant_data)

    @staticmethod
    def _add_len(rd):
        if 'LEN' in rd.entry:
            return rd
        if 'END' in rd.entry:
            return rd.annotate_entries(LEN=rd.END - rd.locus.position + 1)
        raise ValueError('Need `END` to compute `LEN` in reference data')

    @staticmethod
    def _add_end(rd):
        if 'END' in rd.entry:
            return rd
        if 'LEN' in rd.entry:
            return rd.annotate_entries(END=rd.LEN + rd.locus.position - 1)
        raise ValueError('Need `LEN` to compute `END` in reference data')

    def union_rows(*vdses):
        """Combine many VDSes with the same samples but disjoint variants.

        **Examples**

        If a dataset is imported as VDS in chromosome-chunks, the following will combine them into
        one VDS:

        >>> vds_paths = ['chr1.vds', 'chr2.vds']  # doctest: +SKIP
        ... vds_per_chrom = [hl.vds.read_vds(path) for path in vds_paths)  # doctest: +SKIP
        ... hl.vds.VariantDataset.union_rows(*vds_per_chrom)  # doctest: +SKIP

        """

        fd = hl.vds.VariantDataset.ref_block_max_length_field
        mts = [vds.reference_data for vds in vdses]
        n_with_ref_max_len = len([mt for mt in mts if fd in mt.globals])
        any_ref_max = n_with_ref_max_len > 0
        all_ref_max = n_with_ref_max_len == len(mts)

        # if some mts have max ref len but not all, drop it
        if all_ref_max:
            new_ref_mt = hl.MatrixTable.union_rows(*mts).annotate_globals(**{
                fd: hl.max([mt.index_globals()[fd] for mt in mts])
            })
        else:
            if any_ref_max:
                mts = [mt.drop(fd) if fd in mt.globals else mt for mt in mts]
            new_ref_mt = hl.MatrixTable.union_rows(*mts)

        new_var_mt = hl.MatrixTable.union_rows(*(vds.variant_data for vds in vdses))
        return hl.vds.VariantDataset(new_ref_mt, new_var_mt)
