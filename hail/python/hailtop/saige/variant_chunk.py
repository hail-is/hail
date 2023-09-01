from typing import List, Optional

import hail as hl
from hail.methods.qc import require_row_key_variant


class VariantChunk:
    idx: int = 1

    def __init__(self, interval: hl.Interval, groups: Optional[List[str]] = None):
        idx = VariantChunk.idx
        VariantChunk.idx += 1
        self.idx = idx
        self.interval = interval
        self.groups = groups

    @property
    def name(self) -> str:
        return (
            self.interval.start.contig + '_' + str(self.interval.start.position) + '_' + str(self.interval.end.position)
        )

    def to_interval_str(self) -> str:
        return str(self.interval)

    def to_dict(self):
        return {'idx': self.idx, 'interval': self.to_interval_str(), 'groups': self.groups}


def create_chunks_by_contig(
    variants: hl.Table, max_count_per_chunk: int, max_span_per_chunk: int
) -> List[VariantChunk]:
    group_metadata = variants.aggregate(
        hl.agg.group_by(variants.locus.contig, hl.agg.approx_cdf(variants.locus.position, 200))
    )

    chunks = []
    for contig, cdf in group_metadata.items():
        first_rank = 0
        first_locus = cdf.values[0]
        cur_locus = cdf.values[0]

        for i in range(1, len(cdf.values)):
            cur_locus = cdf.values[i]
            cur_rank = cdf.ranks[i]
            chunk_size = cur_rank - first_rank  # approximately how many rows are in interval [ first_locus, cur_locus )
            chunk_span = cur_locus.position - first_locus.position
            if chunk_size > max_count_per_chunk or chunk_span > max_span_per_chunk:
                interval = hl.Interval(first_locus, cur_locus, includes_start=True, includes_end=False)
                chunks.append(VariantChunk(interval))
                first_rank = cur_rank
                first_locus = cur_locus

        interval = hl.Interval(
            first_locus,
            cur_locus,
            includes_start=True,
            includes_end=True,
        )
        chunks.append(VariantChunk(interval))

    return chunks


def create_chunks_by_group(
    variants: hl.Table, group: hl.ArrayExpression, max_count_per_chunk: int, max_span_per_chunk: int
) -> List[VariantChunk]:
    assert require_row_key_variant(variants, 'saige')

    rg = variants.locus.dtype.reference_genome

    variants = variants.select(group=group).explode('group')

    group_metadata = variants.aggregate(
        hl.agg.group_by(
            variants.group.group,
            hl.struct(
                contig=hl.array(hl.agg.collect_as_set(variants.locus.contig)),
                start=hl.agg.min(variants.locus.position),
                end=hl.agg.max(variants.locus.position),
                count=hl.agg.count(variants.locus),
            ),
        )
    )

    group_metadata = sorted(list(group_metadata.items()), key=lambda x: (x.contig, x.start))

    def variant_chunk_from_groups(groups: List[hl.Struct]) -> VariantChunk:
        contig = groups[0].contig
        start = min(g.start for g in groups)
        end = max(g.end for g in groups)
        return VariantChunk(
            hl.Interval(
                hl.Locus(contig, start, reference_genome=rg),
                hl.Locus(contig, end, reference_genome=rg),
                includes_start=True,
                includes_end=True,
            ),
            [g.group for g in groups],
        )

    chunks = []

    groups = [group_metadata[0]]
    current_count = 0

    for group, metadata in group_metadata[1:]:
        first_group = groups[0]
        if (
            (current_count + group.count > max_count_per_chunk)
            or (group.contig != first_group.contig)
            or (group.end - first_group.end > max_span_per_chunk)
        ):
            chunks.append(variant_chunk_from_groups(groups))
            current_count = 0

        groups.append(group)
        current_count += group.count

    chunks.append(variant_chunk_from_groups(groups))

    return chunks


def create_variant_chunks(
    dataset: hl.MatrixTable,
    *,
    group: Optional[hl.ArrayExpression] = None,
    max_count_per_chunk: int = 5000,
    max_span_per_chunk: int = 5_000_000,
) -> List[VariantChunk]:
    if group is not None:
        assert dataset == group._indices.source
    require_row_key_variant(dataset, 'saige')

    variants = dataset.rows()

    if group is None:
        return create_chunks_by_contig(variants, max_count_per_chunk, max_span_per_chunk)
    return create_chunks_by_group(variants, group, max_count_per_chunk, max_span_per_chunk)


class VariantChunks:
    @staticmethod
    def from_bed(file: str) -> 'VariantChunks':
        raise NotImplementedError

    @staticmethod
    def from_locus_intervals(file: str) -> 'VariantChunks':
        raise NotImplementedError

    @staticmethod
    def from_matrix_table(
        mt: hl.MatrixTable,
        *,
        group: Optional[hl.Expression] = None,
        max_count_per_chunk: int = 5000,
        max_span_per_chunk: int = 5_000_000,
    ) -> 'VariantChunks':
        chunks = create_variant_chunks(
            mt, group=group, max_count_per_chunk=max_count_per_chunk, max_span_per_chunk=max_span_per_chunk
        )
        return VariantChunks(chunks)

    def __init__(self, variant_chunks: List[VariantChunk]):
        self.variant_chunks = variant_chunks

    def __iter__(self):
        for chunk in self.variant_chunks:
            yield chunk
