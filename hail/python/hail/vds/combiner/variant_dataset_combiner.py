import collections
import hashlib
import json
import os
import sys
import uuid
from itertools import chain
from math import floor, log
from typing import Collection, Dict, List, NamedTuple, Optional, Union

import hail as hl
from hail.expr import HailType, tmatrix
from hail.utils import FatalError, Interval
from hail.utils.java import info, warning
from .combine import (
    combine_variant_datasets,
    transform_gvcf,
    defined_entry_fields,
    make_variant_stream,
    make_reference_stream,
    combine_r,
    calculate_even_genome_partitioning,
    calculate_new_intervals,
    combine,
)
from ..variant_dataset import VariantDataset


class VDSMetadata(NamedTuple):
    """The path to a Variant Dataset and the number of samples within.

    Parameters
    ----------
    path : :class:`str`
        Path to the variant dataset.
    n_samples : :class:`int`
        Number of samples contained within the Variant Dataset at `path`.

    """

    path: str
    n_samples: int


class CombinerOutType(NamedTuple):
    """A container for the types of a VDS"""

    reference_type: tmatrix
    variant_type: tmatrix


FAST_CODEC_SPEC = """{
  "name": "LEB128BufferSpec",
  "child": {
    "name": "BlockingBufferSpec",
    "blockSize": 65536,
    "child": {
      "name": "ZstdBlockBufferSpec",
      "blockSize": 65536,
      "child": {
        "name": "StreamBlockBufferSpec"
      }
    }
  }
}"""


class VariantDatasetCombiner:  # pylint: disable=too-many-instance-attributes
    """A restartable and failure-tolerant method for combining one or more GVCFs and Variant Datasets.

    Examples
    --------

    A Variant Dataset comprises one or more sequences. A new Variant Dataset is constructed from
    GVCF files and/or extant Variant Datasets. For example, the following produces a new Variant
    Dataset from four GVCF files containing whole genome sequences ::

        gvcfs = [
            'gs://bucket/sample_10123.g.vcf.bgz',
            'gs://bucket/sample_10124.g.vcf.bgz',
            'gs://bucket/sample_10125.g.vcf.bgz',
            'gs://bucket/sample_10126.g.vcf.bgz',
        ]

        combiner = hl.vds.new_combiner(
            output_path='gs://bucket/dataset.vds',
            temp_path='gs://1-day-temp-bucket/',
            gvcf_paths=gvcfs,
            use_genome_default_intervals=True,
        )

        combiner.run()

        vds = hl.read_vds('gs://bucket/dataset.vds')

    The following combines four new samples from GVCFs with multiple extant Variant Datasets::

        gvcfs = [
            'gs://bucket/sample_10123.g.vcf.bgz',
            'gs://bucket/sample_10124.g.vcf.bgz',
            'gs://bucket/sample_10125.g.vcf.bgz',
            'gs://bucket/sample_10126.g.vcf.bgz',
        ]

        vdses = [
            'gs://bucket/hgdp.vds',
            'gs://bucket/1kg.vds'
        ]

        combiner = hl.vds.new_combiner(
            output_path='gs://bucket/dataset.vds',
            temp_path='gs://1-day-temp-bucket/',
            save_path='gs://1-day-temp-bucket/',
            gvcf_paths=gvcfs,
            vds_paths=vdses,
            use_genome_default_intervals=True,
        )

        combiner.run()

        vds = hl.read_vds('gs://bucket/dataset.vds')


    The speed of the Variant Dataset Combiner critically depends on data partitioning. Although the
    partitioning is fully customizable, two high-quality partitioning strategies are available by
    default, one for exomes and one for genomes. These partitioning strategies can be enabled,
    respectively, with the parameters: ``use_exome_default_intervals=True`` and
    ``use_genome_default_intervals=True``.

    The combiner serializes itself to `save_path` so that it can be restarted after failure.

    Parameters
    ----------
    save_path : :class:`str`
        The location to store this VariantDatasetCombiner plan. A failed execution can be restarted
        using this plan.
    output_path : :class:`str`
        The location to store the new VariantDataset.
    temp_path : :class:`str`
        The location to store temporary intermediates. We recommend using a bucket with an automatic
        deletion or lifecycle policy.
    reference_genome : :class:`.ReferenceGenome`
        The reference genome to which all inputs (GVCFs and Variant Datasets) are aligned.
    branch_factor : :class:`int`
        The number of Variant Datasets to combine at once.
    target_records : :class:`int`
        The target number of variants per partition.
    gvcf_batch_size : :class:`int`
        The number of GVCFs to combine into a Variant Dataset at once.
    contig_recoding : :class:`dict` mapping :class:`str` to :class:`str` or :obj:`None`
        This mapping is applied to GVCF contigs before importing them into Hail. This is used to
        handle GVCFs containing invalid contig names. For example, GRCh38 GVCFs which contain the
        contig "1" instead of the correct "chr1".
    vdses : :class:`list` of :class:`.VDSMetadata`
        A list of Variant Datasets to combine. Each dataset is identified by a
        :class:`.VDSMetadata`, which is a pair of a path and the number of samples in said Variant
        Dataset.
    gvcfs : :class:`list` of :class:`.str`
        A list of paths of GVCF files to combine.
    gvcf_sample_names : :class:`list` of :class:`str` or :obj:`None`
        List of names to use for the samples from the GVCF files. Must be the same length as
        `gvcfs`. Must be specified if `gvcf_external_header` is specified.
    gvcf_external_header : :class:`str` or :obj:`None`
        A path to a file containing a VCF header which is applied to all GVCFs. Must be specified if
        `gvcf_sample_names` is specified.
    gvcf_import_intervals : :class:`list` of :class:`.Interval`
        A list of intervals defining how to partition the GVCF files. The same partitioning is used
        for all GVCF files. Finer partitioning yields more parallelism but less work per task.
    gvcf_info_to_keep : :class:`list` of :class:`str` or :obj:`None`
        GVCF ``INFO`` fields to keep in the ``gvcf_info`` entry field. By default, all fields
        except ``END`` and ``DP`` are kept.
    gvcf_reference_entry_fields_to_keep : :class:`list` of :class:`str` or :obj:`None`
        Genotype fields to keep in the reference table. If empty, the first 10,000 reference block
        rows of ``mt`` will be sampled and all fields found to be defined other than ``GT``, ``AD``,
        and ``PL`` will be entry fields in the resulting reference matrix in the dataset.

    """

    _default_gvcf_batch_size = 50
    _default_branch_factor = 100
    _default_target_records = 24_000
    _gvcf_merge_task_limit = 150_000

    # These are used to calculate intervals for reading GVCFs in the combiner
    # The genome interval size results in 2568 partitions for GRCh38. The exome
    # interval size assumes that they are around 2% the size of a genome and
    # result in 65 partitions for GRCh38.
    default_genome_interval_size = 1_200_000
    "A reasonable partition size in basepairs given the density of genomes."

    default_exome_interval_size = 60_000_000
    "A reasonable partition size in basepairs given the density of exomes."

    __serialized_slots__ = [
        '_save_path',
        '_output_path',
        '_temp_path',
        '_reference_genome',
        '_dataset_type',
        '_gvcf_type',
        '_branch_factor',
        '_target_records',
        '_gvcf_batch_size',
        '_contig_recoding',
        '_vdses',
        '_gvcfs',
        '_gvcf_external_header',
        '_gvcf_sample_names',
        '_gvcf_import_intervals',
        '_gvcf_info_to_keep',
        '_gvcf_reference_entry_fields_to_keep',
        '_call_fields',
    ]

    __slots__ = tuple(__serialized_slots__ + ['_uuid', '_job_id', '__intervals_cache'])

    def __init__(
        self,
        *,
        save_path: str,
        output_path: str,
        temp_path: str,
        reference_genome: hl.ReferenceGenome,
        dataset_type: CombinerOutType,
        gvcf_type: Optional[tmatrix] = None,
        branch_factor: int = _default_branch_factor,
        target_records: int = _default_target_records,
        gvcf_batch_size: int = _default_gvcf_batch_size,
        contig_recoding: Optional[Dict[str, str]] = None,
        call_fields: Collection[str],
        vdses: List[VDSMetadata],
        gvcfs: List[str],
        gvcf_sample_names: Optional[List[str]] = None,
        gvcf_external_header: Optional[str] = None,
        gvcf_import_intervals: List[Interval],
        gvcf_info_to_keep: Optional[Collection[str]] = None,
        gvcf_reference_entry_fields_to_keep: Optional[Collection[str]] = None,
    ):
        if gvcf_import_intervals:
            interval = gvcf_import_intervals[0]
            if not isinstance(interval.point_type, hl.tlocus):
                raise ValueError(f'intervals point type must be a locus, found {interval.point_type}')
            if interval.point_type.reference_genome != reference_genome:
                raise ValueError(
                    f'mismatch in intervals ({interval.point_type.reference_genome}) '
                    f'and reference genome ({reference_genome}) types'
                )
        if (gvcf_sample_names is None) != (gvcf_external_header is None):
            raise ValueError("both 'gvcf_sample_names' and 'gvcf_external_header' must be set or unset")
        if gvcf_sample_names is not None and len(gvcf_sample_names) != len(gvcfs):
            raise ValueError(
                "'gvcf_sample_names' and 'gvcfs' must have the same length " f'{len(gvcf_sample_names)} != {len(gvcfs)}'
            )
        if branch_factor < 2:
            raise ValueError(f"'branch_factor' must be at least 2, found {branch_factor}")
        if gvcf_batch_size < 1:
            raise ValueError(f"'gvcf_batch_size' must be at least 1, found {gvcf_batch_size}")

        self._save_path = save_path
        self._output_path = output_path
        self._temp_path = temp_path
        self._reference_genome = reference_genome
        self._dataset_type = dataset_type
        self._gvcf_type = gvcf_type
        self._branch_factor = branch_factor
        self._target_records = target_records
        self._contig_recoding = contig_recoding
        self._call_fields = list(call_fields)
        self._vdses = collections.defaultdict(list)
        for vds in vdses:
            self._vdses[max(1, floor(log(vds.n_samples, self._branch_factor)))].append(vds)
        self._gvcfs = gvcfs
        self._gvcf_sample_names = gvcf_sample_names
        self._gvcf_external_header = gvcf_external_header
        self._gvcf_import_intervals = gvcf_import_intervals
        self._gvcf_info_to_keep = set(gvcf_info_to_keep) if gvcf_info_to_keep is not None else None
        self._gvcf_reference_entry_fields_to_keep = (
            set(gvcf_reference_entry_fields_to_keep) if gvcf_reference_entry_fields_to_keep is not None else None
        )

        self._uuid = uuid.uuid4()
        self._job_id = 1
        self.__intervals_cache = {}
        self._gvcf_batch_size = gvcf_batch_size

    @property
    def gvcf_batch_size(self):
        """The number of GVCFs to combine into a Variant Dataset at once."""
        return self._gvcf_batch_size

    @gvcf_batch_size.setter
    def gvcf_batch_size(self, value: int):
        if value * len(self._gvcf_import_intervals) > VariantDatasetCombiner._gvcf_merge_task_limit:
            old_value = value
            value = VariantDatasetCombiner._gvcf_merge_task_limit // len(self._gvcf_import_intervals)
            warning(f'gvcf_batch_size of {old_value} would produce too many tasks ' f'using {value} instead')
        self._gvcf_batch_size = value

    def __eq__(self, other):
        if other.__class__ != VariantDatasetCombiner:
            return False
        for slot in self.__serialized_slots__:
            if getattr(self, slot) != getattr(other, slot):
                return False
        return True

    @property
    def finished(self) -> bool:
        """Have all GVCFs and input Variant Datasets been combined?"""
        return not self._gvcfs and not self._vdses

    def save(self):
        """Save a :class:`.VariantDatasetCombiner` to its `save_path`."""
        fs = hl.current_backend().fs
        try:
            backup_path = self._save_path + '.bak'
            if fs.exists(self._save_path):
                fs.copy(self._save_path, backup_path)
            with fs.open(self._save_path, 'w') as out:
                json.dump(self, out, indent=2, cls=Encoder)
            if fs.exists(backup_path):
                fs.remove(backup_path)
        except OSError as e:
            # these messages get printed, because there is absolutely no guarantee
            # that the hail context is in a sane state if any of the above operations
            # fail
            print(f'Failed saving {self.__class__.__name__} state at {self._save_path}')
            print(f'An attempt was made to copy {self._save_path} to {backup_path}')
            print('An old version of this state may be there.')
            print(
                'Dumping current state as json to standard output, you may wish '
                'to save this output in order to resume the combiner.'
            )
            json.dump(self, sys.stdout, indent=2, cls=Encoder)
            print()
            raise e

    def run(self):
        """Combine the specified GVCFs and Variant Datasets."""
        flagname = 'no_ir_logging'
        prev_flag_value = hl._get_flags(flagname).get(flagname)
        hl._set_flags(**{flagname: '1'})

        vds_samples = sum(vds.n_samples for vdses in self._vdses.values() for vds in vdses)
        info(
            'Running VDS combiner:\n'
            f'    VDS arguments: {self._num_vdses} datasets with {vds_samples} samples\n'
            f'    GVCF arguments: {len(self._gvcfs)} inputs/samples\n'
            f'    Branch factor: {self._branch_factor}\n'
            f'    GVCF merge batch size: {self._gvcf_batch_size}'
        )
        while not self.finished:
            self.save()
            self.step()
        self.save()
        info('Finished VDS combiner!')
        hl._set_flags(**{flagname: prev_flag_value})

    @staticmethod
    def load(path) -> 'VariantDatasetCombiner':
        """Load a :class:`.VariantDatasetCombiner` from `path`."""
        fs = hl.current_backend().fs
        with fs.open(path) as stream:
            combiner = json.load(stream, cls=Decoder)
            combiner._raise_if_output_exists()
            if combiner._save_path != path:
                warning(
                    'path/save_path mismatch in loaded VariantDatasetCombiner, using '
                    f'{path} as the new save_path for this combiner'
                )
                combiner._save_path = path
            return combiner

    def _raise_if_output_exists(self):
        fs = hl.current_backend().fs
        ref_success_path = os.path.join(VariantDataset._reference_path(self._output_path), '_SUCCESS')
        var_success_path = os.path.join(VariantDataset._variants_path(self._output_path), '_SUCCESS')
        if fs.exists(ref_success_path) and fs.exists(var_success_path):
            raise FatalError(
                f'combiner output already exists at {self._output_path}\n' 'move or delete it before continuing'
            )

    def to_dict(self) -> dict:
        """A serializable representation of this combiner."""
        intervals_typ = hl.tarray(hl.tinterval(hl.tlocus(self._reference_genome)))
        return {
            'name': self.__class__.__name__,
            'save_path': self._save_path,
            'output_path': self._output_path,
            'temp_path': self._temp_path,
            'reference_genome': str(self._reference_genome),
            'dataset_type': self._dataset_type,
            'gvcf_type': self._gvcf_type,
            'branch_factor': self._branch_factor,
            'target_records': self._target_records,
            'gvcf_batch_size': self._gvcf_batch_size,
            'gvcf_external_header': self._gvcf_external_header,  # put this here for humans
            'contig_recoding': self._contig_recoding,
            'gvcf_info_to_keep': None if self._gvcf_info_to_keep is None else list(self._gvcf_info_to_keep),
            'gvcf_reference_entry_fields_to_keep': None
            if self._gvcf_reference_entry_fields_to_keep is None
            else list(self._gvcf_reference_entry_fields_to_keep),
            'call_fields': self._call_fields,
            'vdses': [md for i in sorted(self._vdses, reverse=True) for md in self._vdses[i]],
            'gvcfs': self._gvcfs,
            'gvcf_sample_names': self._gvcf_sample_names,
            'gvcf_import_intervals': intervals_typ._convert_to_json(self._gvcf_import_intervals),
        }

    @property
    def _num_vdses(self):
        return sum(len(v) for v in self._vdses.values())

    def step(self):
        """Run one layer of combinations.

        :meth:`.run` effectively runs :meth:`.step` until all GVCFs and Variant Datasets have been
        combined.

        """
        if self.finished:
            return
        if self._gvcfs:
            self._step_gvcfs()
        else:
            self._step_vdses()
        if not self.finished:
            self._job_id += 1

    def _write_final(self, vds):
        vds.write(self._output_path)

        if VariantDataset.ref_block_max_length_field not in vds.reference_data.globals:
            info("VDS combiner: computing reference block max length...")
            hl.vds.store_ref_block_max_length(self._output_path)

    def _step_vdses(self):
        current_bin = original_bin = min(self._vdses)
        files_to_merge = self._vdses[current_bin][: self._branch_factor]
        if len(files_to_merge) == len(self._vdses[current_bin]):
            del self._vdses[current_bin]
        else:
            self._vdses[current_bin] = self._vdses[current_bin][self._branch_factor :]

        remaining = self._branch_factor - len(files_to_merge)
        while self._num_vdses > 0 and remaining > 0:
            current_bin = min(self._vdses)
            extra = self._vdses[current_bin][-remaining:]
            if len(extra) == len(self._vdses[current_bin]):
                del self._vdses[current_bin]
            else:
                self._vdses[current_bin] = self._vdses[current_bin][:-remaining]
            files_to_merge = extra + files_to_merge
            remaining = self._branch_factor - len(files_to_merge)

        new_n_samples = sum(f.n_samples for f in files_to_merge)
        info(f'VDS Combine (job {self._job_id}): merging {len(files_to_merge)} datasets with {new_n_samples} samples')

        temp_path = self._temp_out_path(f'vds-combine_job{self._job_id}')
        largest_vds = max(files_to_merge, key=lambda vds: vds.n_samples)
        vds = hl.vds.read_vds(
            largest_vds.path,
            _assert_reference_type=self._dataset_type.reference_type,
            _assert_variant_type=self._dataset_type.variant_type,
            _warn_no_ref_block_max_length=False,
        )

        interval_bin = floor(log(new_n_samples, self._branch_factor))
        intervals = self.__intervals_cache.get(interval_bin)

        if intervals is None:
            # we use the reference data since it generally has more rows than the variant data
            intervals, _ = calculate_new_intervals(
                vds.reference_data, self._target_records, os.path.join(temp_path, 'interval_checkpoint.ht')
            )
            self.__intervals_cache[interval_bin] = intervals

        paths = [f.path for f in files_to_merge]
        vdss = self._read_variant_datasets(paths, intervals)
        combined = combine_variant_datasets(vdss)

        if self.finished:
            self._write_final(combined)
            return

        new_path = os.path.join(temp_path, 'dataset.vds')
        combined.write(new_path, overwrite=True, _codec_spec=FAST_CODEC_SPEC)
        new_bin = floor(log(new_n_samples, self._branch_factor))
        # this ensures that we don't somehow stick a vds at the end of
        # the same bin, ending up with a weird ordering issue
        if new_bin <= original_bin:
            new_bin = original_bin + 1
        self._vdses[new_bin].append(VDSMetadata(path=new_path, n_samples=new_n_samples))

    def _step_gvcfs(self):
        step = self._branch_factor
        files_to_merge = self._gvcfs[: self._gvcf_batch_size * step]
        self._gvcfs = self._gvcfs[self._gvcf_batch_size * step :]

        info(
            f'GVCF combine (job {self._job_id}): merging {len(files_to_merge)} GVCFs into '
            f'{(len(files_to_merge) + step - 1) // step} datasets'
        )

        if self._gvcf_external_header is not None:
            sample_names = self._gvcf_sample_names[: self._gvcf_batch_size * step]
            self._gvcf_sample_names = self._gvcf_sample_names[self._gvcf_batch_size * step :]
        else:
            sample_names = None
        header_file = self._gvcf_external_header or files_to_merge[0]
        header_info = hl.eval(hl.get_vcf_header_info(header_file))
        merge_vds = []
        merge_n_samples = []

        intervals_literal = hl.literal([
            hl.Struct(contig=i.start.contig, start=i.start.position, end=i.end.position)
            for i in self._gvcf_import_intervals
        ])

        partition_interval_point_type = hl.tstruct(locus=hl.tlocus(self._reference_genome))
        partition_intervals = [
            hl.Interval(
                start=hl.Struct(locus=i.start),
                end=hl.Struct(locus=i.end),
                includes_start=i.includes_start,
                includes_end=i.includes_end,
                point_type=partition_interval_point_type,
            )
            for i in self._gvcf_import_intervals
        ]
        vcfs = files_to_merge
        if sample_names is None:
            vcfs_lit = hl.literal(vcfs)
            range_ht = hl.utils.range_table(len(vcfs), n_partitions=min(len(vcfs), 32))

            range_ht = range_ht.annotate(
                sample_id=hl.rbind(hl.get_vcf_header_info(vcfs_lit[range_ht.idx]), lambda header: header.sampleIDs[0])
            )

            sample_ids = range_ht.aggregate(hl.agg.collect(range_ht.sample_id))
        else:
            sample_ids = sample_names
        for start in range(0, len(vcfs), step):
            ids = sample_ids[start : start + step]
            merging = vcfs[start : start + step]

            reference_ht = hl.Table._generate(
                contexts=intervals_literal,
                partitions=partition_intervals,
                rowfn=lambda interval, globals: hl._zip_join_producers(
                    hl.enumerate(hl.literal(merging)),
                    lambda idx_and_path: make_reference_stream(
                        hl.import_gvcf_interval(
                            idx_and_path[1],
                            idx_and_path[0],
                            interval.contig,
                            interval.start,
                            interval.end,
                            header_info,
                            call_fields=self._call_fields,
                            array_elements_required=False,
                            reference_genome=self._reference_genome,
                            contig_recoding=self._contig_recoding,
                        ),
                        self._gvcf_reference_entry_fields_to_keep,
                    ),
                    ['locus'],
                    lambda k, v: k.annotate(data=v),
                ),
                globals=hl.struct(g=hl.literal(ids).map(lambda s: hl.struct(__cols=[hl.struct(s=s)]))),
            )
            reference_ht = combine_r(reference_ht, ref_block_max_len_field=None)  # compute max length at the end

            variant_ht = hl.Table._generate(
                contexts=intervals_literal,
                partitions=partition_intervals,
                rowfn=lambda interval, globals: hl._zip_join_producers(
                    hl.enumerate(hl.literal(merging)),
                    lambda idx_and_path: make_variant_stream(
                        hl.import_gvcf_interval(
                            idx_and_path[1],
                            idx_and_path[0],
                            interval.contig,
                            interval.start,
                            interval.end,
                            header_info,
                            call_fields=self._call_fields,
                            array_elements_required=False,
                            reference_genome=self._reference_genome,
                            contig_recoding=self._contig_recoding,
                        ),
                        self._gvcf_info_to_keep,
                    ),
                    ['locus'],
                    lambda k, v: k.annotate(data=v),
                ),
                globals=hl.struct(g=hl.literal(ids).map(lambda s: hl.struct(__cols=[hl.struct(s=s)]))),
            )
            variant_ht = combine(variant_ht)
            vds = VariantDataset(
                reference_ht._unlocalize_entries('__entries', '__cols', ['s']),
                variant_ht._unlocalize_entries('__entries', '__cols', ['s'])._key_rows_by_assert_sorted(
                    'locus', 'alleles'
                ),
            )

            merge_vds.append(vds)
            merge_n_samples.append(len(merging))
        if self.finished and len(merge_vds) == 1:
            self._write_final(merge_vds[0])
            return

        temp_path = self._temp_out_path(f'gvcf-combine_job{self._job_id}/dataset_')
        pad = len(str(len(merge_vds) - 1))
        merge_metadata = [
            VDSMetadata(path=temp_path + str(count).rjust(pad, '0') + '.vds', n_samples=n_samples)
            for count, n_samples in enumerate(merge_n_samples)
        ]
        paths = [md.path for md in merge_metadata]
        hl.vds.write_variant_datasets(merge_vds, paths, overwrite=True, codec_spec=FAST_CODEC_SPEC)
        for md in merge_metadata:
            self._vdses[max(1, floor(log(md.n_samples, self._branch_factor)))].append(md)

    def _temp_out_path(self, extra):
        return os.path.join(self._temp_path, 'combiner-intermediates', f'{self._uuid}_{extra}')

    def _read_variant_datasets(self, inputs: List[str], intervals: List[Interval]):
        reference_type = self._dataset_type.reference_type
        variant_type = self._dataset_type.variant_type
        return [
            hl.vds.read_vds(
                path,
                intervals=intervals,
                _assert_reference_type=reference_type,
                _assert_variant_type=variant_type,
                _warn_no_ref_block_max_length=False,
            )
            for path in inputs
        ]


def new_combiner(
    *,
    output_path: str,
    temp_path: str,
    save_path: Optional[str] = None,
    gvcf_paths: Optional[List[str]] = None,
    vds_paths: Optional[List[str]] = None,
    vds_sample_counts: Optional[List[int]] = None,
    intervals: Optional[List[Interval]] = None,
    import_interval_size: Optional[int] = None,
    use_genome_default_intervals: bool = False,
    use_exome_default_intervals: bool = False,
    gvcf_external_header: Optional[str] = None,
    gvcf_sample_names: Optional[List[str]] = None,
    gvcf_info_to_keep: Optional[Collection[str]] = None,
    gvcf_reference_entry_fields_to_keep: Optional[Collection[str]] = None,
    call_fields: Collection[str] = ['PGT'],
    branch_factor: int = VariantDatasetCombiner._default_branch_factor,
    target_records: int = VariantDatasetCombiner._default_target_records,
    gvcf_batch_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    reference_genome: Union[str, hl.ReferenceGenome] = 'default',
    contig_recoding: Optional[Dict[str, str]] = None,
    force: bool = False,
) -> VariantDatasetCombiner:
    """Create a new :class:`.VariantDatasetCombiner` or load one from `save_path`."""
    if not (gvcf_paths or vds_paths):
        raise ValueError("at least one  of 'gvcf_paths' or 'vds_paths' must be nonempty")
    if gvcf_paths is None:
        gvcf_paths = []
    if len(gvcf_paths) > 0:
        if len(set(gvcf_paths)) != len(gvcf_paths):
            duplicates = [gvcf for gvcf, count in collections.Counter(gvcf_paths).items() if count > 1]
            duplicates = '\n    '.join(duplicates)
            raise ValueError(f'gvcf paths should be unique, the following paths are repeated:{duplicates}')
        if gvcf_sample_names is not None and len(set(gvcf_sample_names)) != len(gvcf_sample_names):
            duplicates = [gvcf for gvcf, count in collections.Counter(gvcf_sample_names).items() if count > 1]
            duplicates = '\n    '.join(duplicates)
            raise ValueError(
                "provided sample names ('gvcf_sample_names') should be unique, "
                f'the following names are repeated:{duplicates}'
            )

    if vds_paths is None:
        vds_paths = []
    if vds_sample_counts is not None and len(vds_paths) != len(vds_sample_counts):
        raise ValueError(
            "'vds_paths' and 'vds_sample_counts' (if present) must have the same length "
            f'{len(vds_paths)} != {len(vds_sample_counts)}'
        )
    if (gvcf_sample_names is None) != (gvcf_external_header is None):
        raise ValueError("both 'gvcf_sample_names' and 'gvcf_external_header' must be set or unset")
    if gvcf_sample_names is not None and len(gvcf_sample_names) != len(gvcf_paths):
        raise ValueError(
            "'gvcf_sample_names' and 'gvcf_paths' must have the same length "
            f'{len(gvcf_sample_names)} != {len(gvcf_paths)}'
        )

    if batch_size is None:
        if gvcf_batch_size is None:
            gvcf_batch_size = VariantDatasetCombiner._default_gvcf_batch_size
        else:
            pass
    else:
        if gvcf_batch_size is None:
            warning(
                'The batch_size parameter is deprecated. '
                'The batch_size parameter will be removed in a future version of Hail. '
                'Please use gvcf_batch_size instead.'
            )
            gvcf_batch_size = batch_size
        else:
            raise ValueError(
                'Specify only one of batch_size and gvcf_batch_size. ' f'Received {batch_size} and {gvcf_batch_size}.'
            )
    del batch_size

    def maybe_load_from_saved_path(save_path: str) -> Optional[VariantDatasetCombiner]:
        if force:
            return None
        fs = hl.current_backend().fs
        if fs.exists(save_path):
            try:
                combiner = load_combiner(save_path)
                warning(f'found existing combiner plan at {save_path}, using it')
                # we overwrite these values as they are serialized, but not part of the
                # hash for an autogenerated name and we want users to be able to overwrite
                # these when resuming a combine (a common reason to need to resume a combine
                # is a failure due to branch factor being too large)
                combiner._branch_factor = branch_factor
                combiner._target_records = target_records
                combiner._gvcf_batch_size = gvcf_batch_size
                return combiner
            except (ValueError, TypeError, OSError, KeyError) as e:
                warning(
                    f'file exists at {save_path}, but it is not a valid combiner plan, overwriting\n'
                    f'    caused by: {e}'
                )
        return None

    # We do the first save_path check now after validating the arguments
    if save_path is not None:
        saved_combiner = maybe_load_from_saved_path(save_path)
        if saved_combiner is not None:
            return saved_combiner

    if len(gvcf_paths) > 0:
        n_partition_args = (
            int(intervals is not None)
            + int(import_interval_size is not None)
            + int(use_genome_default_intervals)
            + int(use_exome_default_intervals)
        )

        if n_partition_args == 0:
            raise ValueError(
                "'new_combiner': require one argument from 'intervals', 'import_interval_size', "
                "'use_genome_default_intervals', or 'use_exome_default_intervals' to choose GVCF partitioning"
            )

        if n_partition_args > 1:
            warning(
                "'new_combiner': multiple colliding arguments found from 'intervals', 'import_interval_size', "
                "'use_genome_default_intervals', or 'use_exome_default_intervals'."
                "\n  The argument found first in the list in this warning will be used, and others ignored."
            )

        if intervals is not None:
            pass
        elif import_interval_size is not None:
            intervals = calculate_even_genome_partitioning(reference_genome, import_interval_size)
        elif use_genome_default_intervals:
            size = VariantDatasetCombiner.default_genome_interval_size
            intervals = calculate_even_genome_partitioning(reference_genome, size)
        elif use_exome_default_intervals:
            size = VariantDatasetCombiner.default_exome_interval_size
            intervals = calculate_even_genome_partitioning(reference_genome, size)
        assert intervals is not None
    else:
        intervals = []

    if isinstance(reference_genome, str):
        reference_genome = hl.get_reference(reference_genome)

    # we need to compute the type that the combiner will have, this will allow us to read matrix
    # tables quickly, especially in an asynchronous environment like query on batch where typing
    # a read uses a blocking round trip.
    vds = None
    gvcf_type = None
    if vds_paths:
        # sync up gvcf_reference_entry_fields_to_keep and they reference entry types from the VDS
        vds = hl.vds.read_vds(vds_paths[0], _warn_no_ref_block_max_length=False)
        vds_ref_entry = set(vds.reference_data.entry) - {'END'}
        if gvcf_reference_entry_fields_to_keep is not None and vds_ref_entry != gvcf_reference_entry_fields_to_keep:
            warning(
                "Mismatch between 'gvcf_reference_entry_fields' to keep and VDS reference data "
                "entry types. Overwriting with reference entry fields from supplied VDS.\n"
                f"    VDS reference entry fields      : {sorted(vds_ref_entry)}\n"
                f"    requested reference entry fields: {sorted(gvcf_reference_entry_fields_to_keep)}"
            )
        gvcf_reference_entry_fields_to_keep = vds_ref_entry

        # sync up call_fields and call fields present in the VDS
        all_entry_types = chain(vds.reference_data._type.entry_type.items(), vds.variant_data._type.entry_type.items())
        vds_call_fields = {name for name, typ in all_entry_types if typ == hl.tcall} - {'LGT', 'GT'}
        if 'LPGT' in vds_call_fields:
            vds_call_fields = (vds_call_fields - {'LPGT'}) | {'PGT'}
        if set(call_fields) != vds_call_fields:
            warning(
                "Mismatch between 'call_fields' and VDS call fields. "
                "Overwriting with call fields from supplied VDS.\n"
                f"    VDS call fields      : {sorted(vds_call_fields)}\n"
                f"    requested call fields: {sorted(call_fields)}\n"
            )
        call_fields = vds_call_fields

    if gvcf_paths:
        mt = hl.import_vcf(
            gvcf_paths[0],
            header_file=gvcf_external_header,
            force_bgz=True,
            array_elements_required=False,
            reference_genome=reference_genome,
            contig_recoding=contig_recoding,
        )
        gvcf_type = mt._type
        if gvcf_reference_entry_fields_to_keep is None:
            rmt = mt.filter_rows(hl.is_defined(mt.info.END))
            gvcf_reference_entry_fields_to_keep = defined_entry_fields(rmt, 100_000) - {'GT', 'PGT', 'PL'}
        if vds is None:
            vds = transform_gvcf(
                mt._key_rows_by_assert_sorted('locus'), gvcf_reference_entry_fields_to_keep, gvcf_info_to_keep
            )
    dataset_type = CombinerOutType(reference_type=vds.reference_data._type, variant_type=vds.variant_data._type)

    if save_path is None:
        sha = hashlib.sha256()
        sha.update(output_path.encode())
        sha.update(temp_path.encode())
        sha.update(str(reference_genome).encode())
        sha.update(str(dataset_type).encode())
        if gvcf_type is not None:
            sha.update(str(gvcf_type).encode())
        for path in vds_paths:
            sha.update(path.encode())
        for path in gvcf_paths:
            sha.update(path.encode())
        if gvcf_external_header is not None:
            sha.update(gvcf_external_header.encode())
        if gvcf_sample_names is not None:
            for name in gvcf_sample_names:
                sha.update(name.encode())
        if gvcf_info_to_keep is not None:
            for kept_info in sorted(gvcf_info_to_keep):
                sha.update(kept_info.encode())
        if gvcf_reference_entry_fields_to_keep is not None:
            for field in sorted(gvcf_reference_entry_fields_to_keep):
                sha.update(field.encode())
        for call_field in sorted(call_fields):
            sha.update(call_field.encode())
        if contig_recoding is not None:
            for key, value in sorted(contig_recoding.items()):
                sha.update(key.encode())
                sha.update(value.encode())
        for interval in intervals:
            sha.update(str(interval).encode())
        digest = sha.hexdigest()
        name = f'vds-combiner-plan_{digest}_{hl.__pip_version__}.json'
        save_path = os.path.join(temp_path, 'combiner-plans', name)
        saved_combiner = maybe_load_from_saved_path(save_path)
        if saved_combiner is not None:
            return saved_combiner
        warning(f'generated combiner save path of {save_path}')

    if vds_sample_counts:
        vdses = [VDSMetadata(path, n_samples) for path, n_samples in zip(vds_paths, vds_sample_counts)]
    else:
        vdses = []
        for path in vds_paths:
            vds = hl.vds.read_vds(
                path,
                _assert_reference_type=dataset_type.reference_type,
                _assert_variant_type=dataset_type.variant_type,
                _warn_no_ref_block_max_length=False,
            )
            n_samples = vds.n_samples()
            vdses.append(VDSMetadata(path, n_samples))

    vdses.sort(key=lambda x: x.n_samples, reverse=True)

    combiner = VariantDatasetCombiner(
        save_path=save_path,
        output_path=output_path,
        temp_path=temp_path,
        reference_genome=reference_genome,
        dataset_type=dataset_type,
        branch_factor=branch_factor,
        target_records=target_records,
        gvcf_batch_size=gvcf_batch_size,
        contig_recoding=contig_recoding,
        call_fields=call_fields,
        vdses=vdses,
        gvcfs=gvcf_paths,
        gvcf_import_intervals=intervals,
        gvcf_external_header=gvcf_external_header,
        gvcf_sample_names=gvcf_sample_names,
        gvcf_info_to_keep=gvcf_info_to_keep,
        gvcf_reference_entry_fields_to_keep=gvcf_reference_entry_fields_to_keep,
    )
    combiner._raise_if_output_exists()
    return combiner


def load_combiner(path: str) -> VariantDatasetCombiner:
    """Load a :class:`.VariantDatasetCombiner` from `path`."""
    return VariantDatasetCombiner.load(path)


class Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, VariantDatasetCombiner):
            return o.to_dict()
        if isinstance(o, HailType):
            return str(o)
        if isinstance(o, tmatrix):
            return o.to_dict()
        return json.JSONEncoder.default(self, o)


class Decoder(json.JSONDecoder):
    def __init__(self, **kwargs):
        super().__init__(object_hook=Decoder._object_hook, **kwargs)

    @staticmethod
    def _object_hook(obj):
        if 'name' not in obj:
            return obj
        name = obj['name']
        if name == VariantDatasetCombiner.__name__:
            del obj['name']
            obj['vdses'] = [VDSMetadata(*x) for x in obj['vdses']]
            obj['dataset_type'] = CombinerOutType(*(tmatrix._from_json(ty) for ty in obj['dataset_type']))
            if 'gvcf_type' in obj and obj['gvcf_type']:
                obj['gvcf_type'] = tmatrix._from_json(obj['gvcf_type'])

            rg = hl.get_reference(obj['reference_genome'])
            obj['reference_genome'] = rg
            intervals_type = hl.tarray(hl.tinterval(hl.tlocus(rg)))
            intervals = intervals_type._convert_from_json(obj['gvcf_import_intervals'])
            obj['gvcf_import_intervals'] = intervals

            return VariantDatasetCombiner(**obj)
        return obj
