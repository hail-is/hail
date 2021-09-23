import collections
import hashlib
import json
import os

from typing import Dict, List, Optional, Union

import hail as hl

from hail.utils.java import warning
from hail.experimental.vcf_combiner.vcf_combiner import calculate_even_genome_partitioning
from .. import VariantDataset


VDSMetadata = collections.namedtuple('VDSMetadata', ['path', 'n_samples'])


class VariantDatasetCombiner:  # pylint: disable=too-many-instance-attributes
    default_gvcf_batch_size = 100
    default_branch_factor = 100
    default_target_records = 30_000

    # These are used to calculate intervals for reading GVCFs in the combiner
    # The genome interval size results in 2568 partions for GRCh38. The exome
    # interval size assumes that they are around 2% the size of a genome and
    # result in 65 partitions for GRCh38.
    default_genome_interval_size = 1_200_000
    default_exome_interval_size = 60_000_000

    # TODO: need something for external header and sample names
    __serialized_slots__ = [
        'save_path',
        'output_path',
        'scratch_path',
        'reference_genome',
        'branch_factor',
        'target_records',
        'gvcf_batch_size',
        'contig_recoding',
        'vdses',
        'gvcfs',
        'gvcf_external_header',
        'gvcf_sample_names',
        'gvcf_import_intervals',
    ]

    __slots__ = tuple(__serialized_slots__ + ['__vds_cache'])

    def __init__(self,
                 *,
                 save_path: str,
                 output_path: str,
                 scratch_path: str,
                 reference_genome: hl.ReferenceGenome,
                 branch_factor: int = default_branch_factor,
                 target_records: int = default_target_records,
                 gvcf_batch_size: int = default_gvcf_batch_size,
                 contig_recoding: Optional[Dict[str, str]] = None,
                 vdses: List[VDSMetadata],
                 gvcfs: List[str],
                 gvcf_sample_names: Optional[List[str]] = None,
                 gvcf_external_header: Optional[str] = None,
                 gvcf_import_intervals: List[hl.utils.Interval],
                 vds_cache: Optional[Dict[str, VariantDataset]] = None,
                 ):
        if not gvcf_import_intervals:
            raise ValueError('gvcf import intervals must be nonempty')
        interval = gvcf_import_intervals[0]
        if not isinstance(interval.point_type, hl.tlocus):
            raise ValueError(f'intervals point type must be a locus, found {interval.point_type}')
        if interval.point_type.reference_genome != reference_genome:
            raise ValueError(f'mismatch in intervals ({interval.point_type.reference_genome}) '
                             f'and reference genome ({reference_genome}) types')
        if (gvcf_sample_names is None) != (gvcf_external_header is None):
            raise ValueError("both 'gvcf_sample_names' and 'gvcf_external_header' must be set or unset")
        if gvcf_sample_names is not None and len(gvcf_sample_names) != len(gvcfs):
            raise ValueError("'gvcf_sample_names' and 'gvcfs' must have the same length "
                             f'{len(gvcf_sample_names)} != {len(gvcfs)}')
        self.save_path = save_path
        self.output_path = output_path
        self.scratch_path = scratch_path
        self.reference_genome = reference_genome
        self.branch_factor = branch_factor
        self.target_records = target_records
        self.gvcf_batch_size = gvcf_batch_size
        self.contig_recoding = contig_recoding
        self.vdses = vdses
        self.gvcfs = gvcfs
        self.gvcf_sample_names = gvcf_sample_names
        self.gvcf_external_header = gvcf_external_header
        self.gvcf_import_intervals = gvcf_import_intervals
        self.__vds_cache = vds_cache

    def __eq__(self, other):
        if other.__class__ != VariantDatasetCombiner:
            return False
        for slot in self.__serialized_slots__:
            if getattr(self, slot) != getattr(other, slot):
                return False
        return True

    def save(self):
        fs = hl.current_backend().fs
        with fs.open(self.save_path, 'w') as out:
            json.dump(self, out, indent=2, cls=Encoder)

    @staticmethod
    def load(path) -> 'VariantDatasetCombiner':
        fs = hl.current_backend().fs
        with fs.open(path) as stream:
            combiner = json.load(stream, cls=Decoder)
            if combiner.save_path != path:
                warning('path/save_path mismatch in loaded VariantDatasetCombiner, using '
                        f'{path} as the new save_path for this combiner')
                combiner.save_path = path
            return combiner

    @property
    def finished(self) -> bool:
        return not self.gvcfs and not self.vdses

    def to_dict(self) -> dict:
        intervals_typ = hl.tarray(hl.tinterval(hl.tlocus(self.reference_genome)))
        return {'name': self.__class__.__name__,
                'save_path': self.save_path,
                'output_path': self.output_path,
                'scratch_path': self.scratch_path,
                'reference_genome': str(self.reference_genome),
                'branch_factor': self.branch_factor,
                'target_records': self.target_records,
                'gvcf_batch_size': self.gvcf_batch_size,
                'gvcf_external_header': self.gvcf_external_header,  # put this here for humans
                'contig_recoding': self.contig_recoding,
                'vdses': self.vdses,
                'gvcfs': self.gvcfs,
                'gvcf_sample_names': self.gvcf_sample_names,
                'gvcf_import_intervals': intervals_typ._convert_to_json(self.gvcf_import_intervals),
                }


def new_combiner(*,
                 output_path: str,
                 scratch_path: str,
                 save_path: Optional[str] = None,
                 gvcf_paths: Optional[List[str]] = None,
                 vds_paths: Optional[List[str]] = None,
                 vds_sample_counts: Optional[List[int]] = None,
                 intervals: Optional[List[hl.utils.Interval]] = None,
                 import_interval_size: Optional[int] = None,
                 use_genome_default_intervals: bool = False,
                 use_exome_default_intervals: bool = False,
                 header: Optional[str] = None,
                 sample_names: Optional[List[str]] = None,
                 branch_factor: int = VariantDatasetCombiner.default_branch_factor,
                 target_records: int = VariantDatasetCombiner.default_target_records,
                 batch_size: int = VariantDatasetCombiner.default_gvcf_batch_size,
                 reference_genome: Union[str, hl.ReferenceGenome] = 'default',
                 contig_recoding: Optional[Dict[str, str]] = None,
                 force: bool = False,
                 ) -> VariantDatasetCombiner:
    if not (gvcf_paths or vds_paths):
        raise ValueError("at least one  of 'gvcf_paths' or 'vds_paths' must be not empty")
    if gvcf_paths is None:
        gvcf_paths = []
    if vds_paths is None:
        vds_paths = []
    if vds_sample_counts is not None:
        assert len(vds_paths) == len(vds_sample_counts)

    n_partition_args = (int(intervals is not None)
                        + int(import_interval_size is not None)
                        + int(use_genome_default_intervals)
                        + int(use_exome_default_intervals))

    if n_partition_args == 0:
        raise ValueError("'new_combiner': require one argument from 'intervals', 'import_interval_size', "
                         "'use_genome_default_intervals', or 'use_exome_default_intervals' to choose GVCF partitioning")

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
                combiner.branch_factor = branch_factor
                combiner.target_records = target_records
                combiner.gvcf_batch_size = batch_size
                return combiner
            except (ValueError, TypeError):
                warning(f'file exists at {save_path}, but it is not a valid combiner plan, overwriting')
        return None

    # We do the first save_path check now after validating the arguments
    if save_path is not None:
        saved_combiner = maybe_load_from_saved_path(save_path)
        if saved_combiner is not None:
            return saved_combiner

    if n_partition_args > 1:
        warning("'run_combiner': multiple colliding arguments found from 'intervals', 'import_interval_size', "
                "'use_genome_default_intervals', or 'use_exome_default_intervals'."
                "\n  The argument found first in the list in this warning will be used, and others ignored.")

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

    if isinstance(reference_genome, str):
        reference_genome = hl.get_reference(reference_genome)

    if save_path is None:
        sha = hashlib.sha256()
        sha.update(output_path.encode())
        sha.update(scratch_path.encode())
        sha.update(str(reference_genome).encode())
        for path in vds_paths:
            sha.update(path.encode())
        for path in gvcf_paths:
            sha.update(path.encode())
        if header is not None:
            sha.update(header.encode())
        if sample_names is not None:
            for name in sample_names:
                sha.update(name.encode())
        if contig_recoding is not None:
            for key, value in sorted(contig_recoding.items()):
                sha.update(key.encode())
                sha.update(value.encode())
        for interval in intervals:
            sha.update(str(interval).encode())
        digest = sha.hexdigest()
        name = f'vds-combiner-plan_{digest}_{hl.__pip_version__}.json'
        save_path = os.path.join(scratch_path, 'combiner-plans', name)
        saved_combiner = maybe_load_from_saved_path(save_path)
        if saved_combiner is not None:
            return saved_combiner

    if vds_sample_counts:
        cache = None
        vdses = [VDSMetadata(path, n_samples) for path, n_samples in zip(vds_paths, vds_sample_counts)]
    else:
        vdses = []
        cache = {}
        for path in vds_paths:
            vds = hl.vds.read_vds(path)
            n_samples = vds.n_samples()
            cache[path] = vds
            vdses.append(VDSMetadata(path, n_samples))

    return VariantDatasetCombiner(save_path=save_path,
                                  output_path=output_path,
                                  scratch_path=scratch_path,
                                  reference_genome=reference_genome,
                                  branch_factor=branch_factor,
                                  target_records=target_records,
                                  gvcf_batch_size=batch_size,
                                  contig_recoding=contig_recoding,
                                  vdses=vdses,
                                  gvcfs=gvcf_paths,
                                  gvcf_import_intervals=intervals,
                                  gvcf_external_header=header,
                                  gvcf_sample_names=sample_names,
                                  vds_cache=cache)


def load_combiner(path: str) -> VariantDatasetCombiner:
    return VariantDatasetCombiner.load(path)


class Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, VariantDatasetCombiner):
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

            rg = hl.get_reference(obj['reference_genome'])
            obj['reference_genome'] = rg
            intervals_type = hl.tarray(hl.tinterval(hl.tlocus(rg)))
            intervals = intervals_type._convert_from_json(obj['gvcf_import_intervals'])
            obj['gvcf_import_intervals'] = intervals

            return VariantDatasetCombiner(**obj)
        return obj
