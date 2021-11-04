import collections
import hashlib
import itertools
import json
import os
import sys
import uuid

from math import floor, log
from typing import Collection, Dict, List, Optional, Union

import hail as hl

from hail.utils import Interval
from hail.utils.java import Env, info, warning
from hail.experimental.vcf_combiner.vcf_combiner import calculate_even_genome_partitioning, \
    calculate_new_intervals
from hail.vds.variant_dataset import VariantDataset
from .combine import combine_variant_datasets, transform_gvcf, defined_entry_fields


VDSMetadata = collections.namedtuple('VDSMetadata', ['path', 'n_samples'])

FAST_CODEC_SPEC = """{
  "name": "LEB128BufferSpec",
  "child": {
    "name": "BlockingBufferSpec",
    "blockSize": 32768,
    "child": {
      "name": "LZ4FastBlockBufferSpec",
      "blockSize": 32768,
      "child": {
        "name": "StreamBlockBufferSpec"
      }
    }
  }
}"""


def read_variant_datasets(inputs: List[str], intervals: List[Interval], intervals_dtype):
    n_inputs = len(inputs)
    paths = list(itertools.chain(
        (VariantDataset._reference_path(path) for path in inputs),
        (VariantDataset._variants_path(path) for path in inputs)))
    mts = Env.spark_backend("read_variant_datasets").read_multiple_matrix_tables(
        paths, intervals, intervals_dtype)
    vdss = [VariantDataset(reference_data=ref, variant_data=var)
            for ref, var in zip(mts[:n_inputs], mts[n_inputs:])]
    return vdss


class VariantDatasetCombiner:  # pylint: disable=too-many-instance-attributes
    default_gvcf_batch_size = 100
    default_branch_factor = 100
    default_target_records = 24_000
    gvcf_merge_task_limit = 150_000

    # These are used to calculate intervals for reading GVCFs in the combiner
    # The genome interval size results in 2568 partitions for GRCh38. The exome
    # interval size assumes that they are around 2% the size of a genome and
    # result in 65 partitions for GRCh38.
    default_genome_interval_size = 1_200_000
    default_exome_interval_size = 60_000_000

    __serialized_slots__ = [
        'save_path',
        'output_path',
        'temp_path',
        'reference_genome',
        'branch_factor',
        'target_records',
        '_gvcf_batch_size',
        'contig_recoding',
        'vdses',
        'gvcfs',
        'gvcf_external_header',
        'gvcf_sample_names',
        'gvcf_import_intervals',
        'gvcf_info_to_keep',
        'gvcf_reference_entry_fields_to_keep',
    ]

    __slots__ = tuple(__serialized_slots__ + ['_uuid', '_job_id', '__intervals_cache'])

    def __init__(self,
                 *,
                 save_path: str,
                 output_path: str,
                 temp_path: str,
                 reference_genome: hl.ReferenceGenome,
                 branch_factor: int = default_branch_factor,
                 target_records: int = default_target_records,
                 gvcf_batch_size: int = default_gvcf_batch_size,
                 contig_recoding: Optional[Dict[str, str]] = None,
                 vdses: List[VDSMetadata],
                 gvcfs: List[str],
                 gvcf_sample_names: Optional[List[str]] = None,
                 gvcf_external_header: Optional[str] = None,
                 gvcf_import_intervals: List[Interval],
                 gvcf_info_to_keep: Optional[Collection[str]] = None,
                 gvcf_reference_entry_fields_to_keep: Optional[Collection[str]] = None,
                 ):
        if not (vdses or gvcfs):
            raise ValueError("one of 'vdses' or 'gvcfs' must be nonempty")
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
        if branch_factor < 2:
            raise ValueError(f"'branch_factor' must be at least 2, found {branch_factor}")
        if gvcf_batch_size < 1:
            raise ValueError(f"'gvcf_batch_size' must be at least 1, found {gvcf_batch_size}")

        self.save_path = save_path
        self.output_path = output_path
        self.temp_path = temp_path
        self.reference_genome = reference_genome
        self.branch_factor = branch_factor
        self.target_records = target_records
        self.contig_recoding = contig_recoding
        self.vdses = collections.defaultdict(list)
        for vds in vdses:
            self.vdses[max(1, floor(log(vds.n_samples, self.branch_factor)))].append(vds)
        self.gvcfs = gvcfs
        self.gvcf_sample_names = gvcf_sample_names
        self.gvcf_external_header = gvcf_external_header
        self.gvcf_import_intervals = gvcf_import_intervals
        self.gvcf_info_to_keep = set(gvcf_info_to_keep) if gvcf_info_to_keep is not None \
            else None
        self.gvcf_reference_entry_fields_to_keep = set(gvcf_reference_entry_fields_to_keep) \
            if gvcf_reference_entry_fields_to_keep is not None else None

        self._uuid = uuid.uuid4()
        self._job_id = 1
        self.__intervals_cache = {}
        self.gvcf_batch_size = gvcf_batch_size

    @property
    def gvcf_batch_size(self):
        return self._gvcf_batch_size

    @gvcf_batch_size.setter
    def gvcf_batch_size(self, value: int):
        if value * len(self.gvcf_import_intervals) > VariantDatasetCombiner.gvcf_merge_task_limit:
            old_value = value
            value = VariantDatasetCombiner.gvcf_merge_task_limit // len(self.gvcf_import_intervals)
            warning(f'gvcf_batch_size of {old_value} would produce too many tasks '
                    f'using {value} instead')
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
        return not self.gvcfs and not self.vdses

    def save(self):
        fs = hl.current_backend().fs
        try:
            backup_path = self.save_path + '.bak'
            if fs.exists(self.save_path):
                fs.copy(self.save_path, backup_path)
            with fs.open(self.save_path, 'w') as out:
                json.dump(self, out, indent=2, cls=Encoder)
            if fs.exists(backup_path):
                fs.remove(backup_path)
        except OSError as e:
            # these messages get printed, because there is absolutely no guarantee
            # that the hail context is in a sane state if any of the above operations
            # fail
            print(f'Failed saving {self.__class__.__name__} state at {self.save_path}')
            print(f'An attempt was made to copy {self.save_path} to {backup_path}')
            print('An old version of this state may be there.')
            print('Dumping current state as json to standard output, you may wish '
                  'to save this output in order to resume the combiner.')
            json.dump(self, sys.stdout, indent=2, cls=Encoder)
            print()
            raise e

    def run(self):
        flagname = 'no_ir_logging'
        prev_flag_value = hl._get_flags(flagname).get(flagname)
        hl._set_flags(**{flagname: '1'})

        vds_samples = sum(vds.n_samples for vdses in self.vdses.values() for vds in vdses)
        info('Running VDS combiner:\n'
             f'    VDS arguments: {self._num_vdses} datasets with {vds_samples} samples\n'
             f'    GVCF arguments: {len(self.gvcfs)} inputs/samples\n'
             f'    Branch factor: {self.branch_factor}\n'
             f'    GVCF merge batch size: {self.gvcf_batch_size}')
        while not self.finished:
            self.save()
            self.step()
        self.save()
        info('Finished VDS combiner!')
        hl._set_flags(**{flagname: prev_flag_value})

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

    def to_dict(self) -> dict:
        intervals_typ = hl.tarray(hl.tinterval(hl.tlocus(self.reference_genome)))
        return {'name': self.__class__.__name__,
                'save_path': self.save_path,
                'output_path': self.output_path,
                'temp_path': self.temp_path,
                'reference_genome': str(self.reference_genome),
                'branch_factor': self.branch_factor,
                'target_records': self.target_records,
                'gvcf_batch_size': self.gvcf_batch_size,
                'gvcf_external_header': self.gvcf_external_header,  # put this here for humans
                'contig_recoding': self.contig_recoding,
                'gvcf_info_to_keep': None if self.gvcf_info_to_keep is None
                else list(self.gvcf_info_to_keep),
                'gvcf_reference_entry_fields_to_keep': None
                if self.gvcf_reference_entry_fields_to_keep is None
                else list(self.gvcf_reference_entry_fields_to_keep),
                'vdses': [md for i in sorted(self.vdses, reverse=True) for md in self.vdses[i]],
                'gvcfs': self.gvcfs,
                'gvcf_sample_names': self.gvcf_sample_names,
                'gvcf_import_intervals': intervals_typ._convert_to_json(self.gvcf_import_intervals),
                }

    @property
    def _num_vdses(self):
        return sum(len(v) for v in self.vdses.values())

    def step(self):
        if self.finished:
            return
        if self.gvcfs:
            self._step_gvcfs()
        else:
            self._step_vdses()
        if not self.finished:
            self._job_id += 1

    def _step_vdses(self):
        current_bin = original_bin = min(self.vdses)
        files_to_merge = self.vdses[current_bin][:self.branch_factor]
        if len(files_to_merge) == len(self.vdses[current_bin]):
            del self.vdses[current_bin]
        else:
            self.vdses[current_bin] = self.vdses[current_bin][self.branch_factor:]

        remaining = self.branch_factor - len(files_to_merge)
        while self._num_vdses > 0 and remaining > 0:
            current_bin = min(self.vdses)
            extra = self.vdses[current_bin][-remaining:]
            if len(extra) == len(self.vdses[current_bin]):
                del self.vdses[current_bin]
            else:
                self.vdses[current_bin] = self.vdses[current_bin][:-remaining]
            files_to_merge = extra + files_to_merge
            remaining = self.branch_factor - len(files_to_merge)

        new_n_samples = sum(f.n_samples for f in files_to_merge)
        info(f'VDS Combine (job {self._job_id}): merging {len(files_to_merge)} datasets with {new_n_samples} samples')

        temp_path = self._temp_out_path(f'vds-combine_job{self._job_id}')
        largest_vds = max(files_to_merge, key=lambda vds: vds.n_samples)
        vds = hl.vds.read_vds(largest_vds.path)

        interval_bin = floor(log(new_n_samples, self.branch_factor))
        intervals, intervals_dtype = self.__intervals_cache.get(interval_bin, (None, None))

        if intervals is None:
            # we use the reference data since it generally has more rows than the variant data
            intervals, intervals_dtype = calculate_new_intervals(vds.reference_data,
                                                                 self.target_records,
                                                                 os.path.join(temp_path, 'interval_checkpoint.ht'))
            self.__intervals_cache[interval_bin] = (intervals, intervals_dtype)

        paths = [f.path for f in files_to_merge]
        vdss = read_variant_datasets(paths, intervals, intervals_dtype)
        combined = combine_variant_datasets(vdss)

        if self.finished:
            combined.write(self.output_path)
            return

        new_path = os.path.join(temp_path, 'dataset.vds')
        combined.write(new_path, overwrite=True, _codec_spec=FAST_CODEC_SPEC)
        new_bin = floor(log(new_n_samples, self.branch_factor))
        # this ensures that we don't somehow stick a vds at the end of
        # the same bin, ending up with a weird ordering issue
        if new_bin <= original_bin:
            new_bin = original_bin + 1
        self.vdses[new_bin].append(VDSMetadata(path=new_path, n_samples=new_n_samples))

    def _step_gvcfs(self):
        step = self.branch_factor
        files_to_merge = self.gvcfs[:self.gvcf_batch_size * step]
        self.gvcfs = self.gvcfs[self.gvcf_batch_size * step:]

        info(f'GVCF combine (job {self._job_id}): merging {len(files_to_merge)} GVCFs into '
             f'{(len(files_to_merge) + step - 1) // step} datasets')

        if self.gvcf_external_header is not None:
            sample_names = self.gvcf_sample_names[:self.gvcf_batch_size * step]
            self.gvcf_sample_names = self.gvcf_sample_names[self.gvcf_batch_size * step:]
        else:
            sample_names = None
        merge_vds = []
        merge_n_samples = []
        vcfs = [transform_gvcf(vcf,
                               reference_entry_fields_to_keep=self.gvcf_reference_entry_fields_to_keep,
                               info_to_keep=self.gvcf_info_to_keep)
                for vcf in hl.import_gvcfs(files_to_merge,
                                           self.gvcf_import_intervals,
                                           array_elements_required=False,
                                           _external_header=self.gvcf_external_header,
                                           _external_sample_ids=[[name] for name in sample_names] if sample_names is not None else None,
                                           reference_genome=self.reference_genome,
                                           contig_recoding=self.contig_recoding)]
        while vcfs:
            merging, vcfs = vcfs[:step], vcfs[step:]
            merge_vds.append(combine_variant_datasets(merging))
            merge_n_samples.append(len(merging))
        if self.finished and len(merge_vds) == 1:
            merge_vds[0].write(self.output_path)
            return

        temp_path = self._temp_out_path(f'gvcf-combine_job{self._job_id}/dataset_')
        pad = len(str(len(merge_vds) - 1))
        merge_metadata = [VDSMetadata(path=temp_path + str(count).rjust(pad, '0') + '.vds',
                                      n_samples=n_samples)
                          for count, n_samples in enumerate(merge_n_samples)]
        paths = [md.path for md in merge_metadata]
        hl.vds.write_variant_datasets(merge_vds, paths, overwrite=True, codec_spec=FAST_CODEC_SPEC)
        for md in merge_metadata:
            self.vdses[max(1, floor(log(md.n_samples, self.branch_factor)))].append(md)

    def _temp_out_path(self, extra):
        return os.path.join(self.temp_path, 'combiner-intermediates', f'{self._uuid}_{extra}')


def new_combiner(*,
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
                 branch_factor: int = VariantDatasetCombiner.default_branch_factor,
                 target_records: int = VariantDatasetCombiner.default_target_records,
                 batch_size: int = VariantDatasetCombiner.default_gvcf_batch_size,
                 reference_genome: Union[str, hl.ReferenceGenome] = 'default',
                 contig_recoding: Optional[Dict[str, str]] = None,
                 force: bool = False,
                 ) -> VariantDatasetCombiner:
    if not (gvcf_paths or vds_paths):
        raise ValueError("at least one  of 'gvcf_paths' or 'vds_paths' must be nonempty")
    if gvcf_paths is None:
        gvcf_paths = []
    if vds_paths is None:
        vds_paths = []
    if vds_sample_counts is not None and len(vds_paths) != len(vds_sample_counts):
        raise ValueError("'vds_paths' and 'vds_sample_counts' (if present) must have the same length "
                         f'{len(vds_paths)} != {len(vds_sample_counts)}')
    if (gvcf_sample_names is None) != (gvcf_external_header is None):
        raise ValueError("both 'gvcf_sample_names' and 'gvcf_external_header' must be set or unset")
    if gvcf_sample_names is not None and len(gvcf_sample_names) != len(gvcf_paths):
        raise ValueError("'gvcf_sample_names' and 'gvcf_paths' must have the same length "
                         f'{len(gvcf_sample_names)} != {len(gvcf_paths)}')

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
            except (ValueError, TypeError, OSError, KeyError):
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

    if gvcf_reference_entry_fields_to_keep is None and vds_paths:
        vds = hl.vds.read_vds(vds_paths[0])
        gvcf_reference_entry_fields_to_keep = set(vds.reference_data.entry) - {'END'}
    elif gvcf_reference_entry_fields_to_keep is None and gvcf_paths:
        mt = hl.import_vcf(gvcf_paths[0], force_bgz=True, reference_genome=reference_genome)
        mt = mt.filter_rows(hl.is_defined(mt.info.END))
        gvcf_reference_entry_fields_to_keep = defined_entry_fields(mt, 100_000) - {'GT', 'PGT', 'PL'}

    if save_path is None:
        sha = hashlib.sha256()
        sha.update(output_path.encode())
        sha.update(temp_path.encode())
        sha.update(str(reference_genome).encode())
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
        else:
            warning(f'generated combiner save path of {save_path}')

    if vds_sample_counts:
        vdses = [VDSMetadata(path, n_samples) for path, n_samples in zip(vds_paths, vds_sample_counts)]
    else:
        vdses = []
        for path in vds_paths:
            vds = hl.vds.read_vds(path)
            n_samples = vds.n_samples()
            vdses.append(VDSMetadata(path, n_samples))

    vdses.sort(key=lambda x: x.n_samples, reverse=True)

    return VariantDatasetCombiner(save_path=save_path,
                                  output_path=output_path,
                                  temp_path=temp_path,
                                  reference_genome=reference_genome,
                                  branch_factor=branch_factor,
                                  target_records=target_records,
                                  gvcf_batch_size=batch_size,
                                  contig_recoding=contig_recoding,
                                  vdses=vdses,
                                  gvcfs=gvcf_paths,
                                  gvcf_import_intervals=intervals,
                                  gvcf_external_header=gvcf_external_header,
                                  gvcf_sample_names=gvcf_sample_names,
                                  gvcf_info_to_keep=gvcf_info_to_keep,
                                  gvcf_reference_entry_fields_to_keep=gvcf_reference_entry_fields_to_keep)


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
