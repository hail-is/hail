import abc

from typing import Optional, Union

from hailtop.aiotools.fs import AsyncFS
import hailtop.batch as hb
from hailtop.utils import AsyncWorkerPool, WaitableSharedPool

from .config import CheckpointConfigMixin


class ResourceGroupInputs:
    def __init__(self, name: str, **inputs: str):
        self.name = name
        self.inputs = inputs


class ResourceFileInput:
    def __init__(self, name: str, input: Optional[str]):
        self.name = name
        self.input = input


class BaseSaigeResource(abc.ABC):
    @classmethod
    async def _use_checkpoint_if_exists_and_requested(
        cls,
        fs: AsyncFS,
        pool: AsyncWorkerPool,
        b: hb.Batch,
        config: CheckpointConfigMixin,
        inputs: Union[ResourceFileInput, ResourceGroupInputs],
    ):
        if config.use_checkpoints:
            if isinstance(inputs, ResourceGroupInputs):
                input_files = inputs.inputs.values()
            else:
                assert isinstance(inputs, ResourceFileInput)
                input_files = [inputs.input]

            waitable_pool = WaitableSharedPool(pool)
            for input in input_files:
                await waitable_pool.call(fs.exists, input)
            await waitable_pool.wait()
            files_exist = waitable_pool.results()

            all_files_exist = all(files_exist)

            if all_files_exist:
                return cls._from_input_files(b, inputs)
        return None

    @classmethod
    def _from_input_files(cls, b: hb.Batch, inputs: Union[ResourceFileInput, ResourceGroupInputs]):
        if isinstance(inputs, ResourceFileInput):
            resource = b.read_input(inputs.input)
        else:
            assert isinstance(inputs, ResourceGroupInputs)
            resource = b.read_input_group(**inputs.inputs)
        return cls(resource)

    @classmethod
    def _from_job_intermediate(cls, j: hb.Job, inputs: Union[ResourceFileInput, ResourceGroupInputs]):
        if isinstance(inputs, ResourceFileInput):
            return cls(j[inputs.name])

        assert isinstance(inputs, ResourceGroupInputs)
        j.declare_resource_group({inputs.name: inputs.inputs})
        return cls(j[inputs.name])

    def __init__(self, resource: hb.Resource):
        self._resource = resource

    def checkpoint_if_requested(self, b: hb.Batch, config: CheckpointConfigMixin, output_file: str):
        if config.checkpoint_output:
            b.write_output(self._resource, output_file)

    def __getattr__(self, item):
        if isinstance(self._resource, hb.ResourceGroup):
            return self._resource[item]
        raise ValueError('cannot get attribute of a ResourceFile')

    def __getitem__(self, item):
        if isinstance(self._resource, hb.ResourceGroup):
            return self._resource[item]
        raise ValueError('cannot get attribute of a ResourceFile')


class PlinkResourceGroup(BaseSaigeResource):
    @staticmethod
    def inputs(root: str) -> ResourceGroupInputs:
        return ResourceGroupInputs('bfile', bed=f'{root}.bed', bim=f'{root}.bim', fam=f'{root}.fam')

    @staticmethod
    async def use_checkpoint_if_exists_and_requested(
        fs: AsyncFS, pool: AsyncWorkerPool, b: hb.Batch, config: CheckpointConfigMixin, bfile_root: str
    ) -> Optional['PlinkResourceGroup']:
        inputs = PlinkResourceGroup.inputs(bfile_root)
        return await PlinkResourceGroup._use_checkpoint_if_exists_and_requested(fs, pool, b, config, inputs)

    @staticmethod
    def from_input_files(b: hb.Batch, bfile_root: str) -> 'PlinkResourceGroup':
        inputs = PlinkResourceGroup.inputs(bfile_root)
        return PlinkResourceGroup._from_input_files(b, inputs)

    @staticmethod
    def from_job_intermediate(j: hb.Job) -> 'PlinkResourceGroup':
        inputs = PlinkResourceGroup.inputs('{root}')
        return PlinkResourceGroup._from_job_intermediate(j, inputs)


class VCFResourceGroup(BaseSaigeResource):
    @staticmethod
    def inputs(root: str) -> ResourceGroupInputs:
        return ResourceGroupInputs('vcf', vcf=f'{root}.vcf.gz', tbi=f'{root}.vcf.gz.tbi')

    @staticmethod
    async def use_checkpoint_if_exists_and_requested(
        fs: AsyncFS, pool: AsyncWorkerPool, b: hb.Batch, config: CheckpointConfigMixin, vcf_root: str
    ) -> Optional['VCFResourceGroup']:
        inputs = VCFResourceGroup.inputs(vcf_root)
        return await VCFResourceGroup._use_checkpoint_if_exists_and_requested(fs, pool, b, config, inputs)

    @staticmethod
    def from_input_files(b: hb.Batch, root: str) -> 'VCFResourceGroup':
        inputs = VCFResourceGroup.inputs(root)
        return VCFResourceGroup._from_input_files(b, inputs)

    @staticmethod
    def from_job_intermediate(j: hb.Job) -> 'VCFResourceGroup':
        inputs = VCFResourceGroup.inputs('{root}')
        return VCFResourceGroup._from_job_intermediate(j, inputs)


class BgenResourceGroup(BaseSaigeResource):
    @staticmethod
    def inputs(root: str) -> ResourceGroupInputs:
        return ResourceGroupInputs('bgen', bgen=f'{root}.bgen', bgi=f'{root}.bgen.bgi', sample=f'{root}.sample')

    @staticmethod
    async def use_checkpoint_if_exists_and_requested(
        fs: AsyncFS, pool: AsyncWorkerPool, b: hb.Batch, config: CheckpointConfigMixin, bgen_root: str
    ) -> Optional['BgenResourceGroup']:
        inputs = BgenResourceGroup.inputs(bgen_root)
        return await BgenResourceGroup._use_checkpoint_if_exists_and_requested(fs, pool, b, config, inputs)

    @staticmethod
    def from_input_files(b: hb.Batch, root: str) -> 'BgenResourceGroup':
        inputs = BgenResourceGroup.inputs(root)
        return BgenResourceGroup._from_input_files(b, inputs)

    @staticmethod
    def from_job_intermediate(j: hb.Job) -> 'BgenResourceGroup':
        inputs = BgenResourceGroup.inputs('{root}')
        return BgenResourceGroup._from_job_intermediate(j, inputs)


class SaigeSparseGRMResourceGroup(BaseSaigeResource):
    @staticmethod
    def inputs(root: str) -> ResourceGroupInputs:
        return ResourceGroupInputs('grm', grm=f'{root}.sparseGRM.mtx', sample_ids=f'{root}.sparseGRM.mtx.sampleIDs.txt')

    @staticmethod
    async def use_checkpoint_if_exists_and_requested(
        fs: AsyncFS, pool: AsyncWorkerPool, b: hb.Batch, config: CheckpointConfigMixin, grm_root: str
    ) -> Optional['SaigeSparseGRMResourceGroup']:
        inputs = SaigeSparseGRMResourceGroup.inputs(grm_root)
        return await SaigeSparseGRMResourceGroup._use_checkpoint_if_exists_and_requested(fs, pool, b, config, inputs)

    @staticmethod
    def from_input_files(b: hb.Batch, grm_root: str) -> 'SaigeSparseGRMResourceGroup':
        inputs = SaigeSparseGRMResourceGroup.inputs(grm_root)
        return SaigeSparseGRMResourceGroup._from_input_files(b, inputs)

    @staticmethod
    def from_job_intermediate(j: hb.Job, relatedness_cutoff: float, num_markers: int) -> 'SaigeSparseGRMResourceGroup':
        suffix = f'_relatednessCutoff_{relatedness_cutoff}_{num_markers}_randomMarkersUsed'
        inputs = SaigeSparseGRMResourceGroup.inputs(f'{{root}}{suffix}')
        return SaigeSparseGRMResourceGroup._from_job_intermediate(j, inputs)


class SaigeGeneGlmmResourceGroup(BaseSaigeResource):
    @staticmethod
    def inputs(glmm_root: str) -> ResourceGroupInputs:
        return ResourceGroupInputs(
            'glmm',
            mtx=f'{glmm_root}.sparseGRM.mtx',
            sample_ids=f'{glmm_root}.sparseGRM.mtx.sampleIDs.txt',
            rda=f'{glmm_root}.rda',
            results=f'{glmm_root}_30markers.SAIGE.results.txt',
            variance_ratio=f'{glmm_root}.gene.varianceRatio.txt',
        )

    @staticmethod
    async def use_checkpoint_if_exists_and_requested(
        fs: AsyncFS, pool: AsyncWorkerPool, b: hb.Batch, config: CheckpointConfigMixin, glmm_root: str
    ) -> Optional['SaigeGeneGlmmResourceGroup']:
        inputs = SaigeGeneGlmmResourceGroup.inputs(glmm_root)
        return await SaigeGeneGlmmResourceGroup._use_checkpoint_if_exists_and_requested(fs, pool, b, config, inputs)

    @staticmethod
    def from_input_files(b: hb.Batch, glmm_root: str) -> 'SaigeGeneGlmmResourceGroup':
        inputs = SaigeGeneGlmmResourceGroup.inputs(glmm_root)
        return SaigeGeneGlmmResourceGroup._from_input_files(b, inputs)

    @staticmethod
    def from_job_intermediate(j: hb.Job) -> 'SaigeGeneGlmmResourceGroup':
        inputs = SaigeGeneGlmmResourceGroup.inputs('{root}')
        return SaigeGeneGlmmResourceGroup._from_job_intermediate(j, inputs)


class SaigeGlmmResourceGroup(BaseSaigeResource):
    @staticmethod
    def inputs(glmm_root: str) -> ResourceGroupInputs:
        return ResourceGroupInputs(
            'glmm',
            rda=f'{glmm_root}.rda',
            results=f'{glmm_root}_30markers.SAIGE.results.txt',
            variance_ratio=f'{glmm_root}.gene.varianceRatio.txt',
        )

    @staticmethod
    async def use_checkpoint_if_exists_and_requested(
        fs: AsyncFS, pool: AsyncWorkerPool, b: hb.Batch, config: CheckpointConfigMixin, glmm_root: str
    ) -> Optional['SaigeGlmmResourceGroup']:
        inputs = SaigeGlmmResourceGroup.inputs(glmm_root)
        return await SaigeGlmmResourceGroup._use_checkpoint_if_exists_and_requested(fs, pool, b, config, inputs)

    @staticmethod
    def from_input_files(b: hb.Batch, glmm_root: str) -> 'SaigeGlmmResourceGroup':
        inputs = SaigeGlmmResourceGroup.inputs(glmm_root)
        return SaigeGlmmResourceGroup._from_input_files(b, inputs)

    @staticmethod
    def from_job_intermediate(j: hb.Job) -> 'SaigeGlmmResourceGroup':
        inputs = SaigeGlmmResourceGroup.inputs('{root}')
        return SaigeGlmmResourceGroup._from_job_intermediate(j, inputs)


class SaigeResultResourceGroup(BaseSaigeResource):
    @staticmethod
    def inputs(root: str) -> ResourceGroupInputs:
        return ResourceGroupInputs('result', result=root)

    @staticmethod
    async def use_checkpoint_if_exists_and_requested(
        fs: AsyncFS, pool: AsyncWorkerPool, b: hb.Batch, config: CheckpointConfigMixin, root: str
    ) -> Optional['SaigeResultResourceGroup']:
        inputs = SaigeResultResourceGroup.inputs(root)
        return await SaigeResultResourceGroup._use_checkpoint_if_exists_and_requested(fs, pool, b, config, inputs)

    @staticmethod
    def from_input_files(b: hb.Batch, root: str) -> 'SaigeResultResourceGroup':
        inputs = SaigeResultResourceGroup.inputs(root)
        return SaigeResultResourceGroup._from_input_files(b, inputs)

    @staticmethod
    def from_job_intermediate(j: hb.Job) -> 'SaigeResultResourceGroup':
        inputs = SaigeResultResourceGroup.inputs('{root}')
        return SaigeResultResourceGroup._from_job_intermediate(j, inputs)


class SaigeGeneResultResourceGroup(BaseSaigeResource):
    @staticmethod
    def inputs(root: str) -> ResourceGroupInputs:
        return ResourceGroupInputs('result', result=root, single=f'{root}_single')

    @staticmethod
    async def use_checkpoint_if_exists_and_requested(
        fs: AsyncFS, pool: AsyncWorkerPool, b: hb.Batch, config: CheckpointConfigMixin, output_root: str
    ) -> Optional['SaigeGeneResultResourceGroup']:
        inputs = SaigeGeneResultResourceGroup.inputs(output_root)
        return await SaigeGeneResultResourceGroup._use_checkpoint_if_exists_and_requested(fs, pool, b, config, inputs)

    @staticmethod
    def from_input_files(b: hb.Batch, root: str) -> 'SaigeGeneResultResourceGroup':
        inputs = SaigeGeneResultResourceGroup.inputs(root)
        return SaigeGeneResultResourceGroup._from_input_files(b, inputs)

    @staticmethod
    def from_job_intermediate(j: hb.Job) -> 'SaigeGeneResultResourceGroup':
        inputs = SaigeGeneResultResourceGroup.inputs('{root}')
        return SaigeGeneResultResourceGroup._from_job_intermediate(j, inputs)


class TextFile(BaseSaigeResource):
    @staticmethod
    def inputs(file: Optional[str], name: Optional[str]) -> ResourceFileInput:
        assert file or name
        return ResourceFileInput(name, file)

    @staticmethod
    async def use_checkpoint_if_exists_and_requested(
        fs: AsyncFS, pool: AsyncWorkerPool, b: hb.Batch, config: CheckpointConfigMixin, file: str
    ) -> Optional['TextFile']:
        inputs = TextFile.inputs(file, None)
        return await TextFile._use_checkpoint_if_exists_and_requested(fs, pool, b, config, inputs)

    @staticmethod
    def from_input_file(b: hb.Batch, file: str) -> 'TextFile':
        inputs = TextFile.inputs(file, None)
        return TextFile._from_input_files(b, inputs)

    @staticmethod
    def from_job_intermediate(j: hb.Job, name: str) -> 'TextFile':
        inputs = TextFile.inputs(None, name)
        return TextFile._from_job_intermediate(j, inputs)
