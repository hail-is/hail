import asyncio
from typing import Optional, NewType, Type, TypeVar, Union, cast

from hailtop.aiotools.fs import AsyncFS
from hailtop.utils import secret_alnum_string
import hailtop.batch as hb


from .config import CheckpointConfigMixin
from .constants import SaigeAnalysisType


RG = TypeVar('RG', bound=hb.ResourceGroup)
RF = TypeVar('RF', bound=hb.ResourceFile)


def declare_resource_group(j, **kwargs):
    uid = secret_alnum_string(5)
    j.declare_resource_group(**{uid: kwargs})


def checkpoint_if_requested(resource: hb.Resource, b: hb.Batch, config: CheckpointConfigMixin, output_file: str):
    if config.checkpoint_output:
        b.write_output(resource, output_file)


async def use_checkpoints_if_exist_and_requested(
    typ: Type[RG], fs: AsyncFS, b: hb.Batch, config: CheckpointConfigMixin, **inputs: str
) -> Optional[RG]:
    if config.use_checkpoints and not config.overwrite:
        await load_files(typ, fs, b, **inputs)
    return None


async def load_files(typ: Type[RG], fs: AsyncFS, b: hb.Batch, **inputs: str) -> Optional[RG]:
    all_files_exist = all(await asyncio.gather(*[fs.exists(input) for input in inputs.values()]))
    if all_files_exist:
        return cast(typ, b.read_input_group(**inputs))
    return None


async def use_checkpoint_if_exists_and_requested(
    typ: Type[RF], fs: AsyncFS, b: hb.Batch, config: CheckpointConfigMixin, input: str
) -> Optional[RF]:
    if config.use_checkpoints and not config.overwrite:
        await load_file(typ, fs, b, input)
    return None


async def load_file(typ: Type[RF], fs: AsyncFS, b: hb.Batch, input: str) -> Optional[RF]:
    if await fs.exists(input):
        return cast(typ, b.read_input(input))
    return None


PlinkResourceGroup = NewType('PlinkResourceGroup', hb.ResourceGroup)
BgenResourceGroup = NewType('BgenResourceGroup', hb.ResourceGroup)
VcfResourceGroup = NewType('VcfResourceGroup', hb.ResourceGroup)
TextResourceFile = NewType('TextResourceFile', hb.ResourceFile)
SaigeSparseGRMResourceGroup = NewType('SaigeSparseGRMResourceGroup', hb.ResourceGroup)
SaigeGeneGLMMResourceGroup = NewType('SaigeGeneGLMMResourceGroup', hb.ResourceGroup)
SaigeGLMMResourceGroup = NewType('SaigeGLMMResourceGroup', hb.ResourceGroup)
SaigeResultResourceGroup = NewType('SaigeResultResourceGroup', hb.ResourceGroup)
SaigeGeneResultResourceGroup = NewType('SaigeGeneResultResourceGroup', hb.ResourceGroup)


def new_plink_file(j: hb.Job) -> PlinkResourceGroup:
    j.declare_resource_group(bfile=dict(bed='{root}.bed', bim='{root}.bim', fam='{root}.fam'))
    return cast(PlinkResourceGroup, j.bfile)


async def load_plink_file(
    fs: AsyncFS, b: hb.Batch, config: Optional[CheckpointConfigMixin], root: str
) -> Optional[PlinkResourceGroup]:
    if config is None:
        return await load_files(PlinkResourceGroup, fs, b, bed=f'{root}.bed', bim=f'{root}.bim', fam=f'{root}.fam')
    return await use_checkpoints_if_exist_and_requested(
        PlinkResourceGroup, fs, b, config, bed=f'{root}.bed', bim=f'{root}.bim', fam=f'{root}.fam'
    )


def new_bgen_file(j: hb.Job) -> BgenResourceGroup:
    j.declare_resource_group(bgen=dict(bgen='{root}.bgen', bgi='{root}.bgen.bgi', sample='{root}.sample'))
    return cast(BgenResourceGroup, j.bgen)


async def load_bgen_file(
    fs: AsyncFS, b: hb.Batch, config: CheckpointConfigMixin, root: str
) -> Optional[BgenResourceGroup]:
    return await use_checkpoints_if_exist_and_requested(
        BgenResourceGroup, fs, b, config, bgen=f'{root}.bgen', bgi=f'{root}.bgen.bgi', sample=f'{root}.sample'
    )


def new_vcf_file(j: hb.Job) -> VcfResourceGroup:
    j.declare_resource_group(vcf=dict(vcf='{root}.vcf.gz', tbi='{root}.vcf.gz.tbi'))
    return cast(VcfResourceGroup, j.vcf)


async def load_vcf_file(
    fs: AsyncFS, b: hb.Batch, config: CheckpointConfigMixin, root: str
) -> Optional[VcfResourceGroup]:
    return await use_checkpoints_if_exist_and_requested(
        VcfResourceGroup, fs, b, config, vcf=f'{root}.vcf.gz', tbi=f'{root}.vcf.gz.tbi'
    )


def new_text_file(j: hb.Job) -> TextResourceFile:
    j.declare_resource_group(file=dict(file='{root}'))
    return cast(TextResourceFile, cast(hb.ResourceGroup, j.file)['file'])


async def load_text_file(
    fs: AsyncFS, b: hb.Batch, config: Optional[CheckpointConfigMixin], file: str
) -> Optional[TextResourceFile]:
    if config is None:
        return await load_file(TextResourceFile, fs, b, file)
    return await use_checkpoint_if_exists_and_requested(TextResourceFile, fs, b, config, file)


def new_saige_sparse_grm_file(j: hb.Job, relatedness_cutoff: float, num_markers: int) -> SaigeSparseGRMResourceGroup:
    suffix = f'_relatednessCutoff_{relatedness_cutoff}_{num_markers}_randomMarkersUsed'
    j.declare_resource_group(
        grm=dict(grm=f'{{root}}{suffix}.sparseGRM.mtx', sample_ids=f'{{root}}{suffix}.sparseGRM.mtx.sampleIDs.txt')
    )
    return cast(SaigeSparseGRMResourceGroup, j.grm)


async def load_saige_sparse_grm_file(
    fs: AsyncFS, b: hb.Batch, config: CheckpointConfigMixin, root: str
) -> SaigeSparseGRMResourceGroup:
    return await use_checkpoints_if_exist_and_requested(
        SaigeSparseGRMResourceGroup,
        fs,
        b,
        config,
        grm=f'{root}.sparseGRM.mtx',
        sample_ids=f'{root}.sparseGRM.mtx.sampleIDs.txt',
    )


def new_saige_gene_glmm_file(j: hb.Job) -> SaigeGeneGLMMResourceGroup:
    j.declare_resource_group(
        glmm=dict(
                mtx='{root}.sparseGRM.mtx',
                sample_ids='{root}.sparseGRM.mtx.sampleIDs.txt',
                rda='{root}.rda',
                results='{root}_30markers.SAIGE.results.txt',
                variance_ratio='{root}.gene.varianceRatio.txt',
            )
    )
    return cast(SaigeGeneGLMMResourceGroup, j.glmm)


async def load_saige_gene_glmm_file(
    fs: AsyncFS, b: hb.Batch, config: CheckpointConfigMixin, root: str
) -> SaigeGeneGLMMResourceGroup:
    return await use_checkpoints_if_exist_and_requested(
        SaigeGeneGLMMResourceGroup,
        fs,
        b,
        config,
        mtx=f'{root}.sparseGRM.mtx',
        sample_ids=f'{root}.sparseGRM.mtx.sampleIDs.txt',
        rda=f'{root}.rda',
        results=f'{root}_30markers.SAIGE.results.txt',
        variance_ratio=f'{root}.gene.varianceRatio.txt',
    )


def new_saige_variant_glmm_file(j: hb.Job) -> SaigeGLMMResourceGroup:
    j.declare_resource_group(glmm=dict(
                rda='{root}.rda',
                variance_ratio='{root}.varianceRatio.txt',
            )
    )
    return cast(SaigeGLMMResourceGroup, j.glmm)


async def load_saige_variant_glmm_file(
    fs: AsyncFS, b: hb.Batch, config: CheckpointConfigMixin, root: str
) -> SaigeGLMMResourceGroup:
    return await use_checkpoints_if_exist_and_requested(
        SaigeGLMMResourceGroup,
        fs,
        b,
        config,
        rda=f'{root}.rda',
        results=f'{root}_30markers.SAIGE.results.txt',
        variance_ratio=f'{root}.gene.varianceRatio.txt',
    )


def new_saige_glmm_file(
    j: hb.Job, analysis_type: SaigeAnalysisType
) -> Union[SaigeGLMMResourceGroup, SaigeGeneGLMMResourceGroup]:
    if analysis_type == SaigeAnalysisType.VARIANT:
        return new_saige_variant_glmm_file(j)
    assert analysis_type == SaigeAnalysisType.GENE
    return new_saige_gene_glmm_file(j)


async def load_saige_glmm_file(
    fs: AsyncFS, b: hb.Batch, config: CheckpointConfigMixin, root: str, analysis_type: SaigeAnalysisType
) -> Union[SaigeGLMMResourceGroup, SaigeGeneGLMMResourceGroup]:
    if analysis_type == SaigeAnalysisType.VARIANT:
        return await load_saige_variant_glmm_file(fs, b, config, root)
    assert analysis_type == SaigeAnalysisType.GENE
    return await load_saige_gene_glmm_file(fs, b, config, root)


def new_saige_variant_result_file(j: hb.Job) -> SaigeResultResourceGroup:
    j.declare_resource_group(result=dict(result='{root}'))
    return cast(SaigeResultResourceGroup, j.result)


async def load_saige_variant_result_file(
    fs: AsyncFS, b: hb.Batch, config: CheckpointConfigMixin, root: str
) -> SaigeResultResourceGroup:
    return await use_checkpoints_if_exist_and_requested(SaigeResultResourceGroup, fs, b, config, result=root)


def new_saige_gene_result_file(j: hb.Job) -> SaigeGeneResultResourceGroup:
    j.declare_resource_group(bgen=dict(result='{root}', single='{root}_single'))
    return cast(SaigeGeneResultResourceGroup, j.bgen)


async def load_saige_gene_result_file(
    fs: AsyncFS, b: hb.Batch, config: CheckpointConfigMixin, root: str
) -> SaigeGeneResultResourceGroup:
    return await use_checkpoints_if_exist_and_requested(
        SaigeGeneResultResourceGroup, fs, b, config, result=root, single=f'{root}_single'
    )


def new_saige_result_file(j: hb.Job, analysis_type: SaigeAnalysisType) -> Union[SaigeGeneResultResourceGroup, SaigeResultResourceGroup]:
    if analysis_type == SaigeAnalysisType.VARIANT:
        return new_saige_variant_result_file(j)
    assert analysis_type == SaigeAnalysisType.GENE
    return new_saige_gene_result_file(j)


async def load_saige_result_file(
    fs: AsyncFS, b: hb.Batch, config: CheckpointConfigMixin, root: str, analysis_type: SaigeAnalysisType
) -> Union[SaigeResultResourceGroup, SaigeGeneResultResourceGroup]:
    if analysis_type == SaigeAnalysisType.VARIANT:
        return await load_saige_variant_result_file(fs, b, config, root)
    assert analysis_type == SaigeAnalysisType.GENE
    return await load_saige_gene_result_file(fs, b, config, root)
