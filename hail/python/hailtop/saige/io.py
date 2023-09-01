from typing import Dict, Optional

import hailtop.fs as hfs
import hailtop.batch as hb

from .config_v2 import CheckpointConfigMixin
from .constants import SaigeAnalysisType


class AliasedResourceGroup:
    def __init__(self, rg: hb.ResourceGroup, aliases: Dict[str, str]):
        self._rg = rg
        self.aliases = aliases

    def __getitem__(self, item: str) -> hb.ResourceFile:
        alias = self.aliases.get(item)
        name = alias or item
        return self._rg.__getitem__(name)

    def __getattr__(self, item: str) -> hb.ResourceFile:
        alias = self.aliases.get(item)
        name = alias or item
        return self._rg.__getattr__(name)

    def __str__(self):
        return str(self._rg)


class PlinkResourceGroup(AliasedResourceGroup):
    @staticmethod
    def use_checkpoint_if_exists(
        b: hb.Batch, config: CheckpointConfigMixin, bfile_root: str
    ) -> Optional['PlinkResourceGroup']:
        if config.use_checkpoints:
            all_files_exist = all(hfs.exists(f'{bfile_root}{ext}') for ext in ['.bed', '.bim', '.fam'])
            if all_files_exist:
                return PlinkResourceGroup.from_input_files(b, bfile_root)
        return None

    @staticmethod
    def from_input_files(b: hb.Batch, bfile: str) -> 'PlinkResourceGroup':
        rg = b.read_input_group(bed=f'{bfile}.bed', bim=f'{bfile}.bim', fam=f'{bfile}.fam')
        return PlinkResourceGroup(rg, aliases={})

    @staticmethod
    def from_job_intermediate(j: hb.Job) -> 'PlinkResourceGroup':
        j.declare_resource_group(bfile={'bed': '{root}.bed', 'bim': '{root}.bim', 'fam': '{root}.fam'})
        return PlinkResourceGroup(j.bfile, aliases={})

    def checkpoint_if_requested(self, b: hb.Batch, config: CheckpointConfigMixin, output_file: str):
        if config.checkpoint_output:
            b.write_output(self._rg, output_file)


class VCFResourceGroup(AliasedResourceGroup):
    @staticmethod
    def use_checkpoint_if_exists(
        b: hb.Batch, config: CheckpointConfigMixin, vcf_root: str
    ) -> Optional['VCFResourceGroup']:
        if config.use_checkpoints:
            all_files_exist = all(hfs.exists(f'{vcf_root}{ext}') for ext in ['.vcf.gz', '.vcf.gz.tbi'])
            if all_files_exist:
                return VCFResourceGroup.from_input_files(b, vcf_root)
        return None

    @staticmethod
    def from_input_files(b: hb.Batch, root: str) -> 'VCFResourceGroup':
        rg = b.read_input_group(vcf=f'{root}.vcf.gz', tbi=f'{root}.vcf.gz.tbi')
        return VCFResourceGroup(rg, aliases={})

    @staticmethod
    def from_job_intermediate(j: hb.Job) -> 'VCFResourceGroup':
        j.declare_resource_group(vcf={'vcf.gz': '{root}.vcf.gz', 'vcf.gz.tbi': '{root}.vcf.gz.tbi'})
        return VCFResourceGroup(j.vcf, aliases={'vcf.gz': 'vcf', 'vcf.gz.tbi': 'tbi'})

    def checkpoint_if_requested(self, b: hb.Batch, config: CheckpointConfigMixin, output_file: str):
        if config.checkpoint_output:
            b.write_output(self._rg, output_file)


class BgenResourceGroup(AliasedResourceGroup):
    @staticmethod
    def use_checkpoint_if_exists(
        b: hb.Batch, config: CheckpointConfigMixin, bgen_root: str
    ) -> Optional['BgenResourceGroup']:
        if config.use_checkpoints:
            all_files_exist = all(hfs.exists(f'{bgen_root}{ext}') for ext in ['.bgen', '.bgen.bgi', '.sample'])
            if all_files_exist:
                return BgenResourceGroup.from_input_files(b, bgen_root)
        return None

    @staticmethod
    def from_input_files(b: hb.Batch, root: str) -> 'BgenResourceGroup':
        rg = b.read_input_group(bgen=f'{root}.bgen', bgi=f'{root}.bgen.bgi', sample=f'{root}.sample')
        return BgenResourceGroup(rg, aliases={})

    @staticmethod
    def from_job_intermediate(j: hb.Job) -> 'BgenResourceGroup':
        j.declare_resource_group(bgen={'bgen': '{root}.bgen', 'bgen.bgi': '{root}.bgen.bgi', 'sample': '{root}.sample'})
        return BgenResourceGroup(j.bgen, aliases={'bgen.bgi': 'bgi'})

    def checkpoint_if_requested(self, b: hb.Batch, config: CheckpointConfigMixin, output_file: str):
        if config.checkpoint_output:
            b.write_output(self._rg, output_file)


class SaigeSparseGRMResourceGroup(AliasedResourceGroup):
    @staticmethod
    def use_checkpoint_if_exists(
        b: hb.Batch, config: CheckpointConfigMixin, grm_root: str
    ) -> Optional['SaigeSparseGRMResourceGroup']:
        if config.use_checkpoints:
            all_files_exist = all(
                hfs.exists(f'{grm_root}{ext}') for ext in ['.sparseGRM.mtx', '.sparseGRM.mtx.sampleIDs.txt']
            )
            if all_files_exist:
                return SaigeSparseGRMResourceGroup.from_input_files(b, grm_root)
        return None

    @staticmethod
    def from_input_files(b: hb.Batch, grm_root: str) -> 'SaigeSparseGRMResourceGroup':
        rg = b.read_input_group(grm=f'{grm_root}.sparseGRM.mtx', sample_ids=f'{grm_root}.sparseGRM.mtx.sampleIDs.txt')
        return SaigeSparseGRMResourceGroup(rg, aliases={})

    @staticmethod
    def from_job_intermediate(j: hb.Job, relatedness_cutoff: float, num_markers: int) -> 'SaigeSparseGRMResourceGroup':
        suffix = f'_relatednessCutoff_{relatedness_cutoff}_{num_markers}_randomMarkersUsed'
        j.declare_resource_group(
            grm={
                suffix: f'{{root}}{suffix}.sparseGRM.mtx',
                f'{suffix}.sampleIDs.txt': f'{{root}}{suffix}.sparseGRM.mtx.sampleIDs.txt',
            }
        )
        return SaigeSparseGRMResourceGroup(j.grm, aliases={'grm': suffix, 'sample_ids': f'{suffix}.sampleIDs.txt'})

    def checkpoint_if_requested(self, b: hb.Batch, config: CheckpointConfigMixin, output_file: str):
        if config.checkpoint_output:
            b.write_output(self._rg, output_file)


class SaigeGlmmResourceGroup(AliasedResourceGroup):
    @staticmethod
    def use_checkpoint_if_exists(
        b: hb.Batch,
        config: CheckpointConfigMixin,
        glmm_root: str,
        analysis_type: SaigeAnalysisType,
        sparse_grm: Optional[SaigeSparseGRMResourceGroup] = None,
    ) -> Optional['SaigeGlmmResourceGroup']:
        if config.use_checkpoints:
            exts = ['.rda', '_30markers.SAIGE.results.txt', f'.{analysis_type.value}.varianceRatio.txt']

            if analysis_type == SaigeAnalysisType.GENE:
                assert sparse_grm is not None
                sparse_sigma_extension = sparse_grm._rg._value.replace("GRM", "Sigma")
                exts.append(f'.{analysis_type.value}.txt{sparse_sigma_extension}')

            all_files_exist = all(
                hfs.exists(f'{glmm_root}{ext}') for ext in ['.sparseGRM.mtx', '.sparseGRM.mtx.sampleIDs.txt']
            )
            if all_files_exist:
                return SaigeGlmmResourceGroup.from_input_files(b, glmm_root, analysis_type, sparse_grm)
        return None

    @staticmethod
    def from_input_files(
        b: hb.Batch,
        glmm_root: str,
        analysis_type: SaigeAnalysisType,
        sparse_grm: Optional[SaigeSparseGRMResourceGroup] = None,
    ) -> 'SaigeGlmmResourceGroup':
        input_files = {
            'rda': f'{glmm_root}.rda',
            'results': f'{glmm_root}_30markers.SAIGE.results.txt',
            'variance_ratio': f'{glmm_root}.{analysis_type.value}.varianceRatio.txt',
        }

        if analysis_type == SaigeAnalysisType.GENE:
            assert sparse_grm is not None
            sparse_sigma_extension = sparse_grm._rg._value.replace("GRM", "Sigma")
            input_files['sigma_variance_ratio'] = f'{glmm_root}.{analysis_type.value}.txt{sparse_sigma_extension}'

        rg = b.read_input_group(**input_files)
        return SaigeGlmmResourceGroup(rg, aliases={})

    @staticmethod
    def from_job_intermediate(
        j: hb.Job, analysis_type: SaigeAnalysisType, sparse_grm: Optional[SaigeSparseGRMResourceGroup] = None
    ) -> 'SaigeGlmmResourceGroup':
        output_files = {
            'rda': '{root}.rda',
            '_30markers.SAIGE.results.txt': '{root}_30markers.SAIGE.results.txt',
            f'{analysis_type}.varianceRatio.txt': f'{{root}}.{analysis_type.value}.varianceRatio.txt',
        }

        aliases = {'results': '_30markers.SAIGE.results.txt', 'variance_ratio': f'{analysis_type}.varianceRatio.txt'}

        if analysis_type == SaigeAnalysisType.GENE:
            assert sparse_grm is not None
            sparse_sigma_extension = sparse_grm._rg._value.replace("GRM", "Sigma")
            suffix = f'{analysis_type}.varianceRatio.txt{sparse_sigma_extension}'
            output_files[suffix] = f'{{root}}{suffix}'
            aliases['sigma_variance_ratio'] = suffix

        j.declare_resource_group(glmm=output_files)
        return SaigeGlmmResourceGroup(j.glmm, aliases=aliases)

    def checkpoint_if_requested(self, b: hb.Batch, config: CheckpointConfigMixin, output_file: str):
        if config.checkpoint_output:
            b.write_output(self._rg, output_file)


class SaigeResultResourceGroup(AliasedResourceGroup):
    @staticmethod
    def use_checkpoint_if_exists(
        b: hb.Batch, config: CheckpointConfigMixin, output_root: str, analysis_type: SaigeAnalysisType
    ) -> Optional['SaigeResultResourceGroup']:
        if config.use_checkpoints:
            if analysis_type == SaigeAnalysisType.GENE:
                exts = ['', '_single']
            else:
                exts = ['']
            all_files_exist = all(hfs.exists(f'{output_root}{ext}') for ext in exts)
            if all_files_exist:
                return SaigeResultResourceGroup.from_input_files(b, output_root, analysis_type)
        return None

    @staticmethod
    def from_input_files(b: hb.Batch, root: str, analysis_type: SaigeAnalysisType) -> 'SaigeResultResourceGroup':
        if analysis_type == SaigeAnalysisType.GENE:
            files = {'gene': root, 'single': f'{root}_single.txt'}
        else:
            assert analysis_type == SaigeAnalysisType.VARIANT
            files = {'single_variant': root}
        rg = b.read_input_group(**files)
        return SaigeResultResourceGroup(rg, aliases={})

    @staticmethod
    def from_job_intermediate(j: hb.Job, analysis_type: SaigeAnalysisType) -> 'SaigeResultResourceGroup':
        if analysis_type == SaigeAnalysisType.GENE:
            outputs = {'.gene.txt': '{root}', '.single.txt': '{root}_single'}
        else:
            assert analysis_type == SaigeAnalysisType.VARIANT
            outputs = {'single_variant.txt': '{root}'}

        j.declare_resource_group(saige=outputs)
        return SaigeResultResourceGroup(j.saige, aliases={'gene': '.gene.txt', 'single': 'single_variant.txt'})

    def checkpoint_if_requested(self, b: hb.Batch, config: CheckpointConfigMixin, output_file: str):
        if config.checkpoint_output:
            b.write_output(self._rg, output_file)


class TextFile:
    @staticmethod
    def use_checkpoint_if_exists(b: hb.Batch, config: CheckpointConfigMixin, file: str) -> Optional['TextFile']:
        if config.use_checkpoints:
            if hfs.exists(file):
                return TextFile.from_input_file(b, file)
        return None

    @staticmethod
    def from_input_file(b: hb.Batch, file: str) -> 'TextFile':
        return TextFile(b.read_input(file))

    def __init__(self, resource_file: hb.ResourceFile):
        self._rf = resource_file

    def checkpoint_if_requested(self, b: hb.Batch, config: CheckpointConfigMixin, output_file: str):
        if config.checkpoint_output:
            b.write_output(self._rf, output_file)
