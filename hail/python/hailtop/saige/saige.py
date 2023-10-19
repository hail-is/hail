import collections

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import hail as hl
from hail.methods.qc import require_col_key_str, require_row_key_variant
from hailtop.aiotools.router_fs import RouterAsyncFS
import hailtop.batch as hb
from hailtop.utils import AsyncWorkerPool

from .constants import SaigeAnalysisType, SaigeInputDataType
from .phenotype import Phenotypes
from .steps import PrepareInputsStep, SparseGRMStep, Step1NullGlmmStep, Step2SPAStep
from .variant_chunk import VariantChunks


class CheckpointDirectory:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@dataclass
class SaigeConfig:
    version: int = 1
    name: Optional[str] = 'saige'
    attributes: Optional[Dict[str, str]] = field(default_factory=collections.defaultdict, kw_only=True)
    prepare_inputs: PrepareInputsStep = field(default_factory=PrepareInputsStep, kw_only=True)
    sparse_grm: SparseGRMStep = field(default_factory=SparseGRMStep, kw_only=True)
    step1_null_glmm: Step1NullGlmmStep = field(default_factory=Step1NullGlmmStep, kw_only=True)
    step2_spa: Step2SPAStep = field(default_factory=Step2SPAStep, kw_only=True)


class SAIGE:
    def __init__(
        self, router_fs_args: Optional[dict] = None, parallelism: int = 10, config: Optional[SaigeConfig] = None
    ):
        if config is None:
            config = SaigeConfig()
        self.config = config

        self.fs = RouterAsyncFS(**router_fs_args)
        self.pool = AsyncWorkerPool(parallelism)

    def _munge_inputs(
        self,
        *,
        mt: Union[str, hl.MatrixTable],
        working_dir: str,
        phenotypes: Union[List[str], Phenotypes],
        variant_chunks: Optional[Union[str, VariantChunks]],
    ) -> Tuple[hl.MatrixTable, str, Phenotypes, VariantChunks]:
        if isinstance(mt, hl.MatrixTable):
            mt_path = f'{working_dir}/input.mt'
            mt.checkpoint(mt_path)
        else:
            mt_path = mt
            mt = hl.read_matrix_table(mt_path)

        require_row_key_variant(mt, 'saige')
        require_col_key_str(mt, 'saige')

        if isinstance(phenotypes, list):
            mt_phenotypes = Phenotypes.from_matrix_table(mt, phenotypes)
            if len(phenotypes) != mt_phenotypes:
                raise ValueError('missing phenotypes from MatrixTable')
            phenotypes = mt_phenotypes

        if variant_chunks is None:
            variant_chunks = VariantChunks.from_matrix_table(mt)
        elif isinstance(variant_chunks, str):
            if variant_chunks.endswith('.bed'):
                variant_chunks = VariantChunks.from_bed(variant_chunks)
            else:
                variant_chunks = VariantChunks.from_locus_intervals(variant_chunks)
        else:
            assert isinstance(variant_chunks, VariantChunks)

        return (mt, mt_path, phenotypes, variant_chunks)

    async def run_saige(
        self,
        *,
        mt: Union[str, hl.MatrixTable],
        phenotypes: Union[List[str], Phenotypes],
        covariates: List[str],
        output_dir: str,
        b: Optional[hb.Batch] = None,
        checkpoint_dir: Optional[str] = None,
        variant_chunks: Optional[Union[str, VariantChunks]] = None,
        run_kwargs: Optional[dict] = None,
    ):
        if checkpoint_dir is None:
            checkpoint_dir = hl.TemporaryDirectory()
        else:
            checkpoint_dir = CheckpointDirectory(checkpoint_dir)

        with checkpoint_dir:
            mt, mt_path, phenotypes, variant_chunks = self._munge_inputs(
                mt=mt, working_dir=checkpoint_dir.name, phenotypes=phenotypes, variant_chunks=variant_chunks
            )

            if b is None:
                b = hb.Batch(name=self.config.name, attributes=self.config.attributes)

            input_phenotypes, input_plink_data = await self.config.prepare_inputs.call(
                self.fs, self.pool, b, mt, phenotypes, checkpoint_dir.name
            )

            user_id_col = list(mt.col_key)[0]

            if 'GP' in list(mt.entry):
                input_data_type = SaigeInputDataType.BGEN
            else:
                assert 'GT' in list(mt.entry)
                input_data_type = SaigeInputDataType.VCF

            for phenotype_group in phenotypes:
                for phenotype in phenotype_group:
                    null_glmm = await self.config.step1_null_glmm.call(
                        self.fs,
                        self.pool,
                        b,
                        input_bfile=input_plink_data,
                        input_phenotypes=input_phenotypes,
                        phenotype=phenotype,
                        analysis_type=SaigeAnalysisType.VARIANT,
                        covariates=covariates,
                        user_id_col=user_id_col,
                        output_dir=checkpoint_dir.name,
                    )
                    for variant_chunk in variant_chunks:
                        await self.config.step2_spa.call(
                            self.fs,
                            self.pool,
                            b,
                            mt_path=mt_path,
                            output_dir=output_dir,
                            analysis_type=SaigeAnalysisType.VARIANT,
                            null_model=null_glmm,
                            input_data_type=input_data_type,
                            chunk=variant_chunk,
                            phenotype=phenotype,
                        )

            run_kwargs = run_kwargs or {}
            run_kwargs.pop('wait')
            b.run(**run_kwargs, wait=True)

            results_tables = []
            for phenotype_group in phenotypes:
                for phenotype in phenotype_group:
                    results_path = self.config.step2_spa.output_dir(output_dir, phenotype)
                    ht = hl.import_table(results_path, impute=True)
                    ht = ht.annotate(phenotype=phenotype.name)
                    results_tables.append(ht)

            if len(results_tables) == 1:
                results = results_tables[0]
            else:
                results = results_tables[0].union(*results_tables[1:])

            results.checkpoint(f'{output_dir}/saige_results.ht')

        return results

    def run_saige_gene(
        self,
        *,
        mt: Union[str, hl.MatrixTable],
        phenotypes: Union[List[str], Phenotypes],
        covariates: List[str],
        group_col: str,
        output_dir: str,
        b: Optional[hb.Batch] = None,
        checkpoint_dir: Optional[str] = None,
        variant_chunks: Optional[VariantChunks] = None,
        run_kwargs: Optional[dict] = None,
    ):
        if checkpoint_dir is None:
            working_dir = hl.TemporaryDirectory()
        else:
            working_dir = CheckpointDirectory(checkpoint_dir)

        with checkpoint_dir:
            mt, mt_path, phenotypes, variant_chunks = self._munge_inputs(
                mt=mt, working_dir=working_dir.name, phenotypes=phenotypes, variant_chunks=variant_chunks
            )

            if group_col not in mt.row:
                raise ValueError(f'could not find row annotation {group_col}')

            group = mt[group_col]
            if group.dtype != hl.tarray(hl.tstruct(group=hl.tstr, ann=hl.tstr)):
                raise ValueError(
                    f'group row annotation must have type {hl.tarray(hl.tstruct(group=hl.tstr, ann=hl.tstr))}. Found {group.dtype}'
                )

            if b is None:
                b = hb.Batch(name=self.config.name, attributes=self.config.attributes)

            input_phenotypes, input_plink_data = await self.config.prepare_inputs.call(
                self.fs, self.pool, b, mt, phenotypes, working_dir.name
            )

            user_id_col = list(mt.col_key)[0]

            if 'GP' in list(mt.entry):
                input_data_type = SaigeInputDataType.BGEN
            else:
                assert 'GT' in list(mt.entry)
                input_data_type = SaigeInputDataType.VCF

            sparse_grm = await self.config.sparse_grm.call(self.fs, self.pool, b, input_plink_data, working_dir.name)

            for phenotype_group in phenotypes:
                for phenotype in phenotype_group:
                    null_glmm = await self.config.step1_null_glmm.call(
                        self.fs,
                        self.pool,
                        b,
                        input_bfile=input_plink_data,
                        input_phenotypes=input_phenotypes,
                        phenotype=phenotype,
                        analysis_type=SaigeAnalysisType.VARIANT,
                        covariates=covariates,
                        user_id_col=user_id_col,
                        output_dir=working_dir.name,
                        # sparse_grm=sparse_grm,  # FIXME: why isn't this needed anymore. must be a bug somewhere
                    )
                    for variant_chunk in variant_chunks:
                        await self.config.step2_spa.call(
                            self.fs,
                            self.pool,
                            b,
                            mt_path=mt_path,
                            output_dir=output_dir,
                            analysis_type=SaigeAnalysisType.VARIANT,
                            null_model=null_glmm,
                            input_data_type=input_data_type,
                            chunk=variant_chunk,
                            phenotype=phenotype,
                            sparse_grm=sparse_grm,
                            group_col=group_col,
                        )

            run_kwargs = run_kwargs or {}
            run_kwargs.pop('wait')
            b.run(**run_kwargs, wait=True)
