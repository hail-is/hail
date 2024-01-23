import collections
import functools

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, cast

import hail as hl
from hail.methods.qc import require_col_key_str, require_row_key_variant
from hailtop.aiotools.router_fs import RouterAsyncFS
import hailtop.batch as hb
from hailtop.utils import async_to_blocking, bounded_gather

from .constants import SaigeAnalysisType, SaigeInputDataType
from .io import load_plink_file, load_text_file
from .phenotype import Phenotype
from .steps import PrepareInputsStep, SparseGRMStep, Step1NullGlmmStep, Step2SPAStep
from .variant_chunk import VariantChunk


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

        self.parallelism = parallelism
        self.fs = RouterAsyncFS(**router_fs_args)

    async def close(self):
        await self.fs.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def run(
        self,
        *,
        mt_path: str,
        null_model_plink_path: str,
        phenotypes_path: str,
        output: str,
        phenotypes: List[str],
        covariates: List[str],
        variant_intervals: List[hl.Interval],
        b: Optional[hb.Batch] = None,
        checkpoint_dir: Optional[str] = None,
        run_kwargs: Optional[dict] = None,
    ):
        with hl.TemporaryDirectory() as temp_dir:
            if b is None:
                b = hb.Batch(name=self.config.name, attributes=self.config.attributes)

            mt = hl.read_matrix_table(mt_path)
            require_col_key_str(mt, 'saige')
            require_row_key_variant(mt, 'saige')

            input_phenotypes = load_text_file(phenotypes_path)
            input_plink_data = cast(PlinkResourceGroup, b.)
            load_text_file(self.fs, b, )

            input_phenotypes, input_plink_data = await self.config.prepare_inputs.call(
                self.fs, b, mt, phenotypes, temp_dir.name, checkpoint_dir
            )

            user_id_col = list(mt.col_key)[0]

            if 'GP' in list(mt.entry):
                input_data_type = SaigeInputDataType.BGEN
            else:
                assert 'GT' in list(mt.entry)
                input_data_type = SaigeInputDataType.VCF

            null_glmms = await bounded_gather(
                *[
                    functools.partial(
                        self.config.step1_null_glmm.call,
                        self.fs,
                        b,
                        input_bfile=input_plink_data,
                        input_phenotypes=input_phenotypes,
                        phenotype=phenotype,
                        analysis_type=SaigeAnalysisType.VARIANT,
                        covariates=covariates,
                        user_id_col=user_id_col,
                        temp_dir=temp_dir.name,
                        checkpoint_dir=checkpoint_dir,
                    )
                    for phenotype in phenotypes
                ],
                parallelism=self.parallelism,
            )

            await bounded_gather(
                *[
                    functools.partial(
                        self.config.step2_spa.call,
                        self.fs,
                        b,
                        mt_path=mt_path,
                        temp_dir=temp_dir,
                        checkpoint_dir=checkpoint_dir,
                        analysis_type=SaigeAnalysisType.VARIANT,
                        null_model=null_glmm,
                        input_data_type=input_data_type,
                        chunk=variant_chunk,
                        phenotype=phenotype,
                    )
                    for phenotype, null_glmm in zip(phenotypes, null_glmms)
                    for variant_chunk in variant_chunks
                ],
                parallelism=self.parallelism,
            )

            run_kwargs = run_kwargs or {}
            run_kwargs.pop('wait')
            b.run(**run_kwargs, wait=True)

            results_tables = []
            for phenotype in phenotypes:
                results_path = self.config.step2_spa.output_dir(temp_dir, checkpoint_dir, phenotype)
                ht = hl.import_table(results_path, impute=True)
                ht = ht.annotate(phenotype=phenotype.name)
                results_tables.append(ht)

            if len(results_tables) == 1:
                results = results_tables[0]
            else:
                results = results_tables[0].union(*results_tables[1:])

            results.write(output, overwrite=True)

        return results


async def async_run_saige(
    config: Optional[SaigeConfig],
    mt_path: str,
    phenotypes: List[Phenotype],
    covariates: List[str],
    variant_chunks: List[VariantChunk],
    output: str,
    batch: Optional[hb.Batch] = None,
    checkpoint_dir: Optional[str] = None,
    run_kwargs: Optional[dict] = None,
    router_fs_args: Optional[dict] = None,
    parallelism: int = 10,
) -> hl.Table:
    saige = SAIGE(router_fs_args, parallelism, config)
    async with saige:
        return await saige.run(
            mt_path=mt_path,
            phenotypes=phenotypes,
            covariates=covariates,
            variant_chunks=variant_chunks,
            output=output,
            b=batch,
            checkpoint_dir=checkpoint_dir,
            run_kwargs=run_kwargs,
        )


def run_saige(
    config: Optional[SaigeConfig],
    mt_path: str,
    phenotypes: List[Phenotype],
    covariates: List[str],
    variant_chunks: List[VariantChunk],
    output: str,
    batch: Optional[hb.Batch] = None,
    checkpoint_dir: Optional[str] = None,
    run_kwargs: Optional[dict] = None,
    router_fs_args: Optional[dict] = None,
    parallelism: int = 10,
) -> hl.Table:
    return async_to_blocking(
        async_run_saige(
            config=config,
            mt_path=mt_path,
            phenotypes=phenotypes,
            covariates=covariates,
            variant_chunks=variant_chunks,
            output=output,
            batch=batch,
            checkpoint_dir=checkpoint_dir,
            run_kwargs=run_kwargs,
            router_fs_args=router_fs_args,
            parallelism=parallelism,
        )
    )
