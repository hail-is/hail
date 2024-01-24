import collections
import functools

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import hail as hl
from hail.methods.qc import require_col_key_str, require_row_key_variant
from hailtop.aiotools.router_fs import RouterAsyncFS
import hailtop.batch as hb
from hailtop.utils import async_to_blocking, bounded_gather

from .constants import SaigeAnalysisType, SaigeInputDataType
from .io import load_plink_file, load_text_file
from .phenotype import Phenotype, Phenotypes, SaigePhenotype
from .steps import CompileAllResultsStep, CompilePhenotypeResultsStep, SparseGRMStep, Step1NullGlmmStep, Step2SPAStep
from .variant_chunk import VariantChunk


@dataclass
class SaigeConfig:
    version: int = 1
    name: Optional[str] = 'saige'
    attributes: Optional[Dict[str, str]] = field(default_factory=collections.defaultdict, kw_only=True)
    sparse_grm: SparseGRMStep = field(default_factory=SparseGRMStep, kw_only=True)
    step1_null_glmm: Step1NullGlmmStep = field(default_factory=Step1NullGlmmStep, kw_only=True)
    step2_spa: Step2SPAStep = field(default_factory=Step2SPAStep, kw_only=True)
    compile_phenotype_results: CompilePhenotypeResultsStep = field(default_factory=CompilePhenotypeResultsStep, kw_only=True)
    compile_all_results: CompileAllResultsStep = field(default_factory=CompileAllResultsStep, kw_only=True)


async def async_saige(
        *,
        mt_path: str,
        null_model_plink_path: str,
        phenotypes_path: str,
        output_path: str,
        phenotypes: Phenotypes,
        covariates: List[str],
        variant_chunks: List[VariantChunk],
        group_annotations_file: Optional[str] = None,
        b: Optional[hb.Batch] = None,
        checkpoint_dir: Optional[str] = None,
        run_kwargs: Optional[dict] = None,
        router_fs_args: Optional[dict] = None,
        parallelism: int = 10,
        config: Optional[SaigeConfig] = None
):
    if config is None:
        config = SaigeConfig()

    async with RouterAsyncFS(**router_fs_args) as fs:
        with hl.TemporaryDirectory() as temp_dir:
            if b is None:
                b = hb.Batch(name=config.name, attributes=config.attributes)

            if group_annotations_file is not None:
                analysis_type = SaigeAnalysisType.GENE
                group_annotations = load_text_file(fs, b, None, group_annotations_file)
            else:
                analysis_type = SaigeAnalysisType.VARIANT
                group_annotations = None

            mt = hl.read_matrix_table(mt_path)
            require_col_key_str(mt, 'saige')
            require_row_key_variant(mt, 'saige')

            input_phenotypes = load_text_file(fs, b, None, phenotypes_path)
            input_null_model_plink_data = load_plink_file(fs, b, None, null_model_plink_path)

            user_id_col = list(mt.col_key)[0]

            if 'GP' in list(mt.entry):
                input_data_type = SaigeInputDataType.BGEN
            else:
                assert 'GT' in list(mt.entry)
                input_data_type = SaigeInputDataType.VCF

            maybe_phenotype_results = await bounded_gather(
                *[
                    functools.partial(
                        config.compile_phenotype_results.check_if_output_exists,
                        fs,
                        b,
                        phenotype=phenotype,
                        temp_dir=temp_dir.name,
                        checkpoint_dir=checkpoint_dir
                    )
                    for phenotype in phenotypes.phenotypes
                ]
            )

            existing_phenotype_results = {phenotype: result
                                          for phenotype, result in zip(phenotypes.phenotypes, maybe_phenotype_results)
                                          if result is not None}

            null_glmms = await bounded_gather(
                *[
                    functools.partial(
                        config.step1_null_glmm.call,
                        fs,
                        b,
                        input_bfile=input_null_model_plink_data,
                        input_phenotypes=input_phenotypes,
                        phenotype=phenotype,
                        analysis_type=analysis_type,
                        covariates=covariates,
                        user_id_col=user_id_col,
                        temp_dir=temp_dir.name,
                        checkpoint_dir=checkpoint_dir,
                    )
                    for phenotype in phenotypes.phenotypes
                    if phenotype not in existing_phenotype_results
                ],
                parallelism=parallelism,
            )

            step2_spa_fs = [(phenotype, chunk, functools.partial(config.step2_spa.call,
                                                                 fs,
                                                                 b,
                                                                 mt_path=mt_path,
                                                                 temp_dir=temp_dir,
                                                                 checkpoint_dir=checkpoint_dir,
                                                                 analysis_type=analysis_type,
                                                                 null_model=null_glmm,
                                                                 input_data_type=input_data_type,
                                                                 interval=chunk,
                                                                 phenotype=phenotype,
                                                                 group_annotations=group_annotations))
                            for phenotype, null_glmm in zip(phenotypes.phenotypes, null_glmms)
                            for chunk in variant_chunks
                            if phenotype not in existing_phenotype_results
                            ]

            step2_spa_results = await bounded_gather(*[f for _, _, f in step2_spa_fs], parallelism=parallelism)

            step2_spa_jobs_by_phenotype = collections.defaultdict(list)
            for ((phenotype, _, _), result) in zip(step2_spa_fs, step2_spa_results):
                if result.source is not None:
                    step2_spa_jobs_by_phenotype[phenotype].append(result.source)

            compiled_results = await bounded_gather(
                *[
                    functools.partial(
                        config.compile_phenotype_results.call,
                        fs=fs,
                        b=b,
                        phenotype=phenotype,
                        results_path=config.step2_spa.output_dir(temp_dir, checkpoint_dir, phenotype.name),
                        dependencies=step2_spa_jobs_by_phenotype[phenotype],
                        temp_dir=temp_dir.name,
                        checkpoint_dir=checkpoint_dir
                    )
                    for phenotype in phenotypes.phenotypes
                ],
                parallelism=parallelism,
            )

            await config.compile_all_results.call(
                fs,
                b,
                results_path=config.compile_phenotype_results.results_path(temp_dir, checkpoint_dir),
                output_ht_path=output_path,
                dependencies=[result.source for result in compiled_results],
            )

            run_kwargs = run_kwargs or {}
            run_kwargs.pop('wait')
            b.run(**run_kwargs, wait=True)


def saige(
        *,
        mt_path: str,
        null_model_plink_path: str,
        phenotypes_path: str,
        output_path: str,
        phenotypes: List[Dict[str, SaigePhenotype]],
        covariates: List[str],
        variant_intervals: List[hl.Interval],
        b: Optional[hb.Batch] = None,
        checkpoint_dir: Optional[str] = None,
        run_kwargs: Optional[dict] = None,
        router_fs_args: Optional[dict] = None,
        parallelism: int = 10,
        config: Optional[SaigeConfig] = None
):
    return async_to_blocking(async_saige(
        mt_path=mt_path,
        null_model_plink_path=null_model_plink_path,
        phenotypes_path=phenotypes_path,
        output_path=output_path,
        phenotypes=phenotypes,
        covariates=covariates,
        variant_intervals=variant_intervals,
        b=b,
        checkpoint_dir=checkpoint_dir,
        run_kwargs=run_kwargs,
        router_fs_args=router_fs_args,
        parallelism=parallelism,
        config=config
    ))
