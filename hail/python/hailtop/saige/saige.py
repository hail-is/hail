import collections
import functools

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import hail as hl
from hail.methods.qc import require_col_key_str, require_row_key_variant
from hailtop.aiotools.router_fs import RouterAsyncFS
import hailtop.batch as hb
from hailtop.utils import async_to_blocking, bounded_gather

from .constants import SaigeAnalysisType, SaigeInputDataType
from .io import load_plink_file, load_text_file
from .phenotype import Phenotype, PhenotypeConfig, SaigePhenotype
from .steps import CompileAllResultsStep, CompilePhenotypeResultsStep, SparseGRMStep, Step1NullGlmmStep, Step2SPAStep
from .variant_chunk import VariantChunk


@dataclass
class SaigeConfig:
    """Class for specifying the configuration properties for all SAIGE jobs.

    Examples
    --------

    Create a custom SaigeConfig that sets the configuration properties for the Sparse GRM step.

    >>> saige_config = SaigeConfig(name='my-saige-analysis', attributes={'pop': 'eur'},
    ...                            sparse_grm=SparseGRMStep(cpu=4, memory='highmem', relatedness_cutoff=0.05))
    """

    name: Optional[str] = 'saige'
    """Name to give to the batch if it is not already created by the user."""

    attributes: Optional[Dict[str, str]] = field(default_factory=collections.defaultdict, kw_only=True)
    """Attributes to give to the batch if it is not already created by the user."""

    sparse_grm: SparseGRMStep = field(default_factory=SparseGRMStep, kw_only=True)
    """Configuration for running the Sparse GRM job."""

    step1_null_glmm: Step1NullGlmmStep = field(default_factory=Step1NullGlmmStep, kw_only=True)
    """Configuration for running the SAIGE Step1 Null GLMM job."""

    step2_spa: Step2SPAStep = field(default_factory=Step2SPAStep, kw_only=True)
    """Configuration for running the SAIGE Step2 saddle point approximation job."""

    compile_phenotype_results: CompilePhenotypeResultsStep = field(default_factory=CompilePhenotypeResultsStep, kw_only=True)
    """Configuration for jobs that compile SAIGE results by phenotype."""

    compile_all_results: CompileAllResultsStep = field(default_factory=CompileAllResultsStep, kw_only=True)
    """Configuration for running the final job that concatenates all results together into a single Hail Table."""


async def async_saige(
        *,
        mt_path: str,
        null_model_plink_path: str,
        output_path: str,
        phenotype_config: PhenotypeConfig,
        variant_intervals: Optional[List[hl.Interval]],
        group_annotations_file: Optional[str] = None,
        null_model_sample_list: Optional[str] = None,
        b: Optional[hb.Batch] = None,
        checkpoint_dir: Optional[str] = None,
        run_kwargs: Optional[dict] = None,
        router_fs_args: Optional[dict] = None,
        parallelism: int = 10,
        saige_config: Optional[SaigeConfig] = None
):
    if saige_config is None:
        saige_config = SaigeConfig()

    if router_fs_args is None:
        router_fs_args = {}

    async with RouterAsyncFS(**router_fs_args) as fs:
        with hl.TemporaryDirectory() as temp_dir:
            if b is None:
                b = hb.Batch(name=saige_config.name, attributes=saige_config.attributes)

            if group_annotations_file is not None:
                analysis_type = SaigeAnalysisType.GENE
                group_annotations = await load_text_file(fs, b, None, group_annotations_file)
            else:
                analysis_type = SaigeAnalysisType.VARIANT
                group_annotations = None

            mt = hl.read_matrix_table(mt_path)
            require_col_key_str(mt, 'saige')
            require_row_key_variant(mt, 'saige')

            if variant_intervals is None:
                first_row = mt.rows().select().head(1).collect()[0]
                last_row = mt.rows().select().tail(1).collect()[0]
                variant_intervals = [hl.Interval(first_row.locus, last_row.locus, includes_start=True, includes_end=True)]

            variant_chunks = [VariantChunk(interval) for interval in variant_intervals]

            input_phenotypes = await load_text_file(fs, b, None, phenotype_config.phenotypes_file)
            input_null_model_plink_data = await load_plink_file(fs, b, None, null_model_plink_path)

            if 'GP' in list(mt.entry):
                input_data_type = SaigeInputDataType.BGEN
            else:
                assert 'GT' in list(mt.entry)
                input_data_type = SaigeInputDataType.VCF

            maybe_phenotype_results = await bounded_gather(
                *[
                    functools.partial(
                        saige_config.compile_phenotype_results.check_if_output_exists,
                        fs,
                        b,
                        phenotype=phenotype,
                        temp_dir=temp_dir,
                        checkpoint_dir=checkpoint_dir
                    )
                    for phenotype in phenotype_config.phenotypes
                ]
            )

            existing_phenotype_results = {phenotype.name: result
                                          for phenotype, result in zip(phenotype_config.phenotypes, maybe_phenotype_results)
                                          if result is not None}

            if analysis_type == SaigeAnalysisType.GENE or saige_config.step1_null_glmm.use_sparse_grm_to_fit_null:
                sparse_grm = saige_config.sparse_grm._call(fs, b, input_null_model_plink_data, temp_dir, checkpoint_dir)
            else:
                sparse_grm = None

            if null_model_sample_list is not None:
                null_model_samples = await load_text_file(fs, b, None, null_model_sample_list)
            else:
                null_model_samples = None

            null_glmms = await bounded_gather(
                *[
                    functools.partial(
                        saige_config.step1_null_glmm._call,
                        fs,
                        b,
                        input_bfile=input_null_model_plink_data,
                        input_phenotypes=input_phenotypes,
                        keep_samples_list=null_model_samples,
                        phenotype_config=phenotype_config,
                        phenotype=phenotype,
                        sparse_grm=sparse_grm,
                        analysis_type=analysis_type,
                        temp_dir=temp_dir,
                        checkpoint_dir=checkpoint_dir,
                    )
                    for phenotype in phenotype_config.phenotypes
                    if phenotype.name not in existing_phenotype_results
                ],
                parallelism=parallelism,
            )

            step2_spa_fs = [(phenotype, chunk, functools.partial(saige_config.step2_spa._call,
                                                                 fs,
                                                                 b,
                                                                 mt_path=mt_path,
                                                                 temp_dir=temp_dir,
                                                                 checkpoint_dir=checkpoint_dir,
                                                                 analysis_type=analysis_type,
                                                                 null_model=null_glmm,
                                                                 input_data_type=input_data_type,
                                                                 chunk=chunk,
                                                                 phenotype=phenotype,
                                                                 phenotype_config=phenotype_config,
                                                                 group_annotations=group_annotations))
                            for phenotype, null_glmm in zip(phenotype_config.phenotypes, null_glmms)
                            for chunk in variant_chunks
                            if phenotype.name not in existing_phenotype_results
                            ]

            step2_spa_results = await bounded_gather(*[f for _, _, f in step2_spa_fs], parallelism=parallelism)

            step2_spa_jobs_by_phenotype = collections.defaultdict(list)
            for ((phenotype, _, _), result) in zip(step2_spa_fs, step2_spa_results):
                if result.source() is not None:
                    step2_spa_jobs_by_phenotype[phenotype.name].append(result.source())

            compiled_results = await bounded_gather(
                *[
                    functools.partial(
                        saige_config.compile_phenotype_results._call,
                        fs=fs,
                        b=b,
                        phenotype=phenotype,
                        results_path=saige_config.step2_spa._output_glob(checkpoint_dir, phenotype.name),
                        dependencies=step2_spa_jobs_by_phenotype[phenotype.name],
                        checkpoint_dir=checkpoint_dir
                    )
                    for phenotype in phenotype_config.phenotypes
                ],
                parallelism=parallelism,
            )

            await saige_config.compile_all_results._call(
                fs,
                b,
                results_path=saige_config.compile_phenotype_results._results_path_glob(checkpoint_dir),
                output_ht_path=output_path,
                dependencies=[result.source() for result in compiled_results],
                mt_path=mt_path,
            )

            run_kwargs = run_kwargs or {}
            b.run(**run_kwargs)


def saige(
        *,
        mt_path: str,
        null_model_plink_path: str,
        output_path: str,
        phenotype_config: PhenotypeConfig,
        variant_intervals: Optional[List[hl.Interval]] = None,
        b: Optional[hb.Batch] = None,
        checkpoint_dir: Optional[str] = None,
        run_kwargs: Optional[dict] = None,
        router_fs_args: Optional[dict] = None,
        parallelism: int = 10,
        saige_config: Optional[SaigeConfig] = None,
        null_model_sample_list: Optional[str] = None,
):
    return async_to_blocking(async_saige(
        mt_path=mt_path,
        null_model_plink_path=null_model_plink_path,
        null_model_sample_list=null_model_sample_list,
        output_path=output_path,
        phenotype_config=phenotype_config,
        variant_intervals=variant_intervals,
        b=b,
        checkpoint_dir=checkpoint_dir,
        run_kwargs=run_kwargs,
        router_fs_args=router_fs_args,
        parallelism=parallelism,
        saige_config=saige_config
    ))


def compute_variant_intervals_by_contig(mt: hl.MatrixTable,
                                        max_count_per_chunk: int = 5000,
                                        max_span_per_chunk: int = 5_000_000) -> List[hl.Interval]:
    """Break up a dataset stored as a Hail Matrix Table into chunks.

    Parameters
    ----------
    mt:
        Input Hail MatrixTable object.
    max_count_per_chunk:
        Maximum number of variants in each chunk.
    max_span_per_chunk:
        Maximum number of base pairs a chunk spans.
    """
    require_row_key_variant(mt, 'saige')

    reference_genome = mt.locus.dtype.reference_genome
    variants = mt.rows()

    group_metadata = variants.aggregate(
        hl.agg.group_by(variants.locus.contig, hl.agg.approx_cdf(variants.locus.position, 200))
    )

    chunks = []

    for contig, cdf in group_metadata.items():
        cdf_values = cdf['values']
        first_rank = 0
        first_position = cdf_values[0]
        cur_position = cdf_values[0]

        for i in range(1, len(cdf_values)):
            cur_position = cdf_values[i]
            cur_rank = cdf.ranks[i]
            chunk_size = cur_rank - first_rank  # approximately how many rows are in interval [ first_position, cur_position )
            chunk_span = cur_position - first_position
            if chunk_size > max_count_per_chunk or chunk_span > max_span_per_chunk:
                interval = hl.Interval(hl.Locus(contig, first_position, reference_genome=reference_genome),
                                       hl.Locus(contig, cur_position, reference_genome=reference_genome),
                                       includes_start=True,
                                       includes_end=False)
                chunks.append(interval)
                first_rank = cur_rank
                first_position = cur_position

        interval = hl.Interval(
            hl.Locus(contig, first_position, reference_genome=reference_genome),
            hl.Locus(contig, cur_position, reference_genome=reference_genome),
            includes_start=True,
            includes_end=True,
        )
        chunks.append(interval)

    return chunks


# # FIXME: what is the group type?
# # FIXME: this should output the groups annotation file as well
# def prepare_variant_chunks_by_group(
#     mt: hl.MatrixTable, group: hl.ArrayExpression, max_count_per_chunk: int = 5000, max_span_per_chunk: int = 5_000_000
# ) -> List[VariantChunk]:
#     require_row_key_variant(mt, 'saige')
#
#     variants = mt.rows()
#
#     rg = variants.locus.dtype.reference_genome
#
#     variants = variants.select(group=group).explode('group')
#
#     group_metadata = variants.aggregate(
#         hl.agg.group_by(
#             variants.group.group,
#             hl.struct(
#                 contig=hl.array(hl.agg.collect_as_set(variants.locus.contig)),
#                 start=hl.agg.min(variants.locus.position),
#                 end=hl.agg.max(variants.locus.position),
#                 count=hl.agg.count(variants.locus),
#             ),
#         )
#     )
#
#     group_metadata = sorted(list(group_metadata.items()), key=lambda x: (x.contig, x.start))
#
#     def variant_chunk_from_groups(groups: List[hl.Struct]) -> VariantChunk:
#         contig = groups[0].contig
#         start = min(g.start for g in groups)
#         end = max(g.end for g in groups)
#         return VariantChunk(
#             hl.Interval(
#                 hl.Locus(contig, start, reference_genome=rg),
#                 hl.Locus(contig, end, reference_genome=rg),
#                 includes_start=True,
#                 includes_end=True,
#             ),
#             [g.group for g in groups],
#         )
#
#     chunks = []
#
#     groups = [group_metadata[0]]
#     current_count = 0
#
#     for group, metadata in group_metadata[1:]:
#         first_group = groups[0]
#         if (
#             (current_count + group.count > max_count_per_chunk)
#             or (group.contig != first_group.contig)
#             or (group.end - first_group.end > max_span_per_chunk)
#         ):
#             chunks.append(variant_chunk_from_groups(groups))
#             current_count = 0
#
#         groups.append(group)
#         current_count += group.count
#
#     chunks.append(variant_chunk_from_groups(groups))
#
#     return chunks


def extract_phenotypes(mt: hl.MatrixTable,
                       phenotypes: Dict[str, Union[str, hl.NumericExpression, hl.BooleanExpression]],
                       covariates: Dict[str, Union[str, hl.NumericExpression, hl.BooleanExpression]],
                       output_file: str) -> PhenotypeConfig:
    """Extract the phenotype information from an existing Hail Matrix Table.

    Examples
    --------

    Extract data from a Matrix Table:

    >>> phenotype_config = extract_phenotypes(mt,
    ...                                       {'is_case': mt.phenotypes.is_case},
    ...                                       {'pc1': mt.covariates.pc1, 'pc2': mt.covariates.pc2},
    ...                                       'gs://my-bucket/phenotypes.tsv')

    Notes
    -----

    Columns that have a BooleanExpression type are treated as binary variables and columns that
    have a NumericExpression type are treated as continuous variables.

    Parameters
    ----------
    mt:
        Hail Matrix Table object to extract phenotype data from.
    phenotypes:
        Mapping between the new phenotype name that will appear in the phenotypes file and either a Matrix Table
        column name or expression.
    covariates:
        Mapping between the new covariate name that will appear in the phenotypes file and either a Matrix Table
        column name or expression.
    output_file:
        Path to cloud storage location to write the phenotypes file to.

    Returns
    -------
    A PhenotypeConfig object that can be used as an input to SAIGE.
    """
    require_col_key_str(mt, 'saige')

    sample_id_col = list(mt.col_key)[0]

    mt = mt.select_cols(**phenotypes, **covariates)
    ht = mt.cols()
    ht.export(output_file, delimiter="\t")

    row_schema = ht.row_value.dtype
    assert isinstance(row_schema, hl.tstruct)

    saige_phenotypes = []
    saige_covariates = []
    for phenotype_name, typ in row_schema.items():
        if typ == hl.tbool:
            phenotype_type = SaigePhenotype.BINARY
        elif typ in (hl.tint, hl.tfloat):
            phenotype_type = SaigePhenotype.CONTINUOUS
        else:
            raise Exception(f'unknown SAIGE phenotype type for ({phenotype_name}, {typ})')

        phenotype = Phenotype(phenotype_name, phenotype_type)

        if phenotype_name in phenotypes.keys():
            saige_phenotypes.append(phenotype)
        else:
            assert phenotype_name in covariates.keys()
            saige_covariates.append(phenotype)

    return PhenotypeConfig(phenotypes_file=output_file,
                           sample_id_col=sample_id_col,
                           phenotypes=saige_phenotypes,
                           covariates=saige_covariates)
