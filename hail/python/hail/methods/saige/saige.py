from typing import Dict, List, Optional, Union

import hailtop.batch as hb
import hailtop.batch_client as bc
import hailtop.fs as hfs

from ...backend.service_backend import ServiceBackend
from ...context import TemporaryFilename
from ...matrixtable import MatrixTable
from ...expr.types import tbool, tint, tint32, tint64, tfloat, tfloat32, tfloat64
from ...expr.aggregators import collect_as_set
from ..impex import export_bgen, export_plink, read_matrix_table

from .config import BaseConfig, NullGlmmModelConfig, SaigeConfig, SparseGRMConfig
from .constants import SaigeAnalysisType, SaigePhenotype, SaigeTestType, saige_phenotype_to_test_type
from .io import BgenResourceGroup, PlinkResourceGroup, SaigeGlmmResourceGroup, SaigeResultResourceGroup, SaigeSparseGRMResourceGroup, VCFResourceGroup


def convert_mt_to_plink(b: hb.Batch, config: BaseConfig, mt: MatrixTable, output_file: str) -> PlinkResourceGroup:
    checkpointed_output = PlinkResourceGroup.use_checkpoint_if_exists(b, config, output_file)
    if checkpointed_output:
        return checkpointed_output
    export_plink(mt, output_file)
    return PlinkResourceGroup.from_input_files(b, output_file)


def create_sparse_grm(b: hb.Batch,
                      config: SparseGRMConfig,
                      input_bfile: PlinkResourceGroup,
                      output_file: str,
                      ) -> SaigeSparseGRMResourceGroup:
    checkpointed_output = SaigeSparseGRMResourceGroup.use_checkpoint_if_exists(b, config, output_file)
    if checkpointed_output:
        return checkpointed_output

    create_sparse_grm_task = b.new_job(name=config.name, attributes=config.attributes)

    (create_sparse_grm_task
     .cpu(config.cpu)
     .storage(config.storage)
     .image(config.image)
     )

    relatedness_cutoff = config.relatedness_cutoff
    num_markers = config.num_markers

    sparse_grm_output = SaigeSparseGRMResourceGroup.from_job_intermediate(
        create_sparse_grm_task, relatedness_cutoff, num_markers
    )

    command = f'''
Rscript /usr/local/bin/createSparseGRM.R \
    --plinkFile={input_bfile} \
    --nThreads={config.cpu} \
    --outputPrefix={sparse_grm_output} \
    --numRandomMarkerforSparseKin={config.num_markers} \
    --relatednessCutoff={relatedness_cutoff}
'''

    create_sparse_grm_task.command(command)

    sparse_grm_output.checkpoint(b, config, output_file)

    return sparse_grm_output


def fit_null_glmm(b: hb.Batch,
                  config: NullGlmmModelConfig,
                  *,
                  input_bfile: PlinkResourceGroup,
                  output_root: str,
                  phenotypes: hb.ResourceFile,
                  trait_type: SaigePhenotype,
                  analysis_type: SaigeAnalysisType,
                  covariates: List[str],
                  pheno_col: str,
                  sparse_grm: Optional[SaigeSparseGRMResourceGroup] = None) -> SaigeGlmmResourceGroup:
    glmm_resource_output = SaigeGlmmResourceGroup.use_checkpoint_if_exists(b, config, output_root, analysis_type, sparse_grm)
    if glmm_resource_output:
        return glmm_resource_output

    name = config.name_with_pheno(pheno_col)
    attributes = config.attributes_with_pheno(pheno_col, analysis_type.value, trait_type.value)

    fit_null_job = (b.new_job(name=name, attributes=attributes)
                     .storage(config.storage)
                     .image(config.image)
                    )

    fit_null_job.cpu(config.cpu)
    fit_null_job.memory(config.memory)
    fit_null_job.spot(config.spot)

    null_glmm = SaigeGlmmResourceGroup.from_job_intermediate(fit_null_job, analysis_type, sparse_grm)

    test_type = saige_phenotype_to_test_type[trait_type]

    if config.inv_normalize:
        inv_normalize_flag = '--invNormalize=TRUE'
    else:
        inv_normalize_flag = ''

    if analysis_type == SaigeAnalysisType.GENE:
        assert sparse_grm is not None
        gene_flags = f''' \
    --IsSparseKin=TRUE \
    --sparseGRMFile={sparse_grm.sparse_grm} \
    --sparseGRMSampleIDFile={sparse_grm.sample_ids} \
    --isCateVarianceRatio=TRUE'''
    else:
        gene_flags = ''

    skip_model_fitting_str = str(config.skip_model_fitting).upper()

    command = f'''
set -o pipefail;

perl -pi -e s/^chr// {input_bfile.bim};

Rscript /usr/local/bin/step1_fitNULLGLMM.R \
    --plinkFile={input_bfile} \
    --phenoFile={phenotypes} \
    --covarColList={covariates} \
    --minCovariateCount={config.min_covariate_count} \
    --phenoCol={pheno_col} \
    --sampleIDColinphenoFile={config.user_id_col} \
    --traitType={test_type.value} \
    --outputPrefix={null_glmm} \
    --outputPrefix_varRatio={fit_null_job.null_glmm}.{analysis_type.value} \
    --skipModelFitting={skip_model_fitting_str} \
    {inv_normalize_flag} \
    {gene_flags} \
    --nThreads={config.cpu} \
    --LOCO=FALSE 2>&1 | tee {fit_null_job.stdout}  
'''

    fit_null_job.command(command)

    null_glmm.checkpoint(b, config, output_root)
    b.write_output(fit_null_job.stdout, f'{output_root}.{analysis_type.value}.log')

    return fit_null_job


def run_saige(b: hb.Batch,
              config: SaigeConfig,
              output_root: str,
              null_model: SaigeGlmmResourceGroup,
              input_data: Union[BgenResourceGroup, VCFResourceGroup],
              analysis_type: SaigeAnalysisType,
              trait_type: SaigePhenotype) -> SaigeResultResourceGroup:

    saige_result_output = SaigeResultResourceGroup.use_checkpoint_if_exists(b, config, output_root, analysis_type)
    if saige_result_output:
        return saige_result_output

    run_saige_task = b.new_job(name=config.name(), attributes=config.attributes(analysis_type))
    (run_saige_task.cpu(config.cpu)
     .storage(config.storage)
     .image(config.image)
     )

    saige_result = SaigeResultResourceGroup.from_job_intermediate(j, analysis_type)

    if config.mkl_off:
        mkl_off = 'export MKL_NUM_THREADS=1; export MKL_DYNAMIC=false; export OMP_NUM_THREADS=1; export OMP_DYNAMIC=false; '
    else:
        mkl_off = ''

    command = f'''
set -o pipefail; {mkl_off} \
Rscript /usr/local/bin/step2_SPAtests.R \
    --minMAF={config.min_maf} \
    --minMAC={config.min_mac} \
    --maxMAFforGroupTest={config.max_maf} \
    --sampleFile={???} \
    --GMMATmodelFile={???} \
    --varianceRatioFile={null_model.variance_ratio} \
    --SAIGEOutputFile={saige_result}'''

    if isinstance(input_data, BgenResourceGroup):
        command += f'--bgenFile={input_data.bgen} --bgenFileIndex={input_data.bgen_idx}'
    else:
        assert isinstance(input_data, VCFResourceGroup)
        command += (f'--vcfFile={input_data.vcf} '
                    f'--vcfFileIndex={input_data.tbi} '
                    f'--chrom={chrom} '
                    f'--vcfField=GT ')

    if analysis_type == SaigeAnalysisType.GENE:
        test_type = saige_phenotype_to_test_type[trait_type]
        if test_type == SaigeTestType.BINARY:
            command += f'--IsOutputPvalueNAinGroupTestforBinary=TRUE '
        else:
            command += (f'--groupFile={group_file} '
                        f'--sparseSigmaFile={sparse_sigma_file} '
                        f'--IsSingleVarinGroupTest=TRUE '
                        f'--IsOutputBETASEinBurdenTest=TRUE ')
    command += f'--IsOutputAFinCaseCtrl=TRUE 2>&1 | tee {run_saige_task.stdout}; '

    if analysis_type == SaigeAnalysisType.GENE:
        command += f"input_length=$(wc -l {group_file} | awk '{{print $1}}'); " \
            f"output_length=$(wc -l {run_saige_task.result['gene.txt']} | awk '{{print $1}}'); " \
            f"echo 'Got input:' $input_length 'output:' $output_length | tee -a {run_saige_task.stdout}; " \
            f"if [[ $input_length > 0 ]]; then echo 'got input' | tee -a {run_saige_task.stdout}; " \
            f"if [[ $output_length == 1 ]]; then echo 'but not enough output' | tee -a {run_saige_task.stdout}; " \
                   f"rm -f {run_saige_task.result['gene.txt']} exit 1; fi; fi"

    run_saige_task.command(command)

    saige_result.checkpoint(b, config, output_root)
    b.write_output(run_saige_task.stdout, f'{output_root}.{analysis_type}.log')

    return saige_result


def _saige_batch(config: BaseConfig,
                 b: hb.Batch,
                 mt: MatrixTable,
                 grouping: str,
                 phenotypes: List[str],
                 covariates: List[str]):
    pheno_columns = phenotypes + covariates
    mt.cols().select(pheno_columns).export(config.pheno_file, delimiter="\t")
    pheno = b.read_input(config.pheno_file)

    input_data = convert_mt_to_plink(b, config, mt, config.plink_bfile)

    partitions = mt.aggregate_rows(collect_as_set(mt[grouping]))
    mt.write(grouped_mt_fp, _partitions=partitions)
    data_chunks = [fp.path for fp in hfs.ls(grouped_mt_fp) if 'part' in fp.path]

    if config.analysis_type == SaigeAnalysisType.GENE:
        sparse_grm = create_sparse_grm(b, config.sparse_grm_config, input_data, config.sparse_grm_config.output_root)
    else:
        sparse_grm = None

    for phenotype in phenotypes:
        hl_pheno = mt[phenotype]
        if hl_pheno.dtype == tbool:
            trait_type = SaigePhenotype.CATEGORICAL
        else:
            assert hl_pheno.dtype in (tfloat, tfloat32, tfloat64, tint, tint32, tint64)
            trait_type = SaigePhenotype.CONTINUOUS

        null_glmm = fit_null_glmm(b,
                                  config.null_glmm_config,
                                  input_bfile=input_data,
                                  output_root=config.output_file,
                                  phenotypes=pheno,
                                  trait_type=trait_type,
                                  analysis_type=config.analysis_type,
                                  covariates=covariates,
                                  pheno_col=phenotype,
                                  sparse_grm=sparse_grm)

        for chunk in data_chunks:
            run_saige(b, config, chunk, ...)

        b.run()


def saige(mt: MatrixTable, grouping: str, phenotypes: List[str], covariates: List[str]):
    return _saige_batch(...)

