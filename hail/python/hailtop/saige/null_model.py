from typing import Dict, Optional

import hailtop.batch as hb

from .config import SaigeNullModelConfig
from .constants import SaigeAnalysisType, SaigePhenotype, saige_phenotype_to_test_type
from .io import SaigeSparseGRMResource
from .utils import rectify_attributes, rectify_name


def fit_null_glmm(b: hb.Batch,
                  config: SaigeNullModelConfig,
                  *,
                  input_bfile: hb.ResourceGroup,
                  output_root: str,
                  phenotypes: hb.ResourceFile,
                  trait_type: SaigePhenotype,
                  analysis_type: SaigeAnalysisType,
                  covariates: hb.ResourceFile,
                  sparse_grm: Optional[SaigeSparseGRMResource] = None,
                  name: Optional[str] = None,
                  attributes: Optional[Dict[str, str]] = None):
    name = rectify_name(config.name, name)

    attributes = rectify_attributes(config.attributes, attributes)
    attributes.update(analysis_type=analysis_type.value,
                      trait_type=trait_type.value)

    fit_null_job = (b.new_job(name=name, attributes=attributes)
                     .storage(config.storage)
                     .image(config.docker_image)
                    )

    fit_null_job.cpu(config.n_threads)
    fit_null_job.memory(config.memory)
    fit_null_job.spot(config.spot)

    output_files = {
        'rda': '{root}.rda',
        '_30markers.SAIGE.results.txt': '{root}_30markers.SAIGE.results.txt',
        f'{analysis_type}.varianceRatio.txt': f'{{root}}.{analysis_type}.varianceRatio.txt',
    }

    if analysis_type == SaigeAnalysisType.GENE:
        sparse_sigma_extension = sparse_grm.mtx_identifier.replace("GRM", "Sigma")
        suffix = f'{analysis_type}.varianceRatio.txt{sparse_sigma_extension}'
        additional_gene_output_files = {
            suffix: f'{{root}}{suffix}'
        }
        output_files.update(additional_gene_output_files)

    fit_null_job.declare_resource_group(null_glmm=output_files)

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
    --phenoCol={config.pheno_col} \
    --sampleIDColinphenoFile={config.user_id_col} \
    --traitType={test_type.value} \
    --outputPrefix={fit_null_job.null_glm} \
    --outputPrefix_varRatio={fit_null_job.null_glm}.{analysis_type.value} \
    --skipModelFitting={skip_model_fitting_str} \
    {inv_normalize_flag} \
    {gene_flags} \
    --nThreads={config.n_threads} \
    --LOCO=FALSE 2>&1 | tee {fit_null_job.stdout}  
'''

    fit_null_job.command(command)

    b.write_output(fit_null_job.null_glmm, output_root)
    b.write_output(fit_null_job.stdout, f'{output_root}.{analysis_type.value}.log')

    return fit_null_job
