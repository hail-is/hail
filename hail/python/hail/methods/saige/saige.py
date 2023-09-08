from typing import List, Optional

import hail as hl
import hailtop.batch as hb

from .config import NullGlmmModelConfig, RunSaigeConfig, SaigeConfig, SparseGRMConfig
from .constants import SaigeAnalysisType, SaigeInputDataType, SaigePhenotype, SaigeTestType, saige_phenotype_to_test_type
from .io import (
    BgenResourceGroup,
    PlinkResourceGroup,
    SaigeGlmmResourceGroup,
    SaigeResultResourceGroup,
    SaigeSparseGRMResourceGroup,
    TextFile,
    VCFResourceGroup
)


def write_phenotype_data(b: hb.Batch, config: SaigeConfig, mt: hl.MatrixTable, output_file: str, pheno_cols: List[str]) -> TextFile:
    checkpointed_output = TextFile.use_checkpoint_if_exists(b, config, output_file)
    if checkpointed_output and not config.overwrite:
        return checkpointed_output
    mt.cols().select(pheno_cols).export(output_file, delimiter="\t")
    return TextFile.from_input_file(b, output_file)


def convert_mt_to_plink(b: hb.Batch, config: SaigeConfig, mt: hl.MatrixTable, output_file: str) -> PlinkResourceGroup:
    checkpointed_output = PlinkResourceGroup.use_checkpoint_if_exists(b, config, output_file)
    if checkpointed_output and not config.overwrite:
        return checkpointed_output
    hl.export_plink(mt, output_file)
    return PlinkResourceGroup.from_input_files(b, output_file)


def find_chunks(mt: hl.MatrixTable, chunk_col: str) -> List[str]:
    return mt.aggregate_rows(hl.agg.collect_as_set(hl.str(mt[chunk_col])))


def format_groups(mt: hl.MatrixTable, input_data_type: SaigeInputDataType, group_col: str) -> hl.MatrixTable:
    # FIXME: are variants required to be biallelic???
    ann = mt.annotate_rows(group_name=mt[group_col])
    if input_data_type == SaigeInputDataType.VCF:
        ann = mt.annotate_rows(variant_name=ann.locus.contig + ':' + hl.str(ann.locus.position) + '_' + ann.alleles[0] + '/' + ann.alleles[1])
    else:
        if 'varid' in mt.col:
            ann = ann.annotate_rows(variant_name=ann['varid'])
        else:
            ann = ann.annotate_rows(variant_name=hl.delimit([ann.locus.contig, hl.str(ann.locus.position), ann.alleles[0], ann.alleles[1]], ':'))
    return ann


def create_sparse_grm(b: hb.Batch,
                      config: SparseGRMConfig,
                      input_bfile: PlinkResourceGroup,
                      output_file: str,
                      ) -> SaigeSparseGRMResourceGroup:
    checkpointed_output = SaigeSparseGRMResourceGroup.use_checkpoint_if_exists(b, config, output_file)
    if checkpointed_output and not config.overwrite:
        return checkpointed_output

    create_sparse_grm_task = b.new_job(name=config.name(), attributes=config.attributes())

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
createSparseGRM.R \
    --plinkFile={input_bfile} \
    --nThreads={config.cpu} \
    --outputPrefix={sparse_grm_output} \
    --numRandomMarkerforSparseKin={config.num_markers} \
    --relatednessCutoff={relatedness_cutoff}
'''

    create_sparse_grm_task.command(command)

    sparse_grm_output.checkpoint_if_requested(b, config, output_file)

    return sparse_grm_output


def fit_null_glmm(b: hb.Batch,
                  config: NullGlmmModelConfig,
                  *,
                  input_bfile: PlinkResourceGroup,
                  output_root: str,
                  phenotypes: TextFile,
                  trait_type: SaigePhenotype,
                  analysis_type: SaigeAnalysisType,
                  covariates: List[str],
                  pheno_col: str,
                  sparse_grm: Optional[SaigeSparseGRMResourceGroup] = None) -> SaigeGlmmResourceGroup:
    glmm_resource_output = SaigeGlmmResourceGroup.use_checkpoint_if_exists(b, config, output_root, analysis_type, sparse_grm)
    if glmm_resource_output and not config.overwrite:
        return glmm_resource_output

    fit_null_job = (b.new_job(name=config.name(phenotype=pheno_col),
                              attributes=config.attributes(analysis_type=analysis_type.value, trait_type=trait_type.value))
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
        gene_flags = f'''--IsSparseKin=TRUE \
    --sparseGRMFile={sparse_grm.sparse_grm} \
    --sparseGRMSampleIDFile={sparse_grm.sample_ids} \
    --isCateVarianceRatio=TRUE'''
    else:
        gene_flags = ''

    skip_model_fitting_str = str(config.skip_model_fitting).upper()

    command = f'''
set -o pipefail;

perl -pi -e s/^chr// {input_bfile.bim};

step1_fitNULLGLMM.R \
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

    null_glmm.checkpoint_if_requested(b, config, output_root)
    b.write_output(fit_null_job.stdout, f'{output_root}.{analysis_type.value}.log')

    return fit_null_job


def run_saige(b: hb.Batch,
              config: RunSaigeConfig,
              checkpointed_mt: str,
              output_dir: str,
              phenotype: str,
              chunk: str,
              null_model: SaigeGlmmResourceGroup,
              sparse_grm: Optional[SaigeSparseGRMResourceGroup],
              analysis_type: SaigeAnalysisType,
              trait_type: SaigePhenotype) -> SaigeResultResourceGroup:
    output_root = config.output_root(output_dir=output_dir, phenotype=phenotype, chunk=chunk)

    saige_result_output = SaigeResultResourceGroup.use_checkpoint_if_exists(b, config, output_root, analysis_type)
    if saige_result_output and not config.overwrite:
        return saige_result_output

    run_saige_task = b.new_job(name=config.name(phenotype=phenotype, group=chunk),
                               attributes=config.attributes(analysis_type=analysis_type))

    (run_saige_task.cpu(config.cpu)
     .storage(config.storage)
     .image(config.image)
     )

    saige_result = SaigeResultResourceGroup.from_job_intermediate(run_saige_task, analysis_type)

    if config.mkl_off:
        mkl_off = 'export MKL_NUM_THREADS=1; export MKL_DYNAMIC=false; export OMP_NUM_THREADS=1; export OMP_DYNAMIC=false; '
    else:
        mkl_off = ''

    if config.input_data_type == SaigeInputDataType:
        input = VCFResourceGroup.from_job_intermediate(run_saige_task)
        export_cmd = f'hl.export_vcf(mt, {input})'
    else:
        input = BgenResourceGroup.from_job_intermediate(run_saige_task)
        export_cmd = f'hl.export_bgen(mt, {input})'

    if analysis_type == SaigeAnalysisType.GENE:
        group_file_cmd = f'''
rows = mt.rows()
groups = rows.group_by(rows["group_name"]).aggregate(variants=hl.str('\t').join(hl.agg.collect_as_set(rows["variant_name"])))
groups.export("{run_saige_task.groups}", header=False)
'''
    else:
        group_file_cmd = ''

    hail_io_command = f'''
cat > read_from_mt.py <<EOF
import hail as hl
mt = hl.read_matrix_table({checkpointed_mt})
mt = mt.filter(mt.chunk == {chunk})
{export_cmd}
{group_file_cmd} 
EOF
python3 read_from_mt.py
'''

    if isinstance(input, BgenResourceGroup):
        input_flags = [
            f'--bgenFile={input.bgen}',
            f'--bgenFileIndex={input.bgen_idx}',
            f'--sampleFile={input.sample}',
        ]
    else:
        assert isinstance(input, VCFResourceGroup)
        input_flags = [
            f'--vcfFile={input.vcf}',
            f'--vcfFileIndex={input.tbi}',
            f'--vcfField=GT',
        ]

    saige_options = [
        f'--minMAF={config.min_maf}',
        f'--minMAC={config.min_mac}',
        f'--maxMAFforGroupTest={config.max_maf_for_group_test}',
        f'--GMMATmodelFile={null_model.rda}',
        f'--varianceRatioFile={null_model.variance_ratio}',
        f'--SAIGEOutputFile={saige_result}',
        f'--numLinesOutput={config.num_lines_output}',
        f'--IsSparse={str(config.is_sparse).upper()}',
        f'--SPAcutoff={config.spa_cutoff}',
        f'--IsOutputAFinCaseCtrl={str(config.output_af_in_case_control).upper()}',
        f'--IsOutputNinCaseCtrl={str(config.output_n_in_case_control).upper()}',
        f'--IsOutputHetHomCountsinCaseCtrl={str(config.output_het_hom_counts).upper()}',
        f'--IsOutputAFinCaseCtrl={str(config.output_af_in_case_control).upper()}',
        f'--IsOutputMAFinCaseCtrlinGroupTest={str(config.output_maf_in_case_control_in_group_test).upper()}',
        f'--IsOutputlogPforSingle={str(config.output_logp_for_single).upper()}',
    ]

    if config.kernel is not None:
        saige_options.append(f'--kernel={config.kernel}')
    if config.method is not None:
        saige_options.append(f'--method={config.method}')
    if config.weights_beta_rare is not None:
        saige_options.append(f'--weights.beta.rare={config.weights_beta_rare}')
    if config.weights_beta_common is not None:
        saige_options.append(f'--weights.beta.common={config.weights_beta_common}')
    if config.weight_maf_cutoff is not None:
        saige_options.append(f'--weightMAFcutoff={config.weight_maf_cutoff}')
    if config.r_corr is not None:
        saige_options.append(f'--r.corr={config.r_corr}')
    if config.cate_var_ratio_min_mac_vec_exclude is not None:
        exclude = ','.join(str(val) for val in config.cate_var_ratio_min_mac_vec_exclude)
        saige_options.append(f'--cateVarRatioMinMACVecExclude={exclude}')
    if config.cate_var_ratio_max_mac_vec_include is not None:
        include = ','.join(str(val) for val in config.cate_var_ratio_max_mac_vec_include)
        saige_options.append(f'--cateVarRatioMaxMACVecInclude={include}')
    if config.dosage_zerod_cutoff is not None:
        saige_options.append(f'--dosageZerodCutoff={config.dosage_zerod_cutoff}')
    if config.output_pvalue_na_in_group_test_for_binary is not None:
        saige_options.append(f'--IsOutputPvalueNAinGroupTestforBinary={str(config.output_pvalue_na_in_group_test_for_binary).upper()}')
    if config.account_for_case_control_imbalance_in_group_test is not None:
        saige_options.append(f'--IsAccountforCasecontrolImbalanceinGroupTest={str(config.account_for_case_control_imbalance_in_group_test).upper()}')
    if config.x_par_region is not None:
        regions = ','.join(config.x_par_region)
        saige_options.append(f'--X_PARregion={regions}')
    if config.rewrite_x_nonpar_for_males is not None:
        saige_options.append(f'--is_rewrite_XnonPAR_forMales={str(config.rewrite_x_nonpar_for_males).upper()}')
    if config.method_to_collapse_ultra_rare is not None:
        saige_options.append(f'--method_to_CollapseUltraRare={config.method_to_collapse_ultra_rare}')
    if config.mac_cutoff_to_collapse_ultra_rare is not None:
        saige_options.append(f'--MACCutoff_to_CollapseUltraRare={config.mac_cutoff_to_collapse_ultra_rare}')
    if config.dosage_cutoff_for_ultra_rare_presence is not None:
        saige_options.append(f'--DosageCutoff_for_UltraRarePresence={config.dosage_cutoff_for_ultra_rare_presence}')

    # FIXME: --weightsIncludeinGroupFile, --weights_for_G2_cond, --sampleFile_male

    gene_options = []

    if analysis_type == SaigeAnalysisType.GENE:
        test_type = saige_phenotype_to_test_type[trait_type]
        if test_type == SaigeTestType.BINARY:
            gene_options = [f'--IsOutputPvalueNAinGroupTestforBinary={str(config.output_pvalue_na_in_group_test_for_binary).upper()}']
        else:
            gene_options = [
                f'--groupFile={run_saige_task.groups}',
                f'--sparseSigmaFile={sparse_grm.grm}',
                f'--IsSingleVarinGroupTest={str(config.single_variant_in_group_test).upper()}',
                f'--IsOutputBETASEinBurdenTest={str(config.output_beta_se_in_burden_test).upper()}',
            ]

    input_flags = '\n'.join(f'    {flag} \\ ' for flag in input_flags)
    saige_options = '\n'.join(f'    {opt} \\ ' for opt in saige_options)
    gene_options = '\n'.join(f'    {opt} \\ ' for opt in gene_options)

    command = f'''
set -o pipefail;
{mkl_off}

{hail_io_command}

step2_SPAtests.R \
{input_flags}
{saige_options}
{gene_options} 2>&1 | tee {run_saige_task.stdout};
'''

    if analysis_type == SaigeAnalysisType.GENE:
        command += f'''
input_length=$(wc -l {run_saige_task.groups} | awk '{{print $1}}')
output_length=$(wc -l {run_saige_task.gene} | awk '{{print $1}}')
echo 'Got input:' $input_length 'output:' $output_length | tee -a {run_saige_task.stdout}
if [[ $input_length > 0 ]] then
    echo 'got input' | tee -a {run_saige_task.stdout};
    if [[ $output_length == 1 ]] then
        echo 'but not enough output' | tee -a {run_saige_task.stdout};
        exit 1; 
    fi;
fi;
'''

    run_saige_task.command(command)

    saige_result.checkpoint_if_requested(b, config, output_root)
    b.write_output(run_saige_task.stdout, f'{output_root}.{analysis_type}.log')

    return saige_result


def _saige_batch(config: SaigeConfig,
                 b: hb.Batch,
                 mt: hl.MatrixTable,
                 *,
                 chunk_col: str,
                 group_col: Optional[str],
                 phenotypes: List[str],
                 covariates: List[str],
                 analysis_type: SaigeAnalysisType):
    pheno_columns = phenotypes + covariates

    with config:
        phenotypes = write_phenotype_data(b, config, mt, config.pheno_file, pheno_columns)
        input_plink_data = convert_mt_to_plink(b, config, mt, config.input_plink_data)
        chunks = find_chunks(mt, chunk_col)

        if analysis_type == SaigeAnalysisType.GENE:
            mt = format_groups(mt, config.run_saige_config.input_data_type, group_col)
            sparse_grm = create_sparse_grm(b, config.sparse_grm_config, input_plink_data, config.sparse_grm)
        else:
            sparse_grm = None

        mt.checkpoint(config.ann_genomic_data, overwrite=config.overwrite)

        for phenotype in pheno_columns:
            hl_pheno = mt[phenotype]
            if hl_pheno.dtype == hl.tbool:
                trait_type = SaigePhenotype.CATEGORICAL
            else:
                assert hl_pheno.dtype in (hl.tfloat, hl.tfloat32, hl.tfloat64, hl.tint, hl.tint32, hl.tint64)
                trait_type = SaigePhenotype.CONTINUOUS

            null_glmm = fit_null_glmm(b,
                                      config.null_glmm_config,
                                      input_bfile=input_plink_data,
                                      output_root=config.null_glmm_config.output_root(config.null_model_dir, phenotype),
                                      phenotypes=phenotypes,
                                      trait_type=trait_type,
                                      analysis_type=analysis_type,
                                      covariates=covariates,
                                      pheno_col=phenotype,
                                      sparse_grm=sparse_grm)

            for chunk in chunks:
                run_saige(b,
                          config.run_saige_config,
                          config.ann_genomic_data,
                          config.output_dir,
                          phenotype,
                          chunk,
                          null_glmm,
                          sparse_grm,
                          analysis_type,
                          trait_type)

        b.run()


def saige(mt: hl.MatrixTable, grouping: str, phenotypes: List[str], covariates: List[str]):
    return _saige_batch(...)

