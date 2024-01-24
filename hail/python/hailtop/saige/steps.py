from dataclasses import dataclass
import json
from typing import Dict, List, Optional, Tuple, Union

import hail as hl
from hailtop.aiotools.fs import AsyncFS
import hailtop.batch as hb

from .config import CheckpointConfigMixin, JobConfigMixin
from .constants import SaigeAnalysisType, SaigeInputDataType, SaigeTestType, saige_phenotype_to_test_type
from .io import (
    PlinkResourceGroup,
    TextResourceFile,
    SaigeGeneGLMMResourceGroup,
    SaigeGLMMResourceGroup,
    SaigeSparseGRMResourceGroup,
    SaigeGeneResultResourceGroup,
    SaigeResultResourceGroup,
    checkpoint_if_requested,
    load_saige_glmm_file,
    load_saige_result_file,
    load_saige_sparse_grm_file,
    load_text_file,
    new_saige_glmm_file,
    new_saige_result_file,
    new_saige_sparse_grm_file,
    new_text_file,
)
from .phenotype import Phenotype
from .variant_chunk import VariantChunk


def get_output_dir(config: CheckpointConfigMixin, temp_dir: str, checkpoint_dir: Optional[str]) -> str:
    if config.use_checkpoints or config.checkpoint_output:
        if checkpoint_dir is None:
            raise ValueError('must specify a checkpoint directory to use checkpoints and/or checkpoint output')
        return checkpoint_dir
    return temp_dir


@dataclass
class SparseGRMStep(CheckpointConfigMixin, JobConfigMixin):
    relatedness_cutoff: float = ...
    num_markers: int = ...
    cpu: Union[str, float, int] = 1
    memory: str = 'highmem'
    storage: str = '10Gi'
    spot: bool = True

    def name(self) -> str:
        return 'sparse-grm'

    def output_root(self, temp_dir: str, checkpoint_dir: Optional[str]) -> str:
        working_dir = get_output_dir(self, temp_dir, checkpoint_dir)
        return f'{working_dir}/sparse-grm'

    def attributes(self) -> Optional[Dict]:
        return None

    async def call(self, fs: AsyncFS, b: hb.Batch, input_bfile: PlinkResourceGroup, temp_dir: str, checkpoint_dir: str):
        output_prefix = self.output_root(temp_dir, checkpoint_dir)
        sparse_grm = await load_saige_sparse_grm_file(fs, b, self, output_prefix)
        if sparse_grm is not None:
            return sparse_grm

        create_sparse_grm_j = b.new_job(name=self.name(), attributes=self.attributes())

        (create_sparse_grm_j.cpu(self.cpu).storage(self.storage).image(self.image).spot(self.spot))

        sparse_grm = new_saige_sparse_grm_file(create_sparse_grm_j, self.relatedness_cutoff, self.num_markers)

        command = f'''
createSparseGRM.R \
    --plinkFile={input_bfile} \
    --nThreads={self.cpu} \
    --outputPrefix={sparse_grm} \
    --numRandomMarkerforSparseKin={self.num_markers} \
    --relatednessCutoff={self.relatedness_cutoff}
'''

        create_sparse_grm_j.command(command)

        checkpoint_if_requested(sparse_grm, b, self, output_prefix)

        return sparse_grm


@dataclass
class Step1NullGlmmStep(CheckpointConfigMixin, JobConfigMixin):
    inv_normalize: bool = True
    skip_model_fitting: bool = True
    min_covariate_count: int = 5
    save_stdout: bool = True

    def output_root(self, output_dir: str, phenotype: Phenotype) -> str:
        return f'{output_dir}/null-model-{phenotype.name}'

    def name(self, phenotype: Phenotype) -> str:
        return f'{phenotype.name}-null-model'

    def attributes(self, analysis_type: SaigeAnalysisType, phenotype: Phenotype) -> Optional[Dict]:
        return {'analysis_type': analysis_type.value, 'trait_type': phenotype.phenotype_type.value}

    async def call(
        self,
        fs: AsyncFS,
        b: hb.Batch,
        *,
        input_bfile: PlinkResourceGroup,
        input_phenotypes: TextResourceFile,
        phenotype: Phenotype,
        analysis_type: SaigeAnalysisType,
        covariates: List[str],
        user_id_col: str,
        temp_dir: str,
        checkpoint_dir: Optional[str],
    ) -> SaigeGLMMResourceGroup:
        working_dir = get_output_dir(self, temp_dir, checkpoint_dir)
        output_root = self.output_root(working_dir, phenotype)

        glmm_resource_output = await load_saige_glmm_file(fs, b, self, output_root, analysis_type)
        if glmm_resource_output:
            return glmm_resource_output

        j = (
            b.new_job(
                name=self.name(phenotype=phenotype),
                attributes=self.attributes(analysis_type, phenotype),
            )
            .storage(self.storage)
            .image(self.image)
            .cpu(self.cpu)
            .memory(self.memory)
            .spot(self.spot)
        )

        null_glmm = new_saige_glmm_file(j, analysis_type)

        command = self.command(
            input_bfile,
            input_phenotypes,
            covariates,
            user_id_col,
            phenotype,
            analysis_type,
            null_glmm,
            j.stdout,
        )

        j.command(command)

        checkpoint_if_requested(null_glmm, b, self, output_root)

        if self.save_stdout:
            b.write_output(j.stdout, f'{output_root}.log')

        return j

    def command(
        self,
        input_bfile: PlinkResourceGroup,
        phenotypes: TextResourceFile,
        covariates: List[str],
        user_id_col: str,
        phenotype: Phenotype,
        analysis_type: SaigeAnalysisType,
        null_glmm: SaigeGLMMResourceGroup,
        stdout: TextResourceFile,
    ) -> str:
        test_type = saige_phenotype_to_test_type[phenotype.phenotype_type]

        if self.inv_normalize:
            inv_normalize_flag = '--invNormalize=TRUE'
        else:
            inv_normalize_flag = ''

        skip_model_fitting_str = str(self.skip_model_fitting).upper()

        command = f'''
set -o pipefail;

perl -pi -e s/^chr// {input_bfile.bim};

step1_fitNULLGLMM.R \
    --plinkFile={input_bfile} \
    --phenoFile={phenotypes} \
    --covarColList={covariates} \
    --minCovariateCount={self.min_covariate_count} \
    --phenoCol={phenotype} \
    --sampleIDColinphenoFile={user_id_col} \
    --traitType={test_type.value} \
    --outputPrefix={null_glmm} \
    --outputPrefix_varRatio={null_glmm}.{analysis_type.value} \
    --skipModelFitting={skip_model_fitting_str} \
    {inv_normalize_flag} \
    --nThreads={self.cpu} \
    --LOCO=FALSE 2>&1 | tee {stdout}
'''
        return command


@dataclass
class Step2SPAStep(CheckpointConfigMixin, JobConfigMixin):
    save_stdout: bool = True
    mkl_off: bool = False
    drop_missing_dosages: bool = ...
    min_mac: float = 0.5  # FIXME
    min_maf: float = 0
    max_maf_for_group_test: float = 0.5
    min_info: float = 0
    num_lines_output: int = 10000
    is_sparse: bool = True
    spa_cutoff: float = 2.0
    output_af_in_case_control: bool = False
    output_n_in_case_control: bool = False
    output_het_hom_counts: bool = False
    kernel: Optional[str] = ...
    method: Optional[str] = ...
    weights_beta_rare: Optional[float] = ...
    weights_beta_common: Optional[float] = ...
    weight_maf_cutoff: Optional[float] = ...
    r_corr: Optional[float] = ...
    single_variant_in_group_test: bool = False
    output_maf_in_case_control_in_group_test: bool = False
    cate_var_ratio_min_mac_vec_exclude: Optional[List[float]] = None
    cate_var_ratio_max_mac_vec_include: Optional[List[float]] = None
    dosage_zerod_cutoff: float = 0.2
    output_pvalue_na_in_group_test_for_binary: bool = False
    account_for_case_control_imbalance_in_group_test: bool = True
    weights_include_in_group_file: bool = False  # fixme with weight
    weights_for_g2_cond: Optional[List[int]] = None
    output_beta_se_in_burden_test: bool = False
    output_logp_for_single: bool = False
    x_par_region: Optional[List[str]] = None
    rewrite_x_nonpar_for_males: bool = False
    method_to_collapse_ultra_rare: str = 'absence_or_presence'
    mac_cutoff_to_collapse_ultra_rare: float = 10
    dosage_cutoff_for_ultra_rare_presence: float = 0.5

    def name(self, phenotype: Phenotype, chunk: VariantChunk) -> str:
        return f'{phenotype.name}-{chunk.idx}'

    def attributes(
        self, *, analysis_type: SaigeAnalysisType, phenotype: Phenotype, chunk: VariantChunk
    ) -> Optional[Dict]:
        return {
            'analysis_type': analysis_type.value,
            'trait_type': phenotype.phenotype_type.value,
            'phenotype': phenotype.name,
            'chunk': chunk.to_interval_str(),
        }

    def output_file_prefix(
        self, temp_dir: str, checkpoint_dir: Optional[str], phenotype_name: str, chunk: VariantChunk
    ) -> str:
        working_dir = get_output_dir(self, temp_dir, checkpoint_dir)
        return f'{working_dir}/results/{phenotype_name}/{chunk.idx}'

    def output_glob(self, temp_dir: str, checkpoint_dir: Optional[str], phenotype_name: str) -> str:
        working_dir = get_output_dir(self, temp_dir, checkpoint_dir)
        return f'{working_dir}/results/{phenotype_name}/*'

    def command(
        self,
        *,
        mt_path: str,
        analysis_type: SaigeAnalysisType,
        null_model: Union[SaigeGeneGLMMResourceGroup, SaigeGLMMResourceGroup],
        input_data_type: SaigeInputDataType,
        chunk: VariantChunk,
        phenotype: Phenotype,
        result: SaigeResultResourceGroup,
        stdout: TextResourceFile,
        sparse_grm: Optional[SaigeSparseGRMResourceGroup],
        group_annotations: Optional[TextResourceFile],
    ):
        if self.mkl_off:
            mkl_off = 'export MKL_NUM_THREADS=1; export MKL_DYNAMIC=false; export OMP_NUM_THREADS=1; export OMP_DYNAMIC=false; '
        else:
            mkl_off = ''

        if input_data_type == SaigeInputDataType.VCF:
            export_cmd = f'hl.export_vcf(mt.filter_entries("GT"), "/data.vcf.bgz")'
            input_flags = [
                f'--vcfFile=/data.vcf.bgz',
                f'--vcfFileIndex=/data.vcf.bgz.tbi',
                f'--vcfField=GT',
            ]
        else:
            export_cmd = f'hl.export_bgen(mt, "/data")'
            input_flags = [
                f'--bgenFile=/data.bgen',
                f'--bgenFileIndex=/data.bgen.idx',
                f'--sampleFile=/data.sample',
            ]

        if analysis_type == SaigeAnalysisType.GENE:
            assert sparse_grm is not None and group_annotations is not None
            assert chunk.groups is not None
            group_ann_filter_cmd = f'''
cat > filter_gene_annotations.py <<EOF
import hail as hl
import json
annotations = hl.import_table("{group_annotations}")
groups = json.loads("{json.dumps(chunk.groups)}")
annotations = annotations.filter(hl.is_defined(groups[annotations.group]))  # fixme: what is the right field name here?
EOF
'''
        else:
            group_ann_filter_cmd = ''

        hail_io_cmd = f'''
cat > read_from_mt.py <<EOF
import hail as hl
mt = hl.read_matrix_table("{mt_path}")
interval = hl.parse_locus_interval("{chunk.to_interval_str()}", reference_genome=mt.locus.dtype.reference_genome)
mt = mt.filter_rows(interval.contains(mt.locus))
{export_cmd}
EOF
python3 read_from_mt.py
'''

        saige_options = [
            f'--chrom={chunk.interval.start.contig}',
            f'--minMAF={self.min_maf}',
            f'--minMAC={self.min_mac}',
            f'--maxMAFforGroupTest={self.max_maf_for_group_test}',
            f'--GMMATmodelFile={null_model.rda}',
            f'--varianceRatioFile={null_model.variance_ratio}',
            f'--SAIGEOutputFile={result}',
            f'--numLinesOutput={self.num_lines_output}',
            f'--IsSparse={str(self.is_sparse).upper()}',
            f'--SPAcutoff={self.spa_cutoff}',
            f'--IsOutputAFinCaseCtrl={str(self.output_af_in_case_control).upper()}',
            f'--IsOutputNinCaseCtrl={str(self.output_n_in_case_control).upper()}',
            f'--IsOutputHetHomCountsinCaseCtrl={str(self.output_het_hom_counts).upper()}',
            f'--IsOutputAFinCaseCtrl={str(self.output_af_in_case_control).upper()}',
            f'--IsOutputMAFinCaseCtrlinGroupTest={str(self.output_maf_in_case_control_in_group_test).upper()}',
            f'--IsOutputlogPforSingle={str(self.output_logp_for_single).upper()}',
        ]

        if self.kernel is not None:
            saige_options.append(f'--kernel={self.kernel}')
        if self.method is not None:
            saige_options.append(f'--method={self.method}')
        if self.weights_beta_rare is not None:
            saige_options.append(f'--weights.beta.rare={self.weights_beta_rare}')
        if self.weights_beta_common is not None:
            saige_options.append(f'--weights.beta.common={self.weights_beta_common}')
        if self.weight_maf_cutoff is not None:
            saige_options.append(f'--weightMAFcutoff={self.weight_maf_cutoff}')
        if self.r_corr is not None:
            saige_options.append(f'--r.corr={self.r_corr}')
        if self.cate_var_ratio_min_mac_vec_exclude is not None:
            exclude = ','.join(str(val) for val in self.cate_var_ratio_min_mac_vec_exclude)
            saige_options.append(f'--cateVarRatioMinMACVecExclude={exclude}')
        if self.cate_var_ratio_max_mac_vec_include is not None:
            include = ','.join(str(val) for val in self.cate_var_ratio_max_mac_vec_include)
            saige_options.append(f'--cateVarRatioMaxMACVecInclude={include}')
        if self.dosage_zerod_cutoff is not None:
            saige_options.append(f'--dosageZerodCutoff={self.dosage_zerod_cutoff}')
        if self.output_pvalue_na_in_group_test_for_binary is not None:
            saige_options.append(
                f'--IsOutputPvalueNAinGroupTestforBinary={str(self.output_pvalue_na_in_group_test_for_binary).upper()}'
            )
        if self.account_for_case_control_imbalance_in_group_test is not None:
            saige_options.append(
                f'--IsAccountforCasecontrolImbalanceinGroupTest={str(self.account_for_case_control_imbalance_in_group_test).upper()}'
            )
        if self.x_par_region is not None:
            regions = ','.join(self.x_par_region)
            saige_options.append(f'--X_PARregion={regions}')
        if self.rewrite_x_nonpar_for_males is not None:
            saige_options.append(f'--is_rewrite_XnonPAR_forMales={str(self.rewrite_x_nonpar_for_males).upper()}')
        if self.method_to_collapse_ultra_rare is not None:
            saige_options.append(f'--method_to_CollapseUltraRare={self.method_to_collapse_ultra_rare}')
        if self.mac_cutoff_to_collapse_ultra_rare is not None:
            saige_options.append(f'--MACCutoff_to_CollapseUltraRare={self.mac_cutoff_to_collapse_ultra_rare}')
        if self.dosage_cutoff_for_ultra_rare_presence is not None:
            saige_options.append(f'--DosageCutoff_for_UltraRarePresence={self.dosage_cutoff_for_ultra_rare_presence}')

        # FIXME: --weightsIncludeinGroupFile, --weights_for_G2_cond, --sampleFile_male

        gene_options = []

        if analysis_type == SaigeAnalysisType.GENE:
            test_type = saige_phenotype_to_test_type[phenotype.phenotype_type]
            if test_type == SaigeTestType.BINARY:
                gene_options = [
                    f'--IsOutputPvalueNAinGroupTestforBinary={str(self.output_pvalue_na_in_group_test_for_binary).upper()}'
                ]
            else:
                gene_options = [
                    f'--groupFile={group_annotations}',
                    f'--sparseSigmaFile={sparse_grm.grm}',
                    f'--IsSingleVarinGroupTest={str(self.single_variant_in_group_test).upper()}',
                    f'--IsOutputBETASEinBurdenTest={str(self.output_beta_se_in_burden_test).upper()}',
                ]

        input_flags = '\n'.join(f'    {flag} \\ ' for flag in input_flags)
        saige_options = '\n'.join(f'    {opt} \\ ' for opt in saige_options)
        gene_options = '\n'.join(f'    {opt} \\ ' for opt in gene_options)

        command = f'''
set -o pipefail;
{mkl_off}

{hail_io_cmd}
{group_ann_filter_cmd}

step2_SPAtests.R \
{input_flags}
{saige_options}
{gene_options} 2>&1 | tee {stdout};
'''

        return command

    async def call(
        self,
        fs: AsyncFS,
        b: hb.Batch,
        *,
        mt_path: str,
        temp_dir: str,
        checkpoint_dir: Optional[str],
        analysis_type: SaigeAnalysisType,
        null_model: Union[SaigeGeneGLMMResourceGroup, SaigeGLMMResourceGroup],
        input_data_type: SaigeInputDataType,
        chunk: VariantChunk,
        phenotype: Phenotype,
        sparse_grm: Optional[SaigeSparseGRMResourceGroup] = None,
        group_annotations: Optional[TextResourceFile] = None,
    ) -> Union[SaigeGeneResultResourceGroup, SaigeResultResourceGroup]:
        output_root = self.output_file_prefix(temp_dir, checkpoint_dir, phenotype.name, chunk)

        results = await load_saige_result_file(fs, b, self, output_root, analysis_type)
        if results is not None:
            return results

        j = (
            b.new_job(
                name=self.name(phenotype=phenotype, chunk=chunk),
                attributes=self.attributes(
                    phenotype=phenotype,
                    chunk=chunk,
                    analysis_type=analysis_type.value,
                ),
            )
            .storage(self.storage)
            .image(self.image)
            .cpu(self.cpu)
            .memory(self.memory)
            .spot(self.spot)
        )

        results = new_saige_result_file(j, analysis_type)

        command = self.command(
            mt_path=mt_path,
            analysis_type=analysis_type,
            null_model=null_model,
            input_data_type=input_data_type,
            chunk=chunk,
            phenotype=phenotype,
            result=results,
            stdout=j.stdout,
            sparse_grm=sparse_grm,
            group_annotations=group_annotations,
        )

        j.command(command)

        checkpoint_if_requested(results, b, self, output_root)

        if self.save_stdout:
            b.write_output(j.stdout, f'{output_root}.log')

        return j


@dataclass
class CompilePhenotypeResultsStep(CheckpointConfigMixin, JobConfigMixin):
    def name(self, phenotype: Phenotype) -> str:
        return f'compile-results-{phenotype.name}'

    def attributes(self, *, phenotype: Phenotype) -> Optional[Dict]:
        return {'phenotype': phenotype.name}

    def results_path_glob(self, temp_dir: str, checkpoint_dir: Optional[str]):
        working_dir = get_output_dir(self, temp_dir, checkpoint_dir)
        return f'{working_dir}/compiled-results/*.txt.gz'

    def output_file(self, temp_dir: str, checkpoint_dir: Optional[str], phenotype_name: str) -> str:
        working_dir = get_output_dir(self, temp_dir, checkpoint_dir)
        return f'{working_dir}/compiled-results/{phenotype_name}.txt.gz'

    def check_if_output_exists(self,
                               fs: AsyncFS,
                               b: hb.Batch,
                               phenotype: Phenotype,
                               temp_dir: str,
                               checkpoint_dir: Optional[str]) -> Optional[TextResourceFile]:
        output_file = self.output_file(temp_dir, checkpoint_dir, phenotype.name)
        return await load_text_file(fs, b, self, output_file)

    def command(self, results_path: str, phenotype_name: str, output_file: str) -> str:
        return f'''
cat > compile_results.py <<EOF
import hail as hl
ht = hl.import_table("{results_path}", impute=True)
ht = ht.annotate(phenotype="{phenotype_name}")
ht.export("{output_file}", overwrite=True)
EOF
python3 compile_results.py
'''

    async def call(
        self,
        fs: AsyncFS,
        b: hb.Batch,
        phenotype: Phenotype,
        results_path: str,
        dependencies: List[hb.Job],
        temp_dir: str,
        checkpoint_dir: Optional[str],
    ) -> TextResourceFile:
        output_file = self.output_file(temp_dir, checkpoint_dir, phenotype.name)
        results = await load_text_file(fs, b, self, output_file)
        if results is not None:
            return results

        j = (b
             .new_job(name=self.name(phenotype), attributes=self.attributes(phenotype=phenotype))
             .image(self.image)
             .cpu(self.cpu)
             .memory(self.memory)
             .depends_on(*dependencies)
        )

        compiled_results = new_text_file(j)

        cmd = self.command(results_path, phenotype.name, compiled_results)

        j.command(cmd)

        checkpoint_if_requested(compiled_results, b, self, output_file)

        return compiled_results


@dataclass
class CompileAllResultsStep(CheckpointConfigMixin, JobConfigMixin):
    def name(self) -> str:
        return 'compile-all-results'

    def command(self, results_path: str, output_file: str) -> str:
        return f'''
cat > compile_results.py <<EOF
import hail as hl
ht = hl.import_table("{results_path}", impute=True)
ht.write("{output_file}", overwrite=True)
EOF
python3 compile_results.py
'''

    async def call(
        self,
        fs: AsyncFS,
        b: hb.Batch,
        results_path: str,
        output_ht_path: str,
        dependencies: List[hb.Job],
    ):
        if fs.isdir(output_ht_path) and not self.overwrite:
            return

        j = (b
             .new_job(name=self.name())
             .image(self.image)
             .cpu(self.cpu)
             .memory(self.memory)
             .depends_on(*dependencies)
        )

        cmd = self.command(results_path, output_ht_path)

        j.command(cmd)
