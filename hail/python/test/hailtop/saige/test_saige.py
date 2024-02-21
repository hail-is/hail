import pytest
import hail as hl
import hailtop.fs as hfs
from hailtop.saige import (
    Phenotype,
    PhenotypeConfig,
    SaigeConfig,
    SaigePhenotype,
    Step1NullGlmmStep,
    Step2SPAStep,
    extract_phenotypes,
    compute_variant_chunks_by_contig,
    saige
)

from ...hail.helpers import *


@pytest.fixture(scope='module')
def ds():
    variants = hl.import_table(resource('dataproc_vep_grch37_annotations.tsv.gz'),
                               key=['locus', 'alleles'],
                               types={'locus': hl.tlocus('GRCh37'), 'alleles': hl.tarray(hl.tstr), 'vep': hl.tstr},
                               force=True)
    variants = variants.add_index()
    variants = variants.key_by(variants.idx)

    dataset = hl.balding_nichols_model(1, 100, variants.count())
    dataset = dataset.add_row_index()
    dataset = dataset.annotate_rows(new_locus=variants[dataset.row_idx].locus, new_alleles=variants[dataset.row_idx].alleles)
    dataset = dataset.key_rows_by()
    dataset = dataset.key_rows_by(locus=dataset.new_locus, alleles=dataset.new_alleles)
    dataset = dataset.key_cols_by(s=hl.str(dataset.sample_idx + 1))
    # dataset = hl.vep(dataset)
    dataset = dataset.annotate_cols(phenotype=hl.struct(height=hl.rand_norm(), psych=hl.rand_bool(0.5), cardio=hl.rand_bool(0.5)))
    dataset = dataset.annotate_cols(cov=hl.struct(c1=hl.rand_norm(), c2=hl.rand_norm()))
    cohorts = hl.array(['cohort1', 'cohort2', 'cohort3'])
    dataset = dataset.annotate_cols(cohort=cohorts[hl.rand_cat([1, 1, 1])])
    dataset.write('gs://hail-jigold/saige-test-dataset.mt')
    assert False
    return dataset


# @pytest.fixture
# def output_path():
#     return hl.utils.new_temp_file('saige_output')


def test_variant_chunking(ds):
    pass


def test_variant_group_chunking(ds):
    pass


def test_single_variant_example1():
    """https://saigegit.github.io/SAIGE-doc/docs/single_example.html#example-1"""

    step1_null_glmm = Step1NullGlmmStep(cpu=2, is_overwrite_variance_ratio_file=True)
    step2_spa = Step2SPAStep(chrom='1',
                             min_maf=0,
                             min_mac=20,
                             is_firth_beta=True,
                             p_cutoff_for_firth=0.05,
                             output_more_details=True,
                             loco=True)
    phenotype_config = PhenotypeConfig(resource(...),
                                       sample_id_col='IID',
                                       phenotypes=[Phenotype('y_binary', SaigePhenotype.BINARY)],
                                       covariates=[Phenotype('x1', SaigePhenotype.CONTINUOUS),
                                                   Phenotype('x2', SaigePhenotype.BINARY)])
    saige_config = SaigeConfig(step1_null_glmm=step1_null_glmm, step2_spa=step2_spa)
    saige(mt_path, null_model_plink_path, phenotypes_path, output_path, phenotype_config=phenotype_config,
          saige_config=saige_config)


def test_single_variant_example2():
    pass


def test_single_variant_example3():
    pass


def test_saige_categorical():
    mt_path = 'gs://hail-jigold/saige-test-dataset.mt'
    ds = hl.read_matrix_table(mt_path)

    with hl.TemporaryDirectory(suffix='.mt', ensure_exists=False):
        with hl.TemporaryFilename(suffix='.txt') as phenotypes_file:
            with hl.TemporaryDirectory() as null_model_plink_dir:
                with hl.TemporaryDirectory(suffix='.ht', ensure_exists=False) as output_path:
                    null_model_plink_path = f'{null_model_plink_dir}/null-model-input'

                    phenotype_information = extract_phenotypes(ds,
                                                               phenotypes={'psych': ds.phenotype.psych, 'cardio': ds.phenotype.cardio},
                                                               covariates={'c1': ds.cov.c1, 'c2': ds.cov.c2},
                                                               output_file=phenotypes_file)

                    variant_chunks = compute_variant_chunks_by_contig(ds)

                    hl.export_plink(ds, null_model_plink_path)

                    saige(mt_path=mt_path,
                          null_model_plink_path=null_model_plink_path,
                          phenotypes_path=phenotypes_file,
                          phenotype_config=phenotype_information,
                          variant_chunks=variant_chunks,
                          output_path=output_path,
                          config=SaigeConfig(step1_null_glmm=Step1NullGlmmStep(min_covariate_count=1, skip_model_fitting=False)))

                    # check results table is there
                    ht = hl.read_table(output_path)
                    ht.describe()


def test_saige_continuous(ds):
    pass


def test_saige_gene(ds):
    pass


def test_saige_custom_config(ds):
    pass

