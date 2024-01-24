import pytest
import hail as hl
import hailtop.fs as hfs
from hailtop.saige import (
    SaigeConfig,
    extract_phenotypes,
    prepare_plink_null_model_input,
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
    variants = variants.select().collect()

    dataset = hl.balding_nichols_model(1, 100, len(variants))
    dataset = dataset.add_row_index()
    dataset = dataset.annotate_rows(new_locus=variants[dataset.name].locus, new_alleles=variants[dataset.name].alleles)
    dataset = dataset.key_rows_by()
    dataset = dataset.key_rows_by(locus=dataset.new_locus, alleles=dataset.new_alleles)
    dataset = dataset.key_cols_by(s=hl.str(dataset.sample_idx + 1))
    # dataset = hl.vep(dataset)
    dataset = dataset.annotate_cols(phenotype=hl.struct(height=hl.rand_norm(), psych=hl.rand_bool(), cardio=hl.rand_bool()))
    dataset = dataset.annotate_cols(cov=hl.struct(c1=hl.rand_norm(), c2=hl.rand_norm()))
    cohorts = ['cohort1', 'cohort2', 'cohort3']
    dataset = dataset.annotate_cols(cohort=cohorts[hl.rand_cat([1, 1, 1])])
    return dataset


# @pytest.fixture
# def output_path():
#     return hl.utils.new_temp_file('saige_output')


def test_variant_chunking(ds):
    pass


def test_variant_group_chunking(ds):
    pass


def test_saige_categorical(ds):
    mt_path = hl.utils.new_temp_file('input-data', '.mt')
    phenotypes_file = hl.utils.new_temp_file('phenotypes', '.txt')
    null_model_plink_path = hl.utils.new_temp_file('null-model-plink-input')
    output_path = hl.utils.new_temp_file('saige-results', '.ht')

    ds.write(mt_path)

    phenotypes, covariates = extract_phenotypes(ds,
                                                phenotypes={'psych': ds.phenotype.psych, 'cardio': ds.phenotype.cardio},
                                                covariates={'c1': ds.cov.c1, 'c2': ds.cov.c2},
                                                output_file=phenotypes_file)

    variant_chunks = compute_variant_chunks_by_contig(ds)

    hl.export_plink(ds, null_model_plink_path)

    saige(mt_path=mt_path,
          null_model_plink_path=null_model_plink_path,
          phenotypes_path=phenotypes_file,
          phenotypes=phenotypes,
          covariates=covariates,
          variant_chunks=variant_chunks,
          output_path=output_path)

    # check results table is there
    hl.import_table(output_path)


def test_saige_continuous(ds):
    pass


def test_saige_gene(ds):
    pass


def test_saige_custom_config(ds):
    pass

