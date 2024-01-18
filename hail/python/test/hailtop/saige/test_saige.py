import pytest
import hail as hl
from hailtop.saige import SAIGE, SaigeConfig

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


@pytest.fixture
def output_path():
    return hl.utils.new_temp_file('saige_output')


def test_variant_chunking(ds):
    pass


def test_phenotype_grouping(ds):
    pass


def test_saige_categorical(ds, output_path):
    saige = SAIGE()
    saige.run(
        mt=ds,
        output=output_path,
        phenotypes=[ds.phenotype.psych],
        covariates=[ds.cov.c1, ds.cov.c2]
    )


def test_saige_categorical_custom_config(ds):
    pass


def test_saige_continuous(ds):
    pass


def test_saige_gene(ds):
    pass

