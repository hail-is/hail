import doctest
import os

import numpy as np
import pytest

import hail as hl

SKIP_OUTPUT_CHECK = doctest.register_optionflag('SKIP_OUTPUT_CHECK')
HAIL_QUERY_N_CORES = os.environ.get('HAIL_QUERY_N_CORES', '*')


@pytest.fixture(autouse=True)
def patch_doctest_check_output(monkeypatch):
    # FIXME: remove once test output matches docs
    base_check_output = doctest.OutputChecker.check_output

    def patched_check_output(self, want, got, optionflags):
        return ((not want)
                or (want.strip() == 'None')
                or (SKIP_OUTPUT_CHECK & optionflags)
                or base_check_output(self, want, got, optionflags | doctest.NORMALIZE_WHITESPACE))

    monkeypatch.setattr('doctest.OutputChecker.check_output', patched_check_output)
    yield
    monkeypatch.undo()


@pytest.fixture(scope="session", autouse=True)
def init(doctest_namespace):
    # This gets run once per process -- must avoid race conditions
    print("setting up doctest...")

    olddir = os.getcwd()
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "docs"))

    hl.init(global_seed=0, master=f'local[{HAIL_QUERY_N_CORES}]')
    hl.reset_global_randomness()

    try:
        generate_datasets(doctest_namespace)
        print("finished setting up doctest...")
        yield
    finally:
        os.chdir(olddir)


@pytest.fixture(autouse=True)
def reset_randomness(init):
    hl.reset_global_randomness()


def generate_datasets(doctest_namespace):
    doctest_namespace['hl'] = hl
    doctest_namespace['np'] = np

    # ds = hl.import_vcf('data/sample.vcf.bgz')
    # ds = ds.sample_rows(0.035)
    # ds = ds.annotate_rows(use_as_marker=hl.rand_bool(0.5),
    #                       panel_maf=0.1,
    #                       anno1=5,
    #                       anno2=0,
    #                       consequence="LOF",
    #                       gene="A",
    #                       score=5.0)
    # ds = ds.annotate_rows(a_index=1)
    # ds = hl.sample_qc(hl.variant_qc(ds))
    # ds = ds.annotate_cols(is_case=True,
    #                       pheno=hl.struct(is_case=hl.rand_bool(0.5),
    #                                       is_female=hl.rand_bool(0.5),
    #                                       age=hl.rand_norm(65, 10),
    #                                       height=hl.rand_norm(70, 10),
    #                                       blood_pressure=hl.rand_norm(120, 20),
    #                                       cohort_name="cohort1"),
    #                       cov=hl.struct(PC1=hl.rand_norm(0, 1)),
    #                       cov1=hl.rand_norm(0, 1),
    #                       cov2=hl.rand_norm(0, 1),
    #                       cohort="SIGMA")
    # ds = ds.annotate_globals(global_field_1=5,
    #                          global_field_2=10,
    #                          pli={'SCN1A': 0.999, 'SONIC': 0.014},
    #                          populations=['AFR', 'EAS', 'EUR', 'SAS', 'AMR', 'HIS'])
    # ds = ds.annotate_rows(gene=['TTN'])
    # ds = ds.annotate_cols(cohorts=['1kg'], pop='EAS')
    # ds.checkpoint('data/example.mt', overwrite=True)
    ds = hl.read_matrix_table('data/example.mt')
    doctest_namespace['ds'] = ds
    doctest_namespace['dataset'] = ds
    doctest_namespace['dataset2'] = ds.annotate_globals(global_field=5)
    doctest_namespace['dataset_to_union_1'] = ds
    doctest_namespace['dataset_to_union_2'] = ds

    v_metadata = ds.rows().annotate_globals(global_field=5).annotate(consequence='SYN')
    doctest_namespace['v_metadata'] = v_metadata

    s_metadata = ds.cols().annotate(pop='AMR', is_case=False, sex='F')
    doctest_namespace['s_metadata'] = s_metadata
    doctest_namespace['cols_to_keep'] = s_metadata
    doctest_namespace['cols_to_remove'] = s_metadata
    doctest_namespace['rows_to_keep'] = v_metadata
    doctest_namespace['rows_to_remove'] = v_metadata

    # small_mt = hl.balding_nichols_model(3, 4, 4)
    # small_mt.checkpoint('data/small.mt', overwrite=True)
    doctest_namespace['small_mt'] = hl.read_matrix_table('data/small.mt')

    # Table
    # table1 = hl.import_table('data/kt_example1.tsv', impute=True, key='ID')
    # table1 = table1.annotate_globals(global_field_1=5, global_field_2=10)
    # table1.write('data/kt_example1.ht')
    table1 = hl.read_table('data/kt_example1.ht')
    doctest_namespace['table1'] = table1
    doctest_namespace['other_table'] = table1

    # table2 = hl.import_table('data/kt_example2.tsv', impute=True, key='ID')
    # table2.write('data/kt_example2.ht')
    doctest_namespace['table2'] = hl.read_table('data/kt_example2.ht')

    # table4 = hl.import_table('data/kt_example4.tsv', impute=True,
    #                          types={'B': hl.tstruct(B0=hl.tbool, B1=hl.tstr),
    #                                 'D': hl.tstruct(cat=hl.tint32, dog=hl.tint32),
    #                                 'E': hl.tstruct(A=hl.tint32, B=hl.tint32)})
    # table4.write('data/kt_example4.ht')
    doctest_namespace['table4'] = hl.read_table('data/kt_example4.ht')

    # people_table = hl.import_table('data/explode_example.tsv', delimiter='\\s+',
    #                                types={'Age': hl.tint32, 'Children': hl.tarray(hl.tstr)},
    #                                key='Name')
    # people_table.write('data/explode_example.ht')
    doctest_namespace['people_table'] = hl.read_table('data/explode_example.ht')

    # TDT
    # tdt_dataset = hl.import_vcf('data/tdt_tiny.vcf')
    # tdt_dataset.write('data/tdt_tiny.mt')
    doctest_namespace['tdt_dataset'] = hl.read_matrix_table('data/tdt_tiny.mt')

    ds2 = hl.variant_qc(ds)
    doctest_namespace['ds2'] = ds2.select_rows(AF=ds2.variant_qc.AF)

    # Expressions
    doctest_namespace['names'] = hl.literal(['Alice', 'Bob', 'Charlie'])
    doctest_namespace['a1'] = hl.literal([0, 1, 2, 3, 4, 5])
    doctest_namespace['a2'] = hl.literal([1, -1, 1, -1, 1, -1])
    doctest_namespace['t'] = hl.literal(True)
    doctest_namespace['f'] = hl.literal(False)
    doctest_namespace['na'] = hl.missing(hl.tbool)
    doctest_namespace['call'] = hl.call(0, 1, phased=False)
    doctest_namespace['a'] = hl.literal([1, 2, 3, 4, 5])
    doctest_namespace['d'] = hl.literal({'Alice': 43, 'Bob': 33, 'Charles': 44})
    doctest_namespace['interval'] = hl.interval(3, 11)
    doctest_namespace['locus_interval'] = hl.parse_locus_interval("1:53242-90543")
    doctest_namespace['locus'] = hl.locus('1', 1034245)
    doctest_namespace['x'] = hl.literal(3)
    doctest_namespace['y'] = hl.literal(4.5)
    doctest_namespace['s1'] = hl.literal({1, 2, 3})
    doctest_namespace['s2'] = hl.literal({1, 3, 5})
    doctest_namespace['s3'] = hl.literal({'Alice', 'Bob', 'Charlie'})
    doctest_namespace['struct'] = hl.struct(a=5, b='Foo')
    doctest_namespace['tup'] = hl.literal(("a", 1, [1, 2, 3]))
    doctest_namespace['s'] = hl.literal('The quick brown fox')
    doctest_namespace['interval2'] = hl.Interval(3, 6)
    doctest_namespace['nd'] = hl.nd.array([[1, 2], [3, 4]])

    # Overview
    doctest_namespace['ht'] = table1
    doctest_namespace['mt'] = ds

    gnomad_data = ds.rows()
    doctest_namespace['gnomad_data'] = gnomad_data.select(gnomad_data.info.AF)

    # BGEN
    bgen = hl.import_bgen('data/example.8bits.bgen',
                          entry_fields=['GT', 'GP', 'dosage'])
    doctest_namespace['variants_table'] = bgen.rows()

    # burden_ds = hl.import_vcf('data/example_burden.vcf')
    # burden_kt = hl.import_table('data/example_burden.tsv', key='Sample', impute=True)
    # burden_ds = burden_ds.annotate_cols(burden=burden_kt[burden_ds.s])
    # burden_ds = burden_ds.annotate_rows(weight=hl.float64(burden_ds.locus.position))
    # burden_ds = hl.variant_qc(burden_ds)
    # genekt = hl.import_locus_intervals('data/gene.interval_list')
    # burden_ds = burden_ds.annotate_rows(gene=genekt[burden_ds.locus])
    # burden_ds = burden_ds.checkpoint('data/example_burden.mt', overwrite=True)
    doctest_namespace['burden_ds'] = hl.read_matrix_table('data/example_burden.mt')

    # ld_score_one_pheno_sumstats = hl.import_table(
    #     'data/ld_score_regression.one_pheno.sumstats.tsv',
    #     types={'locus': hl.tlocus('GRCh37'),
    #            'alleles': hl.tarray(hl.tstr),
    #            'chi_squared': hl.tfloat64,
    #            'n': hl.tint32,
    #            'ld_score': hl.tfloat64,
    #            'phenotype': hl.tstr,
    #            'chi_squared_50_irnt': hl.tfloat64,
    #            'n_50_irnt': hl.tint32,
    #            'chi_squared_20160': hl.tfloat64,
    #            'n_20160': hl.tint32},
    #     key=['locus', 'alleles'])
    # ld_score_one_pheno_sumstats.write('data/ld_score_regression.one_pheno.sumstats.ht')
    doctest_namespace['ld_score_one_pheno_sumstats'] = hl.read_table('data/ld_score_regression.one_pheno.sumstats.ht')

    # mt = hl.import_matrix_table(
    #     'data/ld_score_regression.all_phenos.sumstats.tsv',
    #     row_fields={'locus': hl.tstr,
    #                 'alleles': hl.tstr,
    #                 'ld_score': hl.tfloat64},
    #     entry_type=hl.tstr)
    # mt = mt.key_cols_by(phenotype=mt.col_id)
    # mt = mt.key_rows_by(locus=hl.parse_locus(mt.locus), alleles=mt.alleles.split(','))
    # mt = mt.drop('row_id', 'col_id')
    # mt = mt.annotate_entries(x=mt.x.split(","))
    # mt = mt.transmute_entries(chi_squared=hl.float64(mt.x[0]), n=hl.int32(mt.x[1]))
    # mt = mt.annotate_rows(ld_score=hl.float64(mt.ld_score))
    # mt.write('data/ld_score_regression.all_phenos.sumstats.mt')
    doctest_namespace['ld_score_all_phenos_sumstats'] = hl.read_matrix_table('data/ld_score_regression.all_phenos.sumstats.mt')

    print("finished setting up doctest...")
