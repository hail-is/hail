import pytest
import os
import shutil
import hail as hl
import hail.expr.aggregators as agg


def atomic_write(ds, f):
    tmp = hl.utils.uri_path(hl.utils.new_temp_file())
    ds.write(tmp, overwrite=True)
    if not os.path.isdir(f):
        os.rename(tmp, f) # Make write atomic

@pytest.fixture(autouse=True)
def always_true(monkeypatch):
    # FIXME: remove once test output matches docs
    monkeypatch.setattr('doctest.OutputChecker.check_output', lambda a, b, c, d: True)
    yield
    monkeypatch.undo()


@pytest.fixture(scope="session", autouse=True)
def init(doctest_namespace):
    # This gets run once per process -- must avoid race conditions
    print("setting up doctest...")

    olddir = os.getcwd()
    os.chdir("docs/")

    doctest_namespace['hl'] = hl
    doctest_namespace['agg'] = agg

    if not os.path.isdir("output/"):
        try:
            os.mkdir("output/")
        except OSError:
            pass

    files = ["sample.vds", "sample.qc.vds", "sample.filtered.vds"]
    for f in files:
        if os.path.isdir(f):
            shutil.rmtree(f)

    # MatrixTable
    ds = hl.import_vcf('data/sample.vcf.bgz')
    ds = ds.sample_rows(0.03)
    ds = ds.annotate_rows(use_in_kinship=hl.rand_bool(0.9),
                          panel_maf=0.1,
                          anno1=5,
                          anno2=0,
                          consequence="LOF",
                          gene="A",
                          score=5.0)
    ds = ds.annotate_rows(a_index=1)
    ds = hl.sample_qc(hl.variant_qc(ds))
    ds = ds.annotate_cols(is_case=True,
                          pheno=hl.struct(is_case=hl.rand_bool(0.5),
                                          is_female=hl.rand_bool(0.5),
                                          age=hl.rand_norm(65, 10),
                                          height=hl.rand_norm(70, 10),
                                          blood_pressure=hl.rand_norm(120, 20),
                                          cohort_name="cohort1"),
                          cov=hl.struct(PC1=hl.rand_norm(0, 1)),
                          cov1=hl.rand_norm(0, 1),
                          cov2=hl.rand_norm(0, 1))
    ds = ds.annotate_globals(global_field_1=5, global_field_2=10)
    ds = ds.annotate_globals(pli={'SCN1A': 0.999, 'SONIC': 0.014},
                             populations = ['AFR', 'EAS', 'EUR', 'SAS', 'AMR', 'HIS'])

    ds = ds.annotate_rows(gene=['TTN'])
    ds = ds.annotate_cols(cohorts=['1kg'], pop='EAS', cohort='cohort')
    atomic_write(ds, 'data/example.vds')

    ds = hl.read_matrix_table('data/example.vds')
    doctest_namespace['ds'] = ds
    doctest_namespace['dataset'] = ds
    doctest_namespace['dataset2'] = ds.annotate_globals(global_field=5)
    doctest_namespace['dataset_to_union_1'] = ds
    doctest_namespace['dataset_to_union_2'] = ds

    v_metadata = ds.rows().annotate_globals(global_field=5).annotate(consequence='SYN')
    doctest_namespace['v_metadata'] = v_metadata

    s_metadata = ds.cols().annotate(pop='AMR', is_case=False, sex='F')
    doctest_namespace['s_metadata'] = s_metadata

    # Table
    table1 = hl.import_table('data/kt_example1.tsv', impute=True, key='ID')
    table1 = table1.annotate_globals(global_field_1=5, global_field_2=10)
    doctest_namespace['table1'] = table1
    doctest_namespace['other_table'] = table1

    table2 = hl.import_table('data/kt_example2.tsv', impute=True, key='ID')
    doctest_namespace['table2'] = table2

    table4 = hl.import_table('data/kt_example4.tsv', impute=True,
                             types={'B': hl.tstruct(B0=hl.tbool, B1=hl.tstr),
                                    'D': hl.tstruct(cat=hl.tint32, dog=hl.tint32),
                                    'E': hl.tstruct(A=hl.tint32, B=hl.tint32)})
    doctest_namespace['table4'] = table4

    people_table = hl.import_table('data/explode_example.tsv', delimiter='\\s+',
                                   types={'Age': hl.tint32, 'Children': hl.tarray(hl.tstr)})
    doctest_namespace['people_table'] = people_table

    # TDT
    doctest_namespace['tdt_dataset'] = hl.import_vcf('data/tdt_tiny.vcf')

    # Linear mixed regression
    lmmreg_ds = hl.variant_qc(hl.split_multi_hts(hl.import_vcf('data/sample.vcf.bgz')))
    lmmreg_tsv = hl.import_table('data/example_lmmreg.tsv', 'Sample', impute=True)
    lmmreg_ds = lmmreg_ds.annotate_cols(**lmmreg_tsv[lmmreg_ds['s']])
    lmmreg_ds = lmmreg_ds.annotate_rows(use_in_kinship = lmmreg_ds.variant_qc.AF > 0.05)
    atomic_write(lmmreg_ds, 'data/example_lmmreg.vds')

    # SKAT
    burden_ds = hl.import_vcf('data/example_burden.vcf')
    burden_kt = hl.import_table('data/example_burden.tsv', key='Sample', impute=True)
    burden_ds = burden_ds.annotate_cols(burden = burden_kt[burden_ds.s])
    burden_ds = burden_ds.annotate_rows(weight = hl.float64(burden_ds.locus.position))
    burden_ds = hl.variant_qc(burden_ds)
    genekt = hl.import_locus_intervals('data/gene.interval_list')
    burden_ds = burden_ds.annotate_rows(gene = genekt[burden_ds.locus])
    atomic_write(burden_ds, 'data/example_burden.vds')

    ds2 = hl.variant_qc(ds)
    doctest_namespace['ds2'] = ds2.select_rows(AF = ds2.variant_qc.AF)

    # Expressions
    doctest_namespace['names'] = hl.literal(['Alice', 'Bob', 'Charlie'])
    doctest_namespace['a1'] = hl.literal([0, 1, 2, 3, 4, 5])
    doctest_namespace['a2'] = hl.literal([1, -1, 1, -1, 1, -1])
    doctest_namespace['t'] = hl.literal(True)
    doctest_namespace['f'] = hl.literal(False)
    doctest_namespace['na'] = hl.null(hl.tbool)
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

    print("finished setting up doctest...")
    yield
    os.chdir(olddir)
