import hail as hl
import hail.utils as utils
from ...helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


@fails_spark_backend()
def test_pc_relate_2_against_R_truth():
    mt = hl.import_vcf(resource('pc_relate_bn_input.vcf.bgz'))
    hail_kin = hl.pc_relate_2(mt.GT, 0.00, k=2).checkpoint(utils.new_temp_file(extension='ht'))

    r_kin = hl.import_table(resource('pc_relate_r_truth.tsv.bgz'),
                            types={'i': 'struct{s:str}',
                                   'j': 'struct{s:str}',
                                   'kin': 'float',
                                   'ibd0': 'float',
                                   'ibd1': 'float',
                                   'ibd2': 'float'},
                            key=['i', 'j'])
    assert r_kin.select("kin")._same(hail_kin.select("kin"), tolerance=1e-3, absolute=True)
    assert r_kin.select("ibd0")._same(hail_kin.select("ibd0"), tolerance=1.3e-2, absolute=True)
    assert r_kin.select("ibd1")._same(hail_kin.select("ibd1"), tolerance=2.6e-2, absolute=True)
    assert r_kin.select("ibd2")._same(hail_kin.select("ibd2"), tolerance=1.3e-2, absolute=True)


@fails_spark_backend()
def test_pc_relate_2_simple_example():
    gs = hl.literal(
        [[0, 0, 0, 0, 1, 1, 1, 1],
         [0, 0, 1, 1, 0, 0, 1, 1],
         [0, 1, 0, 1, 0, 1, 0, 1],
         [0, 0, 1, 1, 0, 0, 1, 1]])
    scores = hl.literal([[0, 1], [1, 0], [2, 0], [0, -1]])
    mt = hl.utils.range_matrix_table(n_rows=8, n_cols=4)
    mt = mt.annotate_entries(GT=hl.unphased_diploid_gt_index_call(gs[mt.col_idx][mt.row_idx]))
    mt = mt.annotate_cols(scores=scores[mt.col_idx])
    pcr = hl.pc_relate_2(mt.GT, min_individual_maf=0, scores_expr=mt.scores)

    expected = [
        hl.Struct(i=0, j=1, kin=0.04308803311427006,
                  ibd0=0.8371964347601548, ibd1=0.1532549980226101, ibd2=0.009548567217235104),
        hl.Struct(i=0, j=2, kin=-0.02924995111304614,
                  ibd0=0.8758618068470698, ibd1=0.36527619075804485, ibd2=-0.24113799760511467),
        hl.Struct(i=0, j=3, kin=0.06218009537126304,
                  ibd0=0.8278359528633145, ibd1=0.09560771278831892, ibd2=0.07655633434836666),
        hl.Struct(i=1, j=2, kin=0.08129469602400068,
                  ibd0=-0.1679632749787232, ibd1=2.010747765861444, ibd2=-0.8427844908827204),
        hl.Struct(i=1, j=3, kin=0.06179990046708038,
                  ibd0=0.8946584074678073, ibd1=-0.03651641680393625, ibd2=0.14185800933612888),
        hl.Struct(i=2, j=3, kin=-0.012863506769063277,
                  ibd0=0.7395091389950708, ibd1=0.5724357490861116, ibd2=-0.31194488808118237)
    ]
    ht_expected = hl.Table.parallelize(expected)
    ht_expected = ht_expected.key_by(i=hl.struct(col_idx=ht_expected.i),
                                     j=hl.struct(col_idx=ht_expected.j))
    assert ht_expected._same(pcr)


@fails_spark_backend()
def test_pc_relate_2_paths():
    mt = hl.balding_nichols_model(3, 50, 100)
    _, scores3, _ = hl._hwe_normalized_blanczos(mt.GT, k=3, compute_loadings=False)

    kin1 = hl.pc_relate_2(mt.GT, 0.10, k=2, statistics='kin', block_size=64)
    kin2 = hl.pc_relate_2(mt.GT, 0.05, k=2, min_kinship=0.01, statistics='kin2', block_size=128).cache()
    kin3 = hl.pc_relate_2(mt.GT, 0.02, k=3, min_kinship=0.1, statistics='kin20', block_size=64).cache()
    kin_s1 = hl.pc_relate_2(mt.GT, 0.10, scores_expr=scores3[mt.col_key].scores[:2],
                            statistics='kin', block_size=32)

    assert kin1._same(kin_s1, tolerance=1e-4)

    assert kin1.count() == 50 * 49 / 2

    assert kin2.count() > 0
    assert kin2.filter(kin2.kin < 0.01).count() == 0

    assert kin3.count() > 0
    assert kin3.filter(kin3.kin < 0.1).count() == 0


@fails_spark_backend()
def test_self_kinship():
    mt = hl.balding_nichols_model(3, 10, 50)
    with_self = hl.pc_relate_2(mt.GT, 0.10, k=2, statistics='kin20', block_size=16, include_self_kinship=True)
    without_self = hl.pc_relate_2(mt.GT, 0.10, k=2, statistics='kin20', block_size=16)

    assert with_self.count() == 55
    assert without_self.count() == 45

    with_self_self_kin_only = with_self.filter(with_self.i.sample_idx == with_self.j.sample_idx)
    assert with_self_self_kin_only.count() == 10, with_self_self_kin_only.collect()

    with_self_no_self_kin = with_self.filter(with_self.i.sample_idx != with_self.j.sample_idx)
    assert with_self_no_self_kin.count() == 45, with_self_no_self_kin.collect()
    assert with_self_no_self_kin._same(without_self)

    without_self_self_kin_only = without_self.filter(without_self.i.sample_idx == without_self.j.sample_idx)
    assert without_self_self_kin_only.count() == 0, without_self_self_kin_only.collect()


@fails_spark_backend()
def test_pcrelate_issue_5263():
    mt = hl.balding_nichols_model(3, 50, 100)
    expected = hl.pc_relate_2(mt.GT, 0.10, k=2, statistics='all')
    mt = mt.select_entries(GT2=mt.GT,
                           GT=hl.call(hl.rand_bool(0.5), hl.rand_bool(0.5)))
    actual = hl.pc_relate_2(mt.GT2, 0.10, k=2, statistics='all')
    assert expected._same(actual, tolerance=1e-4)
