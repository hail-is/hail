import hail as hl
import hail.utils as utils

from ...helpers import resource, skip_when_service_backend, test_timeout


@test_timeout(local=6 * 60, batch=6 * 60)
def test_pc_relate_against_R_truth():
    mt = hl.import_vcf(resource('pc_relate_bn_input.vcf.bgz'))
    hail_kin = hl.pc_relate(mt.GT, 0.00, k=2).checkpoint(utils.new_temp_file(extension='ht'))

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


def test_pc_relate_simple_example():
    gs = hl.literal([[0, 0, 0, 0, 1, 1, 1, 1],
                     [0, 0, 1, 1, 0, 0, 1, 1],
                     [0, 1, 0, 1, 0, 1, 0, 1],
                     [0, 0, 1, 1, 0, 0, 1, 1]])
    scores = hl.literal([[1, 1], [-1, 0], [1, -1], [-1, 0]])
    mt = hl.utils.range_matrix_table(n_rows=8, n_cols=4)
    mt = mt.annotate_entries(GT=hl.unphased_diploid_gt_index_call(gs[mt.col_idx][mt.row_idx]))
    mt = mt.annotate_cols(scores=scores[mt.col_idx])
    pcr = hl.pc_relate(mt.GT, min_individual_maf=0, scores_expr=mt.scores)

    expected = [hl.Struct(i=0, j=1, kin=0.0, ibd0=1.0, ibd1=0.0, ibd2=0.0),
                hl.Struct(i=0, j=2, kin=0.0, ibd0=1.0, ibd1=0.0, ibd2=0.0),
                hl.Struct(i=0, j=3, kin=0.0, ibd0=1.0, ibd1=0.0, ibd2=0.0),
                hl.Struct(i=1, j=2, kin=0.0, ibd0=1.0, ibd1=0.0, ibd2=0.0),
                hl.Struct(i=1, j=3, kin=0.0, ibd0=1.0, ibd1=0.0, ibd2=0.0),
                hl.Struct(i=2, j=3, kin=0.0, ibd0=1.0, ibd1=0.0, ibd2=0.0)]
    ht_expected = hl.Table.parallelize(expected)
    ht_expected = ht_expected.key_by(i=hl.struct(col_idx=ht_expected.i),
                                     j=hl.struct(col_idx=ht_expected.j))
    assert ht_expected._same(pcr, tolerance=1e-12, absolute=True)


@test_timeout(6 * 60, batch=14 * 60)
def test_pc_relate_paths():
    mt = hl.balding_nichols_model(3, 50, 100)
    _, scores3, _ = hl._hwe_normalized_blanczos(mt.GT, k=3, compute_loadings=False, q_iterations=10)

    kin1 = hl.pc_relate(mt.GT, 0.10, k=2, statistics='kin', block_size=64)
    kin2 = hl.pc_relate(mt.GT, 0.05, k=2, min_kinship=0.01, statistics='kin2', block_size=128).cache()
    kin3 = hl.pc_relate(mt.GT, 0.02, k=3, min_kinship=0.1, statistics='kin20', block_size=64).cache()
    kin_s1 = hl.pc_relate(mt.GT, 0.10, scores_expr=scores3[mt.col_key].scores[:2],
                          statistics='kin', block_size=64)

    assert kin1._same(kin_s1, tolerance=1e-4)

    assert kin1.count() == 50 * 49 / 2

    assert kin2.count() > 0
    assert kin2.filter(kin2.kin < 0.01).count() == 0

    assert kin3.count() > 0
    assert kin3.filter(kin3.kin < 0.1).count() == 0


@test_timeout(local=6 * 60, batch=10 * 60)
def test_self_kinship():
    mt = hl.balding_nichols_model(3, 10, 50)
    with_self = hl.pc_relate(mt.GT, 0.10, k=2, statistics='kin20', block_size=16, include_self_kinship=True)
    without_self = hl.pc_relate(mt.GT, 0.10, k=2, statistics='kin20', block_size=16)

    assert with_self.count() == 55
    assert without_self.count() == 45

    with_self_self_kin_only = with_self.filter(with_self.i.sample_idx == with_self.j.sample_idx)
    assert with_self_self_kin_only.count() == 10, with_self_self_kin_only.collect()

    with_self_no_self_kin = with_self.filter(with_self.i.sample_idx != with_self.j.sample_idx)
    assert with_self_no_self_kin.count() == 45, with_self_no_self_kin.collect()
    assert with_self_no_self_kin._same(without_self)

    without_self_self_kin_only = without_self.filter(without_self.i.sample_idx == without_self.j.sample_idx)
    assert without_self_self_kin_only.count() == 0, without_self_self_kin_only.collect()


@skip_when_service_backend(reason='intermittent tolerance failures')
@test_timeout(local=6 * 60, batch=10 * 60)
def test_pc_relate_issue_5263():
    mt = hl.balding_nichols_model(3, 50, 100)
    expected = hl.pc_relate(mt.GT, 0.10, k=2, statistics='all')
    mt = mt.select_entries(GT2=mt.GT,
                           GT=hl.call(hl.rand_bool(0.5), hl.rand_bool(0.5)))
    actual = hl.pc_relate(mt.GT2, 0.10, k=2, statistics='all')
    assert expected._same(actual, tolerance=1e-3)
