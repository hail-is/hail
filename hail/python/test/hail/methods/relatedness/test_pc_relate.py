import hail as hl

from ...helpers import (
    qobtest,
    resource,
    skip_when_service_backend,
    skip_when_service_backend_in_azure,
    test_timeout,
)


@test_timeout(local=6 * 60, batch=14 * 60)
def test_pc_relate_against_R_truth():
    with hl.TemporaryDirectory(ensure_exists=False) as vcf_f, hl.TemporaryDirectory(ensure_exists=False) as hail_kin_f:
        mt = hl.import_vcf(resource("pc_relate_bn_input.vcf.bgz")).checkpoint(vcf_f)
        hail_kin = hl.pc_relate(mt.GT, 0.00, k=2).checkpoint(hail_kin_f)

        with hl.TemporaryDirectory(ensure_exists=False) as r_kin_f:
            r_kin = hl.import_table(
                resource("pc_relate_r_truth.tsv.bgz"),
                types={
                    "i": "struct{s:str}",
                    "j": "struct{s:str}",
                    "kin": "float",
                    "ibd0": "float",
                    "ibd1": "float",
                    "ibd2": "float",
                },
                key=["i", "j"],
            ).checkpoint(r_kin_f)
            assert r_kin.select("kin")._same(hail_kin.select("kin"), tolerance=1e-3, absolute=True)
            assert r_kin.select("ibd0")._same(hail_kin.select("ibd0"), tolerance=1.3e-2, absolute=True)
            assert r_kin.select("ibd1")._same(hail_kin.select("ibd1"), tolerance=2.6e-2, absolute=True)
            assert r_kin.select("ibd2")._same(hail_kin.select("ibd2"), tolerance=1.3e-2, absolute=True)


@qobtest
def test_pc_relate_simple_example():
    gs = hl.literal([
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
    ])
    scores = hl.literal([[1, 1], [-1, 0], [1, -1], [-1, 0]])
    mt = hl.utils.range_matrix_table(n_rows=8, n_cols=4)
    mt = mt.annotate_entries(GT=hl.unphased_diploid_gt_index_call(gs[mt.col_idx][mt.row_idx]))
    mt = mt.annotate_cols(scores=scores[mt.col_idx])
    pcr = hl.pc_relate(mt.GT, min_individual_maf=0, scores_expr=mt.scores)

    expected = [
        hl.Struct(i=0, j=1, kin=0.0, ibd0=1.0, ibd1=0.0, ibd2=0.0),
        hl.Struct(i=0, j=2, kin=0.0, ibd0=1.0, ibd1=0.0, ibd2=0.0),
        hl.Struct(i=0, j=3, kin=0.0, ibd0=1.0, ibd1=0.0, ibd2=0.0),
        hl.Struct(i=1, j=2, kin=0.0, ibd0=1.0, ibd1=0.0, ibd2=0.0),
        hl.Struct(i=1, j=3, kin=0.0, ibd0=1.0, ibd1=0.0, ibd2=0.0),
        hl.Struct(i=2, j=3, kin=0.0, ibd0=1.0, ibd1=0.0, ibd2=0.0),
    ]
    ht_expected = hl.Table.parallelize(expected)
    ht_expected = ht_expected.key_by(i=hl.struct(col_idx=ht_expected.i), j=hl.struct(col_idx=ht_expected.j))
    assert ht_expected._same(pcr, tolerance=1e-12, absolute=True)


@test_timeout(6 * 60, batch=14 * 60)
@skip_when_service_backend_in_azure(reason="takes >14 minutes in QoB in Azure")
def test_pc_relate_paths_1():
    with hl.TemporaryDirectory(ensure_exists=False) as bn_f, hl.TemporaryDirectory(
        ensure_exists=False
    ) as scores_f, hl.TemporaryDirectory(ensure_exists=False) as kin1_f, hl.TemporaryDirectory(
        ensure_exists=False
    ) as kins1_f:
        mt = hl.balding_nichols_model(3, 50, 100).checkpoint(bn_f)
        _, scores3, _ = hl._hwe_normalized_blanczos(mt.GT, k=3, compute_loadings=False, q_iterations=10)
        scores3 = scores3.checkpoint(scores_f)

        kin1 = hl.pc_relate(mt.GT, 0.10, k=2, statistics="kin", block_size=64).checkpoint(kin1_f)
        kin_s1 = hl.pc_relate(
            mt.GT,
            0.10,
            scores_expr=scores3[mt.col_key].scores[:2],
            statistics="kin",
            block_size=64,
        ).checkpoint(kins1_f)

        assert kin1._same(kin_s1, tolerance=1e-4)
        assert kin1.count() == 50 * 49 / 2


@test_timeout(6 * 60, batch=14 * 60)
def test_pc_relate_paths_2():
    mt = hl.balding_nichols_model(3, 50, 100).cache()

    kin2 = hl.pc_relate(mt.GT, 0.05, k=2, min_kinship=0.01, statistics="kin2", block_size=128).cache()
    assert kin2.count() > 0
    assert kin2.filter(kin2.kin < 0.01).count() == 0


@test_timeout(6 * 60, batch=14 * 60)
def test_pc_relate_paths_3():
    mt = hl.balding_nichols_model(3, 50, 100).cache()

    kin3 = hl.pc_relate(mt.GT, 0.02, k=3, min_kinship=0.1, statistics="kin20", block_size=64).cache()
    assert kin3.count() > 0
    assert kin3.filter(kin3.kin < 0.1).count() == 0


@test_timeout(6 * 60, batch=14 * 60)
def test_self_kinship_1():
    mt = hl.balding_nichols_model(3, 10, 50).cache()
    with hl.TemporaryDirectory(ensure_exists=False) as f:
        with_self = hl.pc_relate(
            mt.GT, 0.10, k=2, statistics="kin", block_size=16, include_self_kinship=True
        ).checkpoint(f)
        assert with_self.count() == 55
        with_self_self_kin_only = with_self.filter(with_self.i.sample_idx == with_self.j.sample_idx)
        assert with_self_self_kin_only.count() == 10, with_self_self_kin_only.collect()


@test_timeout(6 * 60, batch=14 * 60)
def test_self_kinship_2():
    mt = hl.balding_nichols_model(3, 10, 50).cache()
    with hl.TemporaryDirectory(ensure_exists=False) as f:
        without_self = hl.pc_relate(mt.GT, 0.10, k=2, statistics="kin", block_size=16).checkpoint(f)
        assert without_self.count() == 45
        without_self_self_kin_only = without_self.filter(without_self.i.sample_idx == without_self.j.sample_idx)
        assert without_self_self_kin_only.count() == 0, without_self_self_kin_only.collect()


@test_timeout(6 * 60, batch=14 * 60)
def test_self_kinship_3():
    mt = hl.balding_nichols_model(3, 10, 50).cache()
    with hl.TemporaryDirectory(ensure_exists=False) as with_self_f, hl.TemporaryDirectory(
        ensure_exists=False
    ) as without_self_f:
        with_self = hl.pc_relate(
            mt.GT,
            0.10,
            k=2,
            statistics="kin20",
            block_size=16,
            include_self_kinship=True,
        ).checkpoint(with_self_f)
        without_self = hl.pc_relate(mt.GT, 0.10, k=2, statistics="kin20", block_size=16).checkpoint(without_self_f)

        with_self_no_self_kin = with_self.filter(with_self.i.sample_idx != with_self.j.sample_idx)
        assert with_self_no_self_kin._same(without_self)


@skip_when_service_backend(reason="intermittent tolerance failures")
@test_timeout(local=6 * 60, batch=14 * 60)
def test_pc_relate_issue_5263():
    mt = hl.balding_nichols_model(3, 50, 100)
    expected = hl.pc_relate(mt.GT, 0.10, k=2, statistics="all")
    mt = mt.select_entries(GT2=mt.GT, GT=hl.call(hl.rand_bool(0.5), hl.rand_bool(0.5)))
    actual = hl.pc_relate(mt.GT2, 0.10, k=2, statistics="all")
    assert expected._same(actual, tolerance=1e-3)
