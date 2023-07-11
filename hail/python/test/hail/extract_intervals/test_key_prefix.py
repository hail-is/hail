import hail as hl

from ..helpers import run_in


@run_in('all')
def test_mt_lt(mt, probe_locus):
    expr = mt.filter_rows(mt.locus < probe_locus)
    assert expr.n_partitions() == 15
    assert expr.count() == (245, 100)


@run_in('all')
def test_mt_le(mt, probe_locus):
    expr = mt.filter_rows(mt.locus <= probe_locus)
    assert expr.n_partitions() == 15
    assert expr.count() == (246, 100)


@run_in('all')
def test_mt_eq(mt, probe_locus):
    expr = mt.filter_rows(mt.locus == probe_locus)
    assert expr.n_partitions() == 1
    actual = expr.GT.collect()
    expected = [hl.Call([0, int(i in (13, 17))]) for i in range(100)]
    assert actual == expected


@run_in('all')
def test_mt_ge(mt, probe_locus):
    expr = mt.filter_rows(mt.locus >= probe_locus)
    assert expr.n_partitions() == 6
    assert expr.count() == (101, 100)


@run_in('all')
def test_mt_gt(mt, probe_locus):
    expr = mt.filter_rows(mt.locus > probe_locus)
    assert expr.n_partitions() == 6
    assert expr.count() == (100, 100)


@run_in('all')
def test_ht_lt(ht, probe_locus):
    expr = ht.filter(ht.locus < probe_locus)
    assert expr.n_partitions() == 15
    assert expr.count() == 245


@run_in('all')
def test_ht_le(ht, probe_locus):
    expr = ht.filter(ht.locus <= probe_locus)
    assert expr.n_partitions() == 15
    assert expr.count() == 246


@run_in('all')
def test_ht_eq(ht, probe_locus):
    expr = ht.filter(ht.locus == probe_locus)
    assert expr.n_partitions() == 1
    actual = expr.collect()
    expected = [hl.Struct(
        locus=hl.Locus(contig=20, position=17434581, reference_genome='GRCh37'),
        alleles=['A', 'G'],
        rsid='rs16999198',
        qual=21384.8,
        filters=set(),
        info=hl.Struct(
            NEGATIVE_TRAIN_SITE=False,
            HWP=1.0,
            AC=[2],
            culprit='InbreedingCoeff',
            MQ0=0,
            ReadPosRankSum=0.534,
            AN=200,
            InbreedingCoeff=-0.0134,
            AF=[0.013],
            GQ_STDDEV=134.2,
            FS=2.944,
            DP=22586,
            GQ_MEAN=83.43,
            POSITIVE_TRAIN_SITE=True,
            VQSLOD=4.77,
            ClippingRankSum=0.175,
            BaseQRankSum=4.78,
            MLEAF=[0.013],
            MLEAC=[23],
            MQ=59.75,
            QD=14.65,
            END=None,
            DB=True,
            HaplotypeScore=None,
            MQRankSum=-0.192,
            CCC=1740,
            NCC=0,
            DS=False
        )
    )]
    assert actual == expected


@run_in('all')
def test_ht_ge(ht, probe_locus):
    expr = ht.filter(ht.locus >= probe_locus)
    assert expr.n_partitions() == 6
    assert expr.count() == 101


@run_in('all')
def test_ht_gt(ht, probe_locus):
    expr = ht.filter(ht.locus > probe_locus)
    assert expr.n_partitions() == 6
    assert expr.count() == 100
