from random import randint

import pytest

import hail as hl
from hail.genetics import ReferenceGenome
from hail.utils import FatalError

from ..helpers import qobtest, resource


@qobtest
def test_reference_genome():
    rg = hl.get_reference('GRCh37')
    assert rg.name == "GRCh37"
    assert rg.contigs[0] == "1"
    assert rg.x_contigs == ["X"]
    assert rg.y_contigs == ["Y"]
    assert rg.mt_contigs == ["MT"]
    assert rg.par[0] == hl.eval(hl.parse_locus_interval("X:60001-2699521"))
    assert rg.contig_length("1") == 249250621

    name = "test"
    contigs = ["1", "X", "Y", "MT"]
    lengths = {"1": 10000, "X": 2000, "Y": 4000, "MT": 1000}
    x_contigs = ["X"]
    y_contigs = ["Y"]
    mt_contigs = ["MT"]
    par = [("X", 5, 1000)]

    gr2 = ReferenceGenome(name, contigs, lengths, x_contigs, y_contigs, mt_contigs, par)
    assert gr2.name == name
    assert gr2.contigs == contigs
    assert gr2.x_contigs == x_contigs
    assert gr2.y_contigs == y_contigs
    assert gr2.mt_contigs == mt_contigs
    assert gr2.par == [hl.eval(hl.parse_locus_interval("X:5-1000", gr2))]
    assert gr2.contig_length("1") == 10000
    assert gr2.lengths == lengths

    with hl.TemporaryFilename() as filename:
        gr2.write(filename)


@qobtest
def test_reference_genome_sequence():
    gr3 = ReferenceGenome.read(resource("fake_ref_genome.json"))
    assert gr3.name == "my_reference_genome"
    assert not gr3.has_sequence()

    gr4 = ReferenceGenome.from_fasta_file(
        "test_rg",
        resource("fake_reference.fasta"),
        resource("fake_reference.fasta.fai"),
        mt_contigs=["b", "c"],
        x_contigs=["a"],
    )
    assert gr4.has_sequence()
    assert gr4._sequence_files == (resource("fake_reference.fasta"), resource("fake_reference.fasta.fai"))
    assert gr4.x_contigs == ["a"]

    t = hl.import_table(resource("fake_reference.tsv"), impute=True, min_partitions=4)
    assert hl.eval(t.all(hl.get_sequence(t.contig, t.pos, reference_genome=gr4) == t.base))

    l = hl.locus("a", 7, gr4)
    assert hl.eval(l.sequence_context(before=3, after=3) == "TTTCGAA")

    gr4.remove_sequence()
    assert not gr4.has_sequence()

    gr4.add_sequence(resource("fake_reference.fasta"), resource("fake_reference.fasta.fai"))
    assert gr4.has_sequence()
    assert gr4._sequence_files == (resource("fake_reference.fasta"), resource("fake_reference.fasta.fai"))


@qobtest
def test_reference_genome_liftover():
    grch37 = hl.get_reference('GRCh37')
    grch38 = hl.get_reference('GRCh38')

    assert not grch37.has_liftover('GRCh38')
    assert not grch38.has_liftover('GRCh37')
    grch37.add_liftover(resource('grch37_to_grch38_chr20.over.chain.gz'), 'GRCh38')
    grch38.add_liftover(resource('grch38_to_grch37_chr20.over.chain.gz'), 'GRCh37')
    assert grch37.has_liftover('GRCh38')
    assert grch38.has_liftover('GRCh37')
    assert grch37._liftovers == {'GRCh38': resource('grch37_to_grch38_chr20.over.chain.gz')}
    assert grch38._liftovers == {'GRCh37': resource('grch38_to_grch37_chr20.over.chain.gz')}

    ds = hl.import_vcf(resource('sample.vcf'))
    t = ds.annotate_rows(liftover=hl.liftover(hl.liftover(ds.locus, 'GRCh38'), 'GRCh37')).rows()
    assert t.all(t.locus == t.liftover)

    null_locus = hl.missing(hl.tlocus('GRCh38'))

    rows = [
        {'l37': hl.locus('20', 1, 'GRCh37'), 'l38': null_locus},
        {'l37': hl.locus('20', 60000, 'GRCh37'), 'l38': null_locus},
        {'l37': hl.locus('20', 60001, 'GRCh37'), 'l38': hl.locus('chr20', 79360, 'GRCh38')},
        {'l37': hl.locus('20', 278686, 'GRCh37'), 'l38': hl.locus('chr20', 298045, 'GRCh38')},
        {'l37': hl.locus('20', 278687, 'GRCh37'), 'l38': hl.locus('chr20', 298046, 'GRCh38')},
        {'l37': hl.locus('20', 278688, 'GRCh37'), 'l38': null_locus},
        {'l37': hl.locus('20', 278689, 'GRCh37'), 'l38': null_locus},
        {'l37': hl.locus('20', 278690, 'GRCh37'), 'l38': null_locus},
        {'l37': hl.locus('20', 278691, 'GRCh37'), 'l38': hl.locus('chr20', 298047, 'GRCh38')},
        {'l37': hl.locus('20', 37007586, 'GRCh37'), 'l38': hl.locus('chr12', 32563117, 'GRCh38')},
        {'l37': hl.locus('20', 62965520, 'GRCh37'), 'l38': hl.locus('chr20', 64334167, 'GRCh38')},
        {'l37': hl.locus('20', 62965521, 'GRCh37'), 'l38': null_locus},
    ]
    schema = hl.tstruct(l37=hl.tlocus(grch37), l38=hl.tlocus(grch38))
    t = hl.Table.parallelize(rows, schema)
    assert t.all(
        hl.if_else(
            hl.is_defined(t.l38), hl.liftover(t.l37, 'GRCh38') == t.l38, hl.is_missing(hl.liftover(t.l37, 'GRCh38'))
        )
    )

    t = t.filter(hl.is_defined(t.l38))
    assert t.count() == 6

    t = t.key_by('l38')
    t.count()
    assert list(t.key) == ['l38']

    null_locus_interval = hl.missing(hl.tinterval(hl.tlocus('GRCh38')))
    rows = [
        {'i37': hl.locus_interval('20', 1, 60000, True, False, 'GRCh37'), 'i38': null_locus_interval},
        {
            'i37': hl.locus_interval('20', 60001, 82456, True, True, 'GRCh37'),
            'i38': hl.locus_interval('chr20', 79360, 101815, True, True, 'GRCh38'),
        },
    ]
    schema = hl.tstruct(i37=hl.tinterval(hl.tlocus(grch37)), i38=hl.tinterval(hl.tlocus(grch38)))
    t = hl.Table.parallelize(rows, schema)
    assert t.all(hl.liftover(t.i37, 'GRCh38') == t.i38)

    grch37.remove_liftover("GRCh38")
    grch38.remove_liftover("GRCh37")


@qobtest
def test_liftover_strand():
    grch37 = hl.get_reference('GRCh37')
    grch37.add_liftover(resource('grch37_to_grch38_chr20.over.chain.gz'), 'GRCh38')

    try:
        actual = hl.eval(hl.liftover(hl.locus('20', 60001, 'GRCh37'), 'GRCh38', include_strand=True))
        expected = hl.Struct(result=hl.Locus('chr20', 79360, 'GRCh38'), is_negative_strand=False)
        assert actual == expected

        actual = hl.eval(
            hl.liftover(
                hl.locus_interval('20', 37007582, 37007586, True, True, 'GRCh37'), 'GRCh38', include_strand=True
            )
        )
        expected = hl.Struct(
            result=hl.Interval(
                hl.Locus('chr12', 32563117, 'GRCh38'),
                hl.Locus('chr12', 32563121, 'GRCh38'),
                includes_start=True,
                includes_end=True,
            ),
            is_negative_strand=True,
        )
        assert actual == expected

        with pytest.raises(FatalError):
            hl.eval(hl.liftover(hl.parse_locus_interval('1:10000-10000', reference_genome='GRCh37'), 'GRCh38'))
    finally:
        grch37.remove_liftover("GRCh38")


@qobtest
def test_read_custom_reference_genome():
    # this test doesn't behave properly if these reference genomes are already defined in scope.
    available_rgs = set(hl.current_backend()._references.keys())
    assert 'test_rg_0' not in available_rgs
    assert 'test_rg_1' not in available_rgs
    assert 'test_rg_2' not in available_rgs

    def assert_rg_loaded_correctly(name):
        rg = hl.get_reference(name)
        assert rg.contigs == ["1", "X", "Y", "MT"]
        assert rg.lengths == {"1": 5, "X": 4, "Y": 3, "MT": 2}
        assert rg.x_contigs == ["X"]
        assert rg.y_contigs == ["Y"]
        assert rg.mt_contigs == ["MT"]
        assert rg.par == [hl.Interval(start=hl.Locus("X", 2, name), end=hl.Locus("X", 4, name))]

    assert hl.read_table(resource('custom_references.t')).count() == 14
    assert_rg_loaded_correctly('test_rg_0')
    assert_rg_loaded_correctly('test_rg_1')

    # loading different reference genome with same name should fail
    # (different `test_rg_o` definition)
    with pytest.raises(FatalError):
        hl.read_matrix_table(resource('custom_references_2.t')).count()

    assert hl.read_matrix_table(resource('custom_references.mt')).count_rows() == 14
    assert_rg_loaded_correctly('test_rg_1')
    assert_rg_loaded_correctly('test_rg_2')


@qobtest
def test_custom_reference_read_write():
    hl.ReferenceGenome("dk", ['hello'], {"hello": 123})
    ht = hl.utils.range_table(5)
    ht = ht.key_by(locus=hl.locus('hello', ht.idx + 1, 'dk'))
    with hl.TemporaryDirectory(ensure_exists=False) as foo:
        ht.write(foo)
        expected = ht
        actual = hl.read_table(foo)
        assert actual._same(expected)


@qobtest
def test_locus_from_global_position():
    rg = hl.get_reference('GRCh37')
    max_length = rg.global_positions_dict[rg.contigs[-1]] + rg.lengths[rg.contigs[-1]]
    positions = [0, randint(1, max_length - 2), max_length - 1]

    python = [rg.locus_from_global_position(p) for p in positions]
    scala = hl.eval(hl.map(lambda p: hl.locus_from_global_position(p, rg), positions))

    assert python == scala


def test_locus_from_global_position_negative_pos():
    with pytest.raises(ValueError):
        hl.get_reference('GRCh37').locus_from_global_position(-1)


def test_locus_from_global_position_too_long():
    with pytest.raises(ValueError):
        hl.get_reference('GRCh37').locus_from_global_position(2**64 - 1)
