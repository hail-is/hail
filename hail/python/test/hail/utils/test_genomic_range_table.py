import hail as hl

from ..helpers import run_in


@run_in('local')
def test_genomic_range_table_grch38():
    actual = hl.utils.genomic_range_table(10, reference_genome='GRCh38').collect()
    expected = [hl.Struct(locus=hl.Locus("chr1", pos + 1, reference_genome='GRCh38'))
                for pos in range(10)]
    assert actual == expected


@run_in('local')
def test_genomic_range_table_grch37():
    actual = hl.utils.genomic_range_table(10, reference_genome='GRCh37').collect()
    expected = [hl.Struct(locus=hl.Locus("1", pos + 1, reference_genome='GRCh37'))
                for pos in range(10)]
    assert actual == expected


@run_in('local')
def test_genomic_range_table_canfam3():
    actual = hl.utils.genomic_range_table(10, reference_genome='CanFam3').collect()
    expected = [hl.Struct(locus=hl.Locus("chr1", pos + 1, reference_genome='CanFam3'))
                for pos in range(10)]
    assert actual == expected


@run_in('local')
def test_genomic_range_table_grcm38():
    actual = hl.utils.genomic_range_table(10, reference_genome='GRCm38').collect()
    expected = [hl.Struct(locus=hl.Locus("1", pos + 1, reference_genome='GRCm38'))
                for pos in range(10)]
    assert actual == expected
