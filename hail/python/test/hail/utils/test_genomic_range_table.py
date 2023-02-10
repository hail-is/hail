import hail as hl


def test_genomic_range_table():
    actual = hl.utils.genomic_range_table(10, reference_genome='GRCh38').collect()
    expected = [hl.Struct(locus=hl.locus("chr1", pos + 1))
                for pos in range(10)]
    assert actual == expected
