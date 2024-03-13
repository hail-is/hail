import hail as hl
from hail.genetics import Locus


def test_constructor():
    l = Locus.parse('1:100')

    assert l == Locus('1', 100)
    assert l == Locus(1, 100)
    assert l.reference_genome == hl.default_reference()


def test_call_rich_comparison():
    val = Locus(1, 1)
    expr = hl.locus('1', 1)

    assert hl.eval(val == expr)
    assert hl.eval(expr == val)
