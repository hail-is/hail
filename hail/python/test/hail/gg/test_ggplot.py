# These tests only check that the functions don't error out, they don't check what the output plot looks like.
import hail as hl
from hail.gg import *


def test_geom_point_line_text():
    ht = hl.utils.range_table(20)
    ht = ht.annotate(double=ht.idx * 2)
    ht = ht.annotate(triple=ht.idx * 3)
    fig = ggplot(ht, aes(x=ht.idx)) + aes(y=ht.double) + geom_point() + geom_line(aes(y=ht.triple))
    fig.render()


