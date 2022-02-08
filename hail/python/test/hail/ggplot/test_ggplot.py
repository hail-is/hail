# These tests only check that the functions don't error out, they don't check what the output plot looks like.
import hail as hl
from hail.ggplot import *
import math


def test_geom_point_line_text_col():
    ht = hl.utils.range_table(20)
    ht = ht.annotate(double=ht.idx * 2)
    ht = ht.annotate(triple=ht.idx * 3)
    fig = (ggplot(ht, aes(x=ht.idx)) +
           aes(y=ht.double) +
           geom_point() +
           geom_line(aes(y=ht.triple)) +
           geom_text(aes(label=hl.str(ht.idx))) +
           geom_col(aes(y=ht.triple + ht.double)) +
           coord_cartesian((0, 100), (0, 80)) +
           xlab("my_x") +
           ylab("my_y") +
           ggtitle("Title")
           )
    fig.to_plotly()


def test_manhattan_plot():
    mt = hl.balding_nichols_model(3, 10, 100)
    ht = mt.rows()
    ht = ht.annotate(pval=.02)
    fig = ggplot(ht, aes(x=ht.locus, y=-hl.log10(ht.pval))) + geom_point() + geom_hline(yintercept=-math.log10(5e-8))
    fig.to_plotly()


def test_histogram():
    ht = hl.utils.range_table(10)
    fig = (ggplot(ht, aes(x=ht.idx)) +
           geom_histogram(alpha=0.5)
           )
    fig.to_plotly()