# These tests only check that the functions don't error out, they don't check what the output plot looks like.
import hail as hl
from hail.ggplot import *
import numpy as np
import math
from ..helpers import fails_service_backend


def test_geom_point_line_text_col_area():
    ht = hl.utils.range_table(20)
    ht = ht.annotate(double=ht.idx * 2)
    ht = ht.annotate(triple=ht.idx * 3)
    fig = (ggplot(ht, aes(x=ht.idx)) +
           aes(y=ht.double) +
           geom_point() +
           geom_line(aes(y=ht.triple)) +
           geom_text(aes(label=hl.str(ht.idx))) +
           geom_col(aes(y=ht.triple + ht.double)) +
           geom_area(aes(y=ht.triple - ht.double)) +
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
    pfig = fig.to_plotly()
    expected_ticks = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y')
    assert pfig.layout.xaxis.ticktext == expected_ticks

@fails_service_backend()
def test_histogram():
    num_rows = 101
    num_groups = 5
    ht = hl.utils.range_table(num_rows)
    ht = ht.annotate(mod_3=hl.str(ht.idx % num_groups))
    for position in ["stack", "dodge", "identity"]:
        fig = (ggplot(ht, aes(x=ht.idx)) +
               geom_histogram(aes(fill=ht.mod_3), alpha=0.5, position=position, bins=10)
               )
        pfig = fig.to_plotly()
        assert len(pfig.data) == num_groups
        for idx, bar in enumerate(pfig.data):
            if position in {"stack", "identity"}:
                assert (bar.x == [float(e) for e in range(num_groups, num_rows-1, num_groups*2)]).all()
            else:
                dist_between_bars_in_one_group = (num_rows - 1) / (num_groups * 2)
                single_bar_width = (dist_between_bars_in_one_group / num_groups)
                first_bar_start = single_bar_width / 2 + idx * single_bar_width
                assert (bar.x == np.arange(first_bar_start, num_rows - 1, dist_between_bars_in_one_group)).all()


def test_separate_traces_per_group():
    ht = hl.utils.range_table(30)
    fig = (ggplot(ht, aes(x=ht.idx)) +
           geom_bar(aes(fill=hl.str(ht.idx)))
           )
    assert len(fig.to_plotly().data) == 30


def test_geom_ribbon():
    ht = hl.utils.range_table(20)
    fig = ggplot(ht, aes(x=ht.idx, ymin=ht.idx * 2, ymax=ht.idx * 3)) + geom_ribbon()
    fig.to_plotly()
