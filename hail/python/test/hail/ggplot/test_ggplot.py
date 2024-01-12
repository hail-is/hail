import hail as hl
from hail.ggplot import *
import numpy as np
import math


def test_geom_point_line_text_col_area():
    ht = hl.utils.range_table(20)
    ht = ht.annotate(double=ht.idx * 2)
    ht = ht.annotate(triple=ht.idx * 3)
    fig = (
        ggplot(ht, aes(x=ht.idx))
        + aes(y=ht.double)
        + geom_point()
        + geom_line(aes(y=ht.triple))
        + geom_text(aes(label=hl.str(ht.idx)))
        + geom_col(aes(y=ht.triple + ht.double))
        + geom_area(aes(y=ht.triple - ht.double))
        + coord_cartesian((0, 100), (0, 80))
        + xlab("my_x")
        + ylab("my_y")
        + ggtitle("Title")
    )
    fig.to_plotly()


def test_manhattan_plot():
    mt = hl.balding_nichols_model(3, 10, 100)
    ht = mt.rows()
    ht = ht.annotate(pval=0.02)
    fig = ggplot(ht, aes(x=ht.locus, y=-hl.log10(ht.pval))) + geom_point() + geom_hline(yintercept=-math.log10(5e-8))
    pfig = fig.to_plotly()
    expected_ticks = (
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        '10',
        '11',
        '12',
        '13',
        '14',
        '15',
        '16',
        '17',
        '18',
        '19',
        '20',
        '21',
        '22',
        'X',
        'Y',
    )
    assert pfig.layout.xaxis.ticktext == expected_ticks


def test_histogram():
    num_rows = 101
    num_groups = 5
    ht = hl.utils.range_table(num_rows)
    ht = ht.annotate(mod_3=hl.str(ht.idx % num_groups))
    for position in ["stack", "dodge", "identity"]:
        fig = ggplot(ht, aes(x=ht.idx)) + geom_histogram(aes(fill=ht.mod_3), alpha=0.5, position=position, bins=10)
        pfig = fig.to_plotly()
        assert len(pfig.data) == num_groups
        for idx, bar in enumerate(pfig.data):
            if position in {"stack", "identity"}:
                assert (bar.x == [float(e) for e in range(num_groups, num_rows - 1, num_groups * 2)]).all()
            else:
                dist_between_bars_in_one_group = (num_rows - 1) / (num_groups * 2)
                single_bar_width = dist_between_bars_in_one_group / num_groups
                first_bar_start = single_bar_width / 2 + idx * single_bar_width
                assert (bar.x == np.arange(first_bar_start, num_rows - 1, dist_between_bars_in_one_group)).all()


def test_separate_traces_per_group():
    ht = hl.utils.range_table(30)
    fig = ggplot(ht, aes(x=ht.idx)) + geom_bar(aes(fill=hl.str(ht.idx)))
    assert len(fig.to_plotly().data) == 30


def test_geom_ribbon():
    ht = hl.utils.range_table(20)
    fig = ggplot(ht, aes(x=ht.idx, ymin=ht.idx * 2, ymax=ht.idx * 3)) + geom_ribbon()
    fig.to_plotly()


def test_default_scale_no_repeat_colors():
    num_rows = 20
    ht = hl.utils.range_table(num_rows)
    fig = ggplot(ht, aes(x=ht.idx, y=ht.idx, color=hl.str(ht.idx))) + geom_point()
    pfig = fig.to_plotly()

    scatter_colors = [scatter['marker']['color'] for scatter in pfig['data']]
    num_unique_colors = len(set(scatter_colors))
    assert num_unique_colors == num_rows


def test_scale_color_manual():
    num_rows = 4
    colors = set(["red", "blue"])
    ht = hl.utils.range_table(num_rows)
    fig = (
        ggplot(ht, aes(x=ht.idx, y=ht.idx, color=hl.str(ht.idx % 2)))
        + geom_point()
        + scale_color_manual(values=list(colors))
    )
    pfig = fig.to_plotly()

    assert set([scatter.marker.color for scatter in pfig.data]) == colors


def test_weighted_bar():
    x = hl.array([2, 3, 3, 3, 4, 5, 2])
    w = hl.array([1, 2, 3, 4, 5, 6, 7])
    ht = hl.utils.range_table(7)
    ht = ht.annotate(x=x[ht.idx], w=w[ht.idx])
    fig = ggplot(ht) + geom_bar(aes(x=ht.x, weight=ht.w))

    result = [8, 9, 5, 6]
    for idx, y in enumerate(fig.to_plotly().data[0].y):
        assert y == result[idx]


def test_faceting():
    ht = hl.utils.range_table(10)
    ht = ht.annotate(x=hl.if_else(ht.idx < 4, "less", "more"))
    pfig = (ggplot(ht) + geom_point(aes(x=ht.idx, y=ht.idx)) + facet_wrap(vars(ht.x))).to_plotly()

    assert len(pfig.layout.annotations) == 2


def test_matrix_tables():
    mt = hl.utils.range_matrix_table(3, 3)
    mt = mt.annotate_rows(row_doubled=mt.row_idx * 2)
    mt = mt.annotate_entries(entry_idx=mt.row_idx + mt.col_idx)
    for field, expected in [
        (mt.row_doubled, [(0, 0), (1, 2), (2, 4)]),
        (mt.entry_idx, [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (2, 4)]),
    ]:
        data = (ggplot(mt, aes(x=mt.row_idx, y=field)) + geom_point()).to_plotly().data[0]
        assert len(data.x) == len(expected)
        assert len(data.y) == len(expected)
        for idx, (x, y) in enumerate(zip(data.x, data.y)):
            assert x == expected[idx][0]
            assert y == expected[idx][1]
