# TODO
# qqplot only needs the line defined from 0,0 to 1,1
# add qqplot and manhattan plot to ggplot
# show how we compose ggplot stuff into a manhattan plot, bespoke func but here's how you can hack on it
# https://danielroelfs.com/blog/how-i-create-manhattan-plots-using-ggplot/
# TODO make bespoke funcs that are drop in replacements for hail.plot so we can just delete it tbh
# TODO
# wrapper method for table, object caches aggregations (possibly keyed by hash of IR); keep track of most recent
# debug output for this object should demonstrate what can be plotted without recomputation in some way (maybe these are ultimately equivalent to stats in ggplot, user-readable string can be name of ggplot method; maybe point to lines of user's code from aggregation cache so they can see which things in their code don't need to be recomputed)
# qqplot is a composition of aggregations which should also be individually exposed
# eviction policy for cached results (after n new aggregations)
# might need to wrap altair to make it look more like ggplot for adoption reasons

from typing import Any, Callable, List, Union

from hail import MatrixTable, Table
from hail.expr import Expression


from dataclasses import dataclass

import altair_viewer as altv
import altair as alt
import pandas as pd

from hail.expr import literal


alt.data_transformers.disable_max_rows()


Aesthetic = dict[str, Expression]
PlotTable = Union[Table, MatrixTable]


def aes(**kwargs: Any) -> Aesthetic:
    return {kw: arg if isinstance(arg, Expression) else literal(arg) for kw, arg in kwargs.items()}


_Plot = "Plot"


@dataclass(frozen=True)
class Plot:
    _data: PlotTable
    _aes: Aesthetic
    # TODO this should be a property and a list of aggs not a single df
    _df: pd.DataFrame
    _transformations: List[Callable[[_Plot], _Plot]] = None

    def __add__(self: _Plot, other: Callable[[_Plot], _Plot]) -> _Plot:
        return other(self)


def ggplot(_data: PlotTable, _aes: dict[str, Any]) -> Plot:
    __aes = aes(**_aes)
    return Plot(_data, __aes, pd.DataFrame(hl.struct(**__aes).collect()))


# TODO stat should definitely be a class at this point (have a base class for anything that can be added with apply_df and apply_chart methods
def stat_qq():
    def stat(plot: Plot) -> Plot:
        try:
            sample = plot._aes["sample"]
        except KeyError:
            raise ValueError("Unable to compute `stat_qq`: no value was provided for aesthetic `sample`.")

        ht = sample._indices.source.select(p_value=sample).key_by().select('p_value').key_by('p_value').persist()
        ht = ht.annotate(
            observed_p=-hl.log10(ht['p_value']),
            expected_p=-hl.log10((hl.scan.count() + 1) / ht.count()),
        ).persist()
        ht = ht.annotate(fit=ht.expected_p).persist()
        df = pd.DataFrame(hl.struct(x=ht.expected_p, y=ht.observed_p, fit=ht.fit).collect())

        def transformation(chart: alt.Chart) -> alt.Chart:
            return chart.encode(
                x=alt.X("x", axis=alt.Axis(title="Expected -log10(p)"), scale=alt.Scale(domain=[0, 10])),
                y=alt.Y("y", axis=alt.Axis(title="Observed -log10(p)"), scale=alt.Scale(domain=[0, 10])),
            ).mark_point()

        return plot(plot._data, plot._aes, df, [*plot._transformations, transformation])

    return stat


def stat_qq_line():
    def stat(plot: Plot) -> Plot:
        def transformation(chart: alt.Chart) -> alt.Chart:
            return chart.encode(x="x", y="fit").mark_line(color="red")

        # TODO compute agg if stat_qq hasn't already been applied
        return plot

    return stat


def qqplot(_data: PlotTable, _aes: Aesthetic) -> Plot:
    return ggplot(_data, _aes) + stat_qq() + stat_qq_line()


def to_chart(plot: Plot) -> alt.Chart:
    chart = alt.Chart(plot._df)
    for transformation in plot._transformations:
        chart = transformation(chart)
    return chart


def show(plot: Plot) -> None:
    altv.display(to_chart(plot))


# example usage with data from notebook
import hail as hl


hl.init()
hl.utils.get_1kg('data/')
mt = hl.read_matrix_table('data/1kg.mt')
mt = mt.annotate_cols(**(hl.import_table('data/1kg_annotations.txt', impute=True).key_by('Sample')[mt.s]))
mt = hl.sample_qc(mt)
mt = mt.filter_cols((mt.sample_qc.dp_stats.mean >= 4) & (mt.sample_qc.call_rate >= 0.97))
ab = mt.AD[1] / hl.sum(mt.AD)
mt = mt.filter_entries(
    (mt.GT.is_hom_ref() & (ab <= 0.1))
    | (mt.GT.is_het() & (ab >= 0.25) & (ab <= 0.75))
    | (mt.GT.is_hom_var() & (ab >= 0.9))
)
mt = hl.variant_qc(mt).cache()
mt = mt.filter_rows(mt.variant_qc.AF[1] > 0.01)
gwas = hl.linear_regression_rows(y=mt.CaffeineConsumption, x=mt.GT.n_alt_alleles(), covariates=[1.0])

# ggplot style
_plot = ggplot(gwas, aes(sample=gwas.p_value))
_qqplot = _plot + stat_qq() + stat_qq_line()
show(_qqplot)

# bespoke style
_qqplot = qqplot(gwas, gwas.p_value)
show(_qqplot)
