import abc

import pandas as pd

import hail as hl
from hail.utils.java import warning
from .utils import should_use_for_grouping


class Stat:
    @abc.abstractmethod
    def make_agg(self, mapping, precomputed):
        return

    @abc.abstractmethod
    def listify(self, agg_result):
        # Turns the agg result into a list of data frames to be plotted.
        return

    def get_precomputes(self, mapping):
        return hl.struct()


class StatIdentity(Stat):

    def make_agg(self, mapping, precomputed):
        grouping_variables = {aes_key: mapping[aes_key] for aes_key in mapping.keys()
                              if should_use_for_grouping(aes_key, mapping[aes_key].dtype)}
        non_grouping_variables = {aes_key: mapping[aes_key] for aes_key in mapping.keys() if aes_key not in grouping_variables}
        return hl.agg.group_by(hl.struct(**grouping_variables), hl.agg.collect(hl.struct(**non_grouping_variables)))

    def listify(self, agg_result):
        result = []
        for grouped_struct, collected in agg_result.items():
            columns = list(collected[0].keys())
            data_dict = {}

            for column in columns:
                col_data = [row[column] for row in collected]
                data_dict[column] = pd.Series(col_data)

            df = pd.DataFrame(data_dict)
            df.attrs.update(**grouped_struct)
            result.append(df)
        return result


class StatFunction(StatIdentity):

    def __init__(self, fun):
        self.fun = fun

    def make_agg(self, mapping, precomputed):
        with_y_value = mapping.annotate(y=self.fun(mapping.x))
        return super().make_agg(with_y_value, precomputed)


class StatNone(Stat):
    def make_agg(self, mapping, precomputed):
        return hl.struct()

    def listify(self, agg_result):
        return pd.DataFrame({})


class StatCount(Stat):
    def make_agg(self, mapping, precomputed):
        grouping_variables = {aes_key: mapping[aes_key] for aes_key in mapping.keys()
                              if should_use_for_grouping(aes_key, mapping[aes_key].dtype)}
        return hl.agg.group_by(hl.struct(**grouping_variables), hl.agg.group_by(mapping["x"], hl.agg.count()))

    def listify(self, agg_result):
        result = []
        for grouped_struct, count_by_x in agg_result.items():
            data_dict = {}
            xs, counts = zip(*count_by_x.items())
            data_dict["x"] = pd.Series(xs)
            data_dict["y"] = pd.Series(counts)

            df = pd.DataFrame(data_dict)
            df.attrs.update(**grouped_struct)
            result.append(df)

        return result


class StatBin(Stat):
    DEFAULT_BINS = 30

    def __init__(self, min_val, max_val, bins):
        self.min_val = min_val
        self.max_val = max_val
        self.bins = bins

    def get_precomputes(self, mapping):

        precomputes = {}
        if self.min_val is None:
            precomputes["min_val"] = hl.agg.min(mapping.x)
        if self.max_val is None:
            precomputes["max_val"] = hl.agg.max(mapping.x)
        return hl.struct(**precomputes)

    def make_agg(self, mapping, precomputed):
        grouping_variables = {aes_key: mapping[aes_key] for aes_key in mapping.keys()
                              if should_use_for_grouping(aes_key, mapping[aes_key].dtype)}

        start = self.min_val if self.min_val is not None else precomputed.min_val
        end = self.max_val if self.max_val is not None else precomputed.max_val
        if self.bins is None:
            warning(f"No number of bins was specfied for geom_histogram, defaulting to {self.DEFAULT_BINS} bins")
            bins = self.DEFAULT_BINS
        else:
            bins = self.bins
        return hl.agg.group_by(hl.struct(**grouping_variables), hl.agg.hist(mapping["x"], start, end, bins))

    def listify(self, agg_result):
        items = list(agg_result.items())
        x_edges = items[0][1].bin_edges
        num_edges = len(x_edges)

        result = []

        for grouped_struct, hist in items:
            data_rows = []
            y_values = hist.bin_freq
            for i, x in enumerate(x_edges[:num_edges - 1]):
                data_rows.append({"x": x, "y": y_values[i]})
            df = pd.DataFrame.from_records(data_rows)
            df.attrs.update(**grouped_struct)
            result.append(df)
        return result
