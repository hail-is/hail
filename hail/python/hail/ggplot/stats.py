import abc

import hail as hl
from hail.utils.java import warning
from .utils import is_continuous_type


class Stat:
    @abc.abstractmethod
    def make_agg(self, mapping, precomputed):
        return

    @abc.abstractmethod
    def listify(self, agg_result):
        # Turns the agg result into a data list to be plotted.
        return

    def get_precomputes(self, mapping):
        return hl.struct()


class StatIdentity(Stat):
    def make_agg(self, mapping, precomputed):
        return hl.agg.collect(mapping)

    def listify(self, agg_result):
        # Collect aggregator returns a list, nothing to do.
        return agg_result


class StatFunction(Stat):

    def __init__(self, fun):
        self.fun = fun

    def make_agg(self, combined, precomputed):
        with_y_value = combined.annotate(y=self.fun(combined.x))
        return hl.agg.collect(with_y_value)

    def listify(self, agg_result):
        # Collect aggregator returns a list, nothing to do.
        return agg_result


class StatNone(Stat):
    def make_agg(self, mapping, precomputed):
        return hl.struct()

    def listify(self, agg_result):
        return []


class StatCount(Stat):
    def make_agg(self, mapping, precomputed):
        discrete_variables = {aes_key: mapping[aes_key] for aes_key in mapping.keys()
                              if not is_continuous_type(mapping[aes_key].dtype)}
        discrete_variables["x"] = mapping["x"]
        return hl.agg.group_by(hl.struct(**discrete_variables), hl.agg.count())

    def listify(self, agg_result):
        unflattened_items = agg_result.items()
        res = []
        for discrete_variables, count in unflattened_items:
            arg_dict = {key: value for key, value in discrete_variables.items()}
            arg_dict["y"] = count
            new_struct = hl.Struct(**arg_dict)
            res.append(new_struct)
        return res


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
        discrete_variables = {aes_key: mapping[aes_key] for aes_key in mapping.keys()
                              if not is_continuous_type(mapping[aes_key].dtype)}

        start = self.min_val if self.min_val is not None else precomputed.min_val
        end = self.max_val if self.max_val is not None else precomputed.max_val
        if self.bins is None:
            warning(f"No number of bins was specfied for geom_histogram, defaulting to {self.DEFAULT_BINS} bins")
            bins = self.DEFAULT_BINS
        else:
            bins = self.bins
        return hl.agg.group_by(hl.struct(**discrete_variables), hl.agg.hist(mapping["x"], start, end, bins))

    def listify(self, agg_result):
        items = list(agg_result.items())
        x_edges = items[0][1].bin_edges
        num_edges = len(x_edges)
        data_rows = []
        for key, hist in items:
            y_values = hist.bin_freq
            for i, x in enumerate(x_edges[:num_edges - 1]):
                x_value = x
                data_rows.append(hl.Struct(x=x_value, y=y_values[i], **key))
        return data_rows
