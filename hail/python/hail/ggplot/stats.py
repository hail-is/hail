import abc

import hail as hl
from .utils import is_continuous_type


class Stat:
    @abc.abstractmethod
    def make_agg(self, mapping):
        return

    @abc.abstractmethod
    def listify(self, agg_result):
        # Turns the agg result into a data list to be plotted.
        return


class StatIdentity(Stat):
    def make_agg(self, mapping):
        return hl.agg.collect(mapping)

    def listify(self, agg_result):
        # Collect aggregator returns a list, nothing to do.
        return agg_result


class StatFunction(Stat):

    def __init__(self, fun):
        self.fun = fun

    def make_agg(self, combined):
        with_y_value = combined.annotate(y=self.fun(combined.x))
        return hl.agg.collect(with_y_value)

    def listify(self, agg_result):
        # Collect aggregator returns a list, nothing to do.
        return agg_result


class StatNone(Stat):
    def make_agg(self, mapping):
        return hl.struct()

    def listify(self, agg_result):
        return []


class StatCount(Stat):
    def make_agg(self, mapping):
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

    def __init__(self, start, end, bins):
        self.start = start
        self.end = end
        self.bins = bins

    def make_agg(self, mapping):
        discrete_variables = {aes_key: mapping[aes_key] for aes_key in mapping.keys()
                              if not is_continuous_type(mapping[aes_key].dtype)}
        return hl.agg.group_by(hl.struct(**discrete_variables), hl.agg.hist(mapping["x"], self.start, self.end, self.bins))

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
