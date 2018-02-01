from hail.history import *
from hail.typecheck import *
from hail.utils.java import Env


class BetaDist(HistoryMixin):
    """
    Represents a
    `beta distribution <https://en.wikipedia.org/wiki/Beta_distribution>`__
    with parameters a and b.
    """

    @typecheck_method(a=numeric,
                      b=numeric)
    @record_init
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def _jrep(self):
        return Env.hail().stats.BetaDist.apply(float(self.a), float(self.b))


class UniformDist(HistoryMixin):
    """
    Represents a uniform distribution on the interval [min, max].
    """

    @typecheck_method(min=numeric,
                      max=numeric)
    @record_init
    def __init__(self, min, max):
        if min >= max:
            raise ValueError("min must be less than max")
        self.min = min
        self.max = max

    def _jrep(self):
        return Env.hail().stats.UniformDist.apply(float(self.min), float(self.max))


class TruncatedBetaDist(HistoryMixin):
    """
    Represents a truncated
    `beta distribution <https://en.wikipedia.org/wiki/Beta_distribution>`__
    with parameters a and b and support [min, max]. Draws are made
    via rejection sampling, i.e. returning the first draw from Beta(a,b) that falls in [min, max].
    This procedure may be slow if the probability mass of Beta(a,b) over [min, max] is small.
    """
    @typecheck_method(a=numeric,
                      b=numeric,
                      min=numeric,
                      max=numeric)
    @record_init
    def __init__(self, a, b, min, max):
        if min >= max:
            raise ValueError("min must be less than max")
        elif min < 0:
            raise ValueError("min cannot be less than 0")
        elif max > 1:
            raise ValueError("max cannot be greater than 1")

        self.min = min
        self.max = max
        self.a = a
        self.b = b

    def _jrep(self):
        return Env.hail().stats.TruncatedBetaDist.apply(float(self.a), float(self.b), float(self.min), float(self.max))
