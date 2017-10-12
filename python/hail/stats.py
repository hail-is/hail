from hail.java import Env
from hail.typecheck import *
from hail.history import *


class BetaDist(HistoryMixin):
    """
    Represents a beta distribution with parameters a and b.
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
    Represents a uniform distribution on the interval [minVal, maxVal].
    """

    @typecheck_method(minVal=numeric,
                      maxVal=numeric)
    @record_init
    def __init__(self, minVal, maxVal):
        if minVal >= maxVal:
            raise ValueError("min must be less than max")
        self.minVal = minVal
        self.maxVal = maxVal

    def _jrep(self):
        return Env.hail().stats.UniformDist.apply(float(self.minVal), float(self.maxVal))


class TruncatedBetaDist(HistoryMixin):
    """
    Represents a truncated beta distribution with parameters a and b and support [minVal, maxVal]. Draws are made
    via rejection sampling, which may be slow if the probability mass of Beta(a,b) over [minVal, maxVal] is small.
    """
    @typecheck_method(a=numeric,
                      b=numeric,
                      minVal=numeric,
                      maxVal=numeric)
    @record_init
    def __init__(self, a, b, minVal, maxVal):
        if minVal >= maxVal:
            raise ValueError("min must be less than max")
        elif minVal < 0:
            raise ValueError("min cannot be less than 0")
        elif maxVal > 1:
            raise ValueError("max cannot be greater than 1")

        self.minVal = minVal
        self.maxVal = maxVal
        self.a = a
        self.b = b

    def _jrep(self):
        return Env.hail().stats.TruncatedBetaDist.apply(float(self.a), float(self.b), float(self.minVal), float(self.maxVal))
