class BetaDist:
    """
    Represents a beta distribution with parameters a and b.
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b


class UniformDist:
    """
    Represents a uniform distribution on the interval [minVal, maxVal].
    """
    def __init__(self, minVal, maxVal):
        if minVal >= maxVal:
            raise ValueError("min must be less than max")
        self.minVal = minVal
        self.maxVal = maxVal

class TruncatedBetaDist:
    """
    Represents a beta distribution that has a lower and upper bound on values drawn from it. Note that this is accomplished
    by rejection sampling, so if minVal and maxVal are very close together or the probability of a value being drawn between
    minVal and maxVal is very low, a program using this distribution could take a very long time to execute.
    """
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