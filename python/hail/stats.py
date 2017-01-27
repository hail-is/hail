class BetaDist:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class UniformDist(object):
    def __init__(self, minVal, maxVal):
        self.minVal = minVal
        self.maxVal = maxVal
        if minVal > maxVal:
            raise ValueError("min must be less than max")

class TruncatedBetaDist:
    def __init__(self, a, b, minVal, maxVal):
        self.minVal = minVal
        self.maxVal = maxVal
        if minVal > maxVal:
            raise ValueError("min must be less than max")