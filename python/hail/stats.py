class BetaDist:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class UniformDist(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        if (a > b):
            raise ValueError("a cannot be greater than b")