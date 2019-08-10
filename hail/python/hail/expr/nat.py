import abc


class NatBase(object):
    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError

    @abc.abstractmethod
    def unify(self, t):
        raise NotImplementedError

    @abc.abstractmethod
    def subst(self):
        raise NotImplementedError


class NatLiteral(NatBase):
    def __init__(self, n):
        self.n = n

    def clear(self):
        pass

    def unify(self, t):
        return isinstance(t, NatLiteral) and t.n == self.n

    def subst(self):
        return self


class NatVariable(NatBase):
    _nat = None

    def clear(self):
        NatVariable._nat = None

    def unify(self, other):
        assert isinstance(other, NatBase)
        if NatVariable._nat is not None:
            return NatVariable._nat.unify(other)
        else:
            NatVariable._nat = other
            return True

    def subst(self):
        assert NatVariable._nat is not None
        return NatVariable._nat
