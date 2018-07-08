from collections.abc import MutableMapping

class Bidict(MutableMapping):
    def __init__(self):
        self.d = {}
        self.inverse = {}

    def revgetitem(self, value):
        return self.inverse[value]

    def revcontains(self, value):
        return value in self.inverse

    def __getitem__(self, key):
        return self.d[key]

    def __setitem__(self, key, value):
        if key in self.d:
            old_value = self.d[key]
            del self.inverse[old_value]

        self.d[key] = value
        assert value not in self.inverse
        self.inverse[value] = key

    def __delitem__(self, key):
        if key in self.d:
            old_value = self.d[key]
            del self.inverse[old_value]
            del self.d[key]

    def __iter__(self):
        return iter(self.d)

    def __len__(self):
        return len(self.d)
