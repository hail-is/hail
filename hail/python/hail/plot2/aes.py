from collections.abc import Mapping

class Aesthetic(Mapping):
    # kwargs values should either be strings or fields of a table. We will have to resolve the fact that all tables
    # need to match the base ggplot table at some point.

    def __init__(self, properties):
        self.properties = properties

    def __getitem__(self, item):
        return self.properties[item]

    def __len__(self):
        return len(self.properties)

    def __iter__(self):
        return iter(self.properties)


def aes(**kwargs):
    return Aesthetic(kwargs)

