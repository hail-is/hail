from collections.abc import Mapping
import itertools


class Aesthetic(Mapping):
    # kwargs values should either be strings or fields of a table. We will have to resolve the fact that all tables
    # need to match the base ggplot table at some point.

    def __init__(self, properties, lambda_properties):
        self.properties = properties
        self.lambda_properties = lambda_properties

    def __getitem__(self, item):
        if item in self.properties:
            return self.properties[item]
        else:
            return self.lambda_properties[item]

    def __len__(self):
        return len(self.properties) + len(self.lambda_properties)

    def __contains__(self, item):
        return item in self.properties or item in self.lambda_properties

    def __iter__(self):
        i1 = iter(self.properties)
        i2 = iter(self.lambda_properties)

        return itertools.chain(i1, i2)

    def __repr__(self):
        return self.properties.__repr__()

    def merge(self, other):
        return Aesthetic({**self.properties, **other.properties}, {**self.lambda_properties, **other.properties})


def aes(**kwargs):
    hail_field_properties = {}
    lambda_properties = {}

    for k, v in kwargs.items():
        if callable(v):
            lambda_properties[k] = v
        else:
            hail_field_properties[k] = v
    return Aesthetic(hail_field_properties, lambda_properties)

