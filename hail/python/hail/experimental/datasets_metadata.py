import json

from hail import __resource


def get_datasets_metadata():
    with __resource('experimental/datasets.json').open('r') as fp:
        return json.load(fp)
