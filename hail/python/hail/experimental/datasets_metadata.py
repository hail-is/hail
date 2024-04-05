import importlib.resources as r
import json


def get_datasets_metadata():
    with r.files('hail.experimental').joinpath("datasets.json").open('r') as fp:
        return json.load(fp)
