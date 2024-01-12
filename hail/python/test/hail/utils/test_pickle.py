import pickle

import dill

import hail as hl


def test_simple_struct_pickle():
    x = hl.Struct(key="group", value="adj")
    assert x == pickle.loads(pickle.dumps(x))


def test_simple_struct_dill():
    x = hl.Struct(key="group", value="adj")
    assert x == dill.loads(dill.dumps(x))


def test_frozendict_pickle():
    x = {hl.utils.frozendict({"abc": 123, "def": "hello"}): 10}
    assert x == pickle.loads(pickle.dumps(x))


def test_frozendict_dill():
    x = {hl.utils.frozendict({"abc": 123, "def": "hello"}): 10}
    assert x == dill.loads(dill.dumps(x))


def test_locus_pickle():
    x = hl.Locus("1", 123)
    assert x == pickle.loads(pickle.dumps(x))


def test_locus_dill():
    x = hl.Locus("1", 123)
    assert x == dill.loads(dill.dumps(x))
