from hailtop import dictfix


def test_batch_example():
    spec = {
        "input": dictfix.NoneOr({"logs": None}),
        "main": dictfix.NoneOr({"logs": None}),
        "output": dictfix.NoneOr({"logs": None}),
    }

    expected = {"input": None, "main": None, "output": None}

    actual = dict()
    dictfix.dictfix(actual, spec)
    assert actual == expected

    actual = {"input": None}
    dictfix.dictfix(actual, spec)
    assert actual == expected

    actual = {"main": None}
    dictfix.dictfix(actual, spec)
    assert actual == expected

    actual = {"main": None, "output": None}
    dictfix.dictfix(actual, spec)
    assert actual == expected

    expected = {"input": None, "main": {"id": 3, "logs": None}, "output": None}
    actual = {"main": {"id": 3}, "output": None}
    dictfix.dictfix(actual, spec)
    assert actual == expected

    expected = {
        "input": None,
        "main": {"id": 3, "logs": "abc\n123"},
        "output": {"id": 4, "logs": None},
    }
    actual = {"main": {"id": 3, "logs": "abc\n123"}, "output": {"id": 4}}
    dictfix.dictfix(actual, spec)
    assert actual == expected


def test_type_asserts():
    actual = {"x": 1}
    try:
        dictfix.dictfix(actual, {"x": str})
    except AssertionError:
        pass
    else:
        assert False, actual

    actual = {"x": "hi"}
    try:
        dictfix.dictfix(actual, {"x": int})
    except AssertionError:
        pass
    else:
        assert False, actual


def test_defaults():
    actual = {"x": None}
    dictfix.dictfix(actual, {"x": 3})
    assert actual == {"x": 3}

    actual = dict()
    dictfix.dictfix(actual, {"x": 3})
    assert actual == {"x": 3}

    actual = {"y": 1}
    dictfix.dictfix(actual, {"x": 3})
    assert actual == {"x": 3, "y": 1}

    actual = {"x": 0}
    dictfix.dictfix(actual, {"x": 3})
    assert actual == {"x": 0}

    actual = None
    actual = dictfix.dictfix(actual, {"x": 3})
    assert actual == {"x": 3}
