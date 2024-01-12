from hailtop import yamlx


def test_multiline_str_is_literal_block():
    actual = yamlx.dump({"hello": "abc", "multiline": "abc\ndef"})
    expected = """hello: abc
multiline: |-
  abc
  def
"""
    assert actual == expected
