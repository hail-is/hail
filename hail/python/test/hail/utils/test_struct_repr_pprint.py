import hail as hl
from pprint import pformat


def test_repr_empty_struct():
    assert repr(hl.Struct()) == 'Struct()'


def test_repr_identifier_key_struct():
    assert repr(hl.Struct(x=3)) == 'Struct(x=3)'


def test_repr_two_identifier_keys_struct():
    assert repr(hl.Struct(x=3, y=3)) == 'Struct(x=3, y=3)'


def test_repr_non_identifier_key_struct():
    assert repr(hl.Struct(**{'x': 3, 'y ': 3})) == "Struct(**{'x': 3, 'y ': 3})"


def test_repr_struct_in_struct_all_identifiers():
    assert repr(hl.Struct(x=3, y=3, z=hl.Struct(a=5))) == 'Struct(x=3, y=3, z=Struct(a=5))'


def test_repr_struct_in_struct_some_non_identifiers1():
    assert repr(hl.Struct(x=3, y=3, z=hl.Struct(**{'a ': 5}))) == "Struct(x=3, y=3, z=Struct(**{'a ': 5}))"


def test_repr_struct_in_struct_some_non_identifiers2():
    assert repr(hl.Struct(**{'x': 3, 'y ': 3, 'z': hl.Struct(a=5)})) == "Struct(**{'x': 3, 'y ': 3, 'z': Struct(a=5)})"


def test_pformat_empty_struct():
    assert pformat(hl.Struct()) == 'Struct()'


def test_pformat_identifier_key_struct():
    assert pformat(hl.Struct(x=3)) == 'Struct(x=3)'


def test_pformat_two_identifier_keys_struct():
    assert pformat(hl.Struct(x=3, y=3)) == 'Struct(x=3, y=3)'


def test_pformat_non_identifier_key_struct():
    assert pformat(hl.Struct(**{'x': 3, 'y ': 3})) == "Struct(**{'x': 3, 'y ': 3})"


def test_pformat_struct_in_struct_all_identifiers():
    assert pformat(hl.Struct(x=3, y=3, z=hl.Struct(a=5))) == 'Struct(x=3, y=3, z=Struct(a=5))'


def test_pformat_struct_in_struct_some_non_identifiers1():
    assert pformat(hl.Struct(x=3, y=3, z=hl.Struct(**{'a ': 5}))) == "Struct(x=3, y=3, z=Struct(**{'a ': 5}))"


def test_pformat_struct_in_struct_some_non_identifiers2():
    assert (
        pformat(hl.Struct(**{'x': 3, 'y ': 3, 'z': hl.Struct(a=5)})) == "Struct(**{'x': 3, 'y ': 3, 'z': Struct(a=5)})"
    )


def test_pformat_small_struct_in_big_struct():
    x = hl.Struct(a0=0, a1=1, a2=2, a3=3, a4=4, a5=hl.Struct(b0='', b1='na', b2='nana', b3='nanana'))
    expected = """
Struct(a0=0,
       a1=1,
       a2=2,
       a3=3,
       a4=4,
       a5=Struct(b0='', b1='na', b2='nana', b3='nanana'))
""".strip()
    assert pformat(x) == expected


def test_pformat_big_struct_in_small_struct():
    x = hl.Struct(a5=hl.Struct(b0='', b1='na', b2='nana', b3='nanana', b5='ndasdfhjwafdhjskfdshjkfhdjksfhdsjk'))
    expected = """
Struct(a5=Struct(b0='',
                 b1='na',
                 b2='nana',
                 b3='nanana',
                 b5='ndasdfhjwafdhjskfdshjkfhdjksfhdsjk'))
""".strip()
    assert pformat(x) == expected


def test_pformat_big_struct_in_small_struct():
    x = hl.Struct(a5=hl.Struct(b0='', b1='na', b2='nana', b3='nanana', b5='ndasdfhjwafdhjskfdshjkfhdjksfhdsjk'))
    expected = """
Struct(a5=Struct(b0='',
                 b1='na',
                 b2='nana',
                 b3='nanana',
                 b5='ndasdfhjwafdhjskfdshjkfhdjksfhdsjk'))
""".strip()
    assert pformat(x) == expected


def test_array_of_struct_all_identifier():
    expected = """
[Struct(x=3243),
 Struct(x=3243),
 Struct(x=3243),
 Struct(x=3243),
 Struct(x=3243),
 Struct(x=3243),
 Struct(x=3243),
 Struct(x=3243),
 Struct(x=3243),
 Struct(x=3243)]
""".strip()
    assert pformat([hl.Struct(**{'x': 3243}) for _ in range(10)]) == expected


def test_array_of_struct_non_identifier():
    expected = """
[Struct(**{'x': 3243, 'y ': 123}),
 Struct(**{'x': 3243, 'y ': 123}),
 Struct(**{'x': 3243, 'y ': 123}),
 Struct(**{'x': 3243, 'y ': 123}),
 Struct(**{'x': 3243, 'y ': 123}),
 Struct(**{'x': 3243, 'y ': 123}),
 Struct(**{'x': 3243, 'y ': 123}),
 Struct(**{'x': 3243, 'y ': 123}),
 Struct(**{'x': 3243, 'y ': 123}),
 Struct(**{'x': 3243, 'y ': 123})]
""".strip()
    assert pformat([hl.Struct(**{'x': 3243, 'y ': 123}) for _ in range(10)]) == expected
