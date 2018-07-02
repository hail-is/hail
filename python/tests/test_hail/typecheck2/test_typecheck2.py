import unittest
from typing import *

from hail.typecheck2 import typecheck


class Tests(unittest.TestCase):

    def test_noargs(self):
        def f_noargs():
            typecheck(f_noargs)

        f_noargs()

    def test_simple(self):
        def f_simple1(x: int):
            typecheck(f_simple1)

        def f_simple2(x: str):
            typecheck(f_simple2)

        f_simple1(1)
        f_simple2('a')

        with self.assertRaises(TypeError):
            f_simple1('1')
        with self.assertRaises(TypeError):
            f_simple2(1)
        with self.assertRaises(TypeError):
            f_simple2(None)

    def test_optional(self):
        def f_optional(x: Optional[str]):
            typecheck(f_optional)

        f_optional('')
        f_optional('a')
        f_optional(None)

        with self.assertRaises(TypeError):
            f_optional(1)

    def test_float(self):
        def f_float(x: float):
            typecheck(f_float)

        f_float(1)
        f_float(1.5)

        with self.assertRaises(TypeError):
            f_float([])

    def test_union(self):
        def f_union(x: Union[int, str]):
            typecheck(f_union)

        f_union(1)
        f_union('1')

        with self.assertRaises(TypeError):
            f_union(1.2)

    def test_list(self):
        def f_list(x: List):
            typecheck(f_list)

        def f_list_int(x: List[int]):
            typecheck(f_list_int)

        f_list([])
        f_list(['a'])
        f_list(['a', 1, None])

        f_list_int([])
        f_list_int([1, 2, 3])

        with self.assertRaises(TypeError):
            f_list({})
        with self.assertRaises(TypeError):
            f_list(tuple([1, 2, 3]))
        with self.assertRaises(TypeError):
            f_list_int(['1'])
        with self.assertRaises(TypeError):
            f_list_int(tuple([]))
        with self.assertRaises(TypeError):
            f_list_int(tuple([1]))

    def test_sequence(self):
        def f_sequence(x: Sequence):
            typecheck(f_sequence)

        def f_sequence_float(x: Sequence[float]):
            typecheck(f_sequence_float)

        f_sequence([])
        f_sequence(tuple())

        f_sequence_float([])
        f_sequence_float(tuple())
        f_sequence_float([1, 1.5, 2])
        f_sequence_float((1, 1.5, 2))

        with self.assertRaises(TypeError):
            f_sequence({})
        with self.assertRaises(TypeError):
            f_sequence(set([1, 2, 3]))
        with self.assertRaises(TypeError):
            f_sequence_float(['1'])
        with self.assertRaises(TypeError):
            f_sequence_float(set())
        with self.assertRaises(TypeError):
            f_sequence_float(tuple(['1']))

    def test_set(self):
        def f_set(x: Set):
            typecheck(f_set)

        def f_set_union(x: Set[Union[int, str]]):
            typecheck(f_set_union)

        f_set(set())
        f_set({1, 2, 3})
        f_set({1, 'a', None})

        f_set_union(set())
        f_set_union({'1'})
        f_set_union({'1', 1})
        f_set_union({1, 2, 3, 4})

        with self.assertRaises(TypeError):
            f_set([])
        with self.assertRaises(TypeError):
            f_set(())
        with self.assertRaises(TypeError):
            f_set(frozenset())
        with self.assertRaises(TypeError):
            f_set_union({1, 1.5})

    def test_frozenset(self):
        def f_frozenset(x: FrozenSet):
            typecheck(f_frozenset)

        def f_frozenset_int(x: FrozenSet[int]):
            typecheck(f_frozenset_int)

        f_frozenset(frozenset())
        f_frozenset(frozenset([1, 2, 3]))

        f_frozenset_int(frozenset())
        f_frozenset_int(frozenset([1, 2, 3]))

        with self.assertRaises(TypeError):
            f_frozenset(set())
        with self.assertRaises(TypeError):
            f_frozenset([])
        with self.assertRaises(TypeError):
            f_frozenset_int(set())
        with self.assertRaises(TypeError):
            f_frozenset_int(frozenset(['1']))

    def test_collection(self):
        def f_collection(x: Collection):
            typecheck(f_collection)

        def f_collection_str(x: Collection[str]):
            typecheck(f_collection_str)

        f_collection([])
        f_collection([x for x in []])
        f_collection([1, 2, 3])
        f_collection({1, 2, 3})
        f_collection({})
        f_collection({}.keys())
        f_collection({}.items())

        f_collection_str(['1', '2', '3'])
        f_collection_str(tuple(str(x) for x in [1, 2, 3]))

        with self.assertRaises(TypeError):
            f_collection(1)
        with self.assertRaises(TypeError):
            f_collection_str(1)
        with self.assertRaises(TypeError):
            f_collection_str([1, 2, 3])

    def test_tuple(self):
        def f_tuple(x: Tuple):
            typecheck(f_tuple)

        def f_tuple2(x: Tuple[Any, ...]):
            typecheck(f_tuple2)

        def f_tuple_var(x: Tuple[int, ...]):
            typecheck(f_tuple_var)

        def f_tuple_sized1(x: Tuple[int]):
            typecheck(f_tuple_sized1)

        def f_tuple_sized2(x: Tuple[int, str, Union[str, int]]):
            typecheck(f_tuple_sized2)

        f_tuple(tuple())
        f_tuple(tuple([1, 2, 3, 4]))
        f_tuple(tuple([1, 2, 3, '4']))

        f_tuple2(tuple())
        f_tuple2(tuple([1, 2, 3, 4]))
        f_tuple2(tuple([1, 2, 3, '4']))

        f_tuple_sized1(tuple([1]))

        f_tuple_sized2((1, '1', 1))
        f_tuple_sized2((1, '1', '2'))

        with self.assertRaises(TypeError):
            f_tuple([])
        with self.assertRaises(TypeError):
            f_tuple2([])
        with self.assertRaises(TypeError):
            f_tuple_sized1(())
        with self.assertRaises(TypeError):
            f_tuple_sized1(tuple([1, 2]))
        with self.assertRaises(TypeError):
            f_tuple_sized1(tuple([1, 2, 3]))
        with self.assertRaises(TypeError):
            f_tuple_sized1(tuple(['1']))
        with self.assertRaises(TypeError):
            f_tuple_sized2(1, 1, 1.5)
        with self.assertRaises(TypeError):
            f_tuple_sized2(1, '1')
        with self.assertRaises(TypeError):
            f_tuple_sized2(1, '1', '1')

    def test_dict(self):
        def f_dict(x: Dict):
            typecheck(f_dict)

        def f_dict_str_int(x: Dict[str, int]):
            typecheck(f_dict_str_int)

        f_dict({})
        f_dict({'1': 1})
        f_dict({1: 1})

        f_dict_str_int({})
        f_dict_str_int({'1': 1})

        with self.assertRaises(TypeError):
            f_dict([])
        with self.assertRaises(TypeError):
            f_dict(set())
        with self.assertRaises(TypeError):
            f_dict_str_int({1: 1})
        with self.assertRaises(TypeError):
            f_dict_str_int({'1': '1'})

    def test_mapping(self):
        def f_mapping(x: Mapping):
            typecheck(f_mapping)

        def f_mapping_str_int(x: Mapping[str, int]):
            typecheck(f_mapping_str_int)

        f_mapping({})
        f_mapping({'1': 1})
        f_mapping({1: 1})

        f_mapping_str_int({})
        f_mapping_str_int({'1': 1})

        with self.assertRaises(TypeError):
            f_mapping([])
        with self.assertRaises(TypeError):
            f_mapping(set())
        with self.assertRaises(TypeError):
            f_mapping_str_int({1: 1})
        with self.assertRaises(TypeError):
            f_mapping_str_int({'1': '1'})

    def test_varargs(self):
        def f_varargs(x: int, *xs: str):
            typecheck(f_varargs)

        def f_varargs2(*xs: str):
            typecheck(f_varargs2)

        def f_varargs3(x: int, *xs: Union[str, int]):
            typecheck(f_varargs3)

        with self.assertRaises(TypeError):
            f_varargs('1')
        with self.assertRaises(TypeError):
            f_varargs(1, 1)
        with self.assertRaises(TypeError):
            f_varargs2(1)
        with self.assertRaises(TypeError):
            f_varargs2('1', 1, '1')
        with self.assertRaises(TypeError):
            f_varargs3('1')
        with self.assertRaises(TypeError):
            f_varargs3(1, 1.5)

        f_varargs(1)
        f_varargs(1, '1')
        f_varargs(1, '1', '2', '3')

        f_varargs2()
        f_varargs2('1', '2')
        f_varargs2(*['1'])

        f_varargs3(1)
        f_varargs3(1, 1)
        f_varargs3(1, '1')
        f_varargs3(1, '1', 1, '2', 2)

    def test_kwargs(self):
        def f_kwargs(x: int, **xs: int):
            typecheck(f_kwargs)

        def f_kwargs2(x: int, **xs: List[Union[str, int]]):
            typecheck(f_kwargs2)

        f_kwargs(1, xx=1, y=2, **{'foo bar baz': 1})
        f_kwargs2(1, xx=[1], y=[], **{'foo bar baz': ['1', 1]})

        with self.assertRaises(TypeError):
            f_kwargs('1')
        with self.assertRaises(TypeError):
            f_kwargs(y='s')
        with self.assertRaises(TypeError):
            f_kwargs(y=1, z=2, bad=1.5)
        with self.assertRaises(TypeError):
            f_kwargs2('1')
        with self.assertRaises(TypeError):
            f_kwargs2(1, x=[], y=[1.5])

    def test_callable(self):
        def f_callable(x: Callable[[int, str], str]):
            typecheck(f_callable)

        f_callable(lambda x, y: x + y)

        with self.assertRaises(TypeError):
            f_callable(lambda x: x)
        with self.assertRaises(TypeError):
            f_callable(lambda x, y, z: x)
        with self.assertRaises(TypeError):
            def f(*xs):
                pass

            f_callable(f)
        with self.assertRaises(TypeError):
            def f(x, y, *, z=5):
                pass

            f_callable(f)
        with self.assertRaises(TypeError):
            def f(x, y, **kw):
                pass

            f_callable(f)

    def test_complicated(self):
        def f_complicated(x: List[List[Set[Union[str, int]]]],
                          y: Dict[Union[int, str], Sequence[Optional[int]]]):
            typecheck(f_complicated)

        f_complicated([[{1, 2}]], {'s': [None, 1]})
        with self.assertRaises(TypeError):
            f_complicated([[{1, '1'}]], {'1': [1, None, '1']})
