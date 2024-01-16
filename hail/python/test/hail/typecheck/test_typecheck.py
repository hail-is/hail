import unittest

from hail.typecheck.check import (
    anytype,
    dictof,
    func_spec,
    lazy,
    nullable,
    numeric,
    oneof,
    sequenceof,
    sized_tupleof,
    transformed,
    tupleof,
    typecheck,
    typecheck_method,
)


class Tests(unittest.TestCase):
    def test_varargs(self):
        @typecheck(x=int, y=int)
        def f1(x, y):
            return x + y

        # ensure that f1 and f2 both run with correct arguments
        f1(1, 2)

        self.assertRaises(TypeError, lambda: f1('2', 3))

        @typecheck(x=int, y=int)
        def bad_signature_1(x, y, *args):
            pass

        @typecheck(x=int, y=int)
        def bad_signature_2(x, y, **kwargs):
            pass

        @typecheck(x=int)
        def bad_signature_3(x, *args, **kwargs):
            pass

        for f in [bad_signature_1, bad_signature_2, bad_signature_3]:
            self.assertRaises(RuntimeError, lambda: f(1, 2))

        @typecheck()
        def f():
            pass

        f()

        @typecheck(x=int, y=int, args=int)
        def good_signature_1(x, y, *args):
            pass

        good_signature_1(1, 2)
        good_signature_1(1, 2, 3)
        good_signature_1(1, 2, 3, 4, 5)

        self.assertRaises(TypeError, lambda: good_signature_1(1, 2, 3, '4'))
        self.assertRaises(TypeError, lambda: good_signature_1(1, 2, '4'))

        @typecheck(x=int, y=int, kwargs=int)
        def good_signature_2(x, y, **kwargs):
            pass

        good_signature_2(1, 2, a=5, z=2)
        good_signature_2(1, 2)
        good_signature_2(1, 2, a=5)

        self.assertRaises(TypeError, lambda: good_signature_2(1, 2, a='2'))
        self.assertRaises(TypeError, lambda: good_signature_2(1, 2, a='2', b=5, c=10))

        @typecheck(x=int, y=int, args=int, kwargs=int)
        def good_signature_3(x, y, *args, **kwargs):
            pass

        good_signature_3(1, 2)
        good_signature_3(1, 2, 3)
        good_signature_3(1, 2, a=3)
        good_signature_3(1, 2, 3, a=4)

        self.assertRaises(TypeError, lambda: good_signature_3(1, 2, a='2'))
        self.assertRaises(TypeError, lambda: good_signature_3(1, 2, '3', b=5, c=10))
        self.assertRaises(TypeError, lambda: good_signature_3(1, 2, '3', b='5', c=10))

        @typecheck(x=int, y=int, args=int, kwargs=oneof(sequenceof(int), str))
        def good_signature_4(x, y, *args, **kwargs):
            pass

        good_signature_4(1, 2)
        good_signature_4(1, 2, 3)
        good_signature_4(1, 2, a='1')
        good_signature_4(1, 2, 3, a=[1, 2, 3])
        good_signature_4(1, 2, 3, a=[1, 2, 3], b='5')
        good_signature_4(1, 2, a=[1, 2, 3], b='5')

        self.assertRaises(TypeError, lambda: good_signature_4(1, 2, a=2))
        self.assertRaises(TypeError, lambda: good_signature_4(1, 2, '3', b='5', c=10))

        @typecheck(x=sized_tupleof(str, int, int))
        def good_signature_5(x):
            pass

        good_signature_5(("1", 5, 10))
        self.assertRaises(TypeError, lambda: good_signature_5("1", 2, 2))
        self.assertRaises(TypeError, lambda: good_signature_5(("1", 5, 10), ("2", 10, 20)))

        @typecheck(x=int, y=str, z=sequenceof(sized_tupleof(str, int, int)), args=int)
        def good_signature_6(x, y, z, *args):
            pass

        good_signature_6(7, "hello", [("1", 5, 10), ("3", 10, 1)], 1, 2, 3)
        good_signature_6(7, "hello", [("1", 5, 10), ("3", 10, 1)])
        good_signature_6(7, "hello", [], 1, 2)
        good_signature_6(7, "hello", [])
        self.assertRaises(TypeError, lambda: good_signature_6(1, "2", ("3", 4, 5)))
        self.assertRaises(TypeError, lambda: good_signature_6(7, "hello", [(9, 5.6, 10), (4, "hello", 1)], 1, 2, 3))

    def test_helpers(self):
        # check nullable
        @typecheck(x=nullable(int))
        def f(x):
            pass

        f(5)
        f(None)
        self.assertRaises(TypeError, lambda: f('2'))

        # check integral
        @typecheck(x=int)
        def f(x):
            pass

        f(1)
        self.assertRaises(TypeError, lambda: f(1.1))

        # check numeric
        @typecheck(x=numeric)
        def f(x):
            pass

        f(1)
        f(1.0)
        self.assertRaises(TypeError, lambda: f('1.1'))

        # check strlike
        @typecheck(x=str)
        def f(x):
            pass

        f('str')
        f('unicode')
        self.assertRaises(TypeError, lambda: f(['abc']))

    def test_nested(self):
        @typecheck(x=int, y=oneof(nullable(str), sequenceof(sequenceof(dictof(oneof(str, int), anytype)))))
        def f(x, y):
            pass

        f(5, None)
        f(5, '7')
        f(5, [])
        f(5, [[]])
        f(5, [[{}]])
        f(5, [[{'6': None}]])
        f(5, [[{'6': None}]])
        f(5, [[{'6': None, 5: {1, 2, 3, 4}}]])
        self.assertRaises(TypeError, lambda: f(2, 2))

    def test_class_methods(self):
        class Foo:
            @typecheck_method(a=int, b=str)
            def __init__(self, a, b):
                pass

            @typecheck_method(x=int, y=int)
            def a(self, x, y):
                pass

            @staticmethod
            @typecheck(x=int, y=int)
            def b(x, y):
                pass

            # error because it should be typecheck_method
            @typecheck(x=int, y=int)
            def c(self, x, y):
                pass

            @typecheck_method(x=int, y=int, args=str, kwargs=int)
            def d(self, x, y, *args, **kwargs):
                pass

        Foo(2, '2')

        self.assertRaises(TypeError, lambda: Foo('2', '2'))

        f = Foo(2, '2')

        f.a(2, 2)
        f.b(2, 2)
        Foo.b(2, 2)
        f.d(1, 2)
        f.d(1, 2, '3')
        f.d(1, 2, '3', z=5)

        self.assertRaises(TypeError, lambda: f.a('2', '2'))
        self.assertRaises(TypeError, lambda: f.b('2', '2'))
        self.assertRaises(TypeError, lambda: Foo.b('2', '2'))
        self.assertRaises(RuntimeError, lambda: f.c(2, 2))
        self.assertRaises(TypeError, lambda: f.d(2, 2, 3))
        self.assertRaises(TypeError, lambda: f.d(2, 2, z='2'))

    def test_lazy(self):
        foo_type = lazy()

        class Foo:
            def __init__(self):
                pass

            @typecheck_method(other=foo_type)
            def bar(self, other):
                pass

        foo_type.set(Foo)

        foo = Foo()
        foo2 = Foo()

        foo.bar(foo)
        foo.bar(foo2)

        self.assertRaises(TypeError, lambda: foo.bar(2))

    def test_coercion(self):
        @typecheck(
            a=transformed((int, lambda x: 'int'), (str, lambda x: 'str')),
            b=sequenceof(dictof(str, transformed((int, lambda x: 'int'), (str, lambda x: 'str')))),
        )
        def foo(a, b):
            return a, b

        self.assertRaises(TypeError, lambda: foo(5.5, [{'5': 5}]))
        self.assertRaises(TypeError, lambda: foo(5, [{'5': 5.5}]))

        a, b = foo(5, [])
        self.assertEqual(a, 'int')

        a, b = foo('5', [])
        self.assertEqual(a, 'str')

        a, b = foo(5, [{'5': 5, '6': '6'}, {'10': 10}])
        self.assertEqual(a, 'int')
        self.assertEqual(b, [{'5': 'int', '6': 'str'}, {'10': 'int'}])

    def test_function_checker(self):
        @typecheck(f=func_spec(3, int))
        def foo(f):
            return f(1, 2, 3)

        l1 = lambda: 5
        l2 = 5
        l3 = lambda x, y, z: x + y + z

        self.assertRaises(TypeError, lambda: foo(l1))
        self.assertRaises(TypeError, lambda: foo(l2))
        foo(l3)

        @typecheck(f=func_spec(0, int))
        def eval(f):
            return f()

        self.assertEqual(eval(lambda x=1: x), 1)
        self.assertEqual(eval(lambda x=None: 1), 1)
        self.assertRaises(TypeError, lambda: eval(lambda x: 1))

        @typecheck(f=func_spec(2, int), a=int, b=int)
        def apply(f, a, b):
            return f(a, b)

        self.assertEqual(apply(lambda x, y=2, z=3: x + y + z, 5, 7), 15)

    def test_complex_signature(self):
        @typecheck(a=int, b=str, c=sequenceof(int), d=tupleof(str), e=dict)
        def f(a, b='5', c=[10], *d, **e):
            pass

        f(
            1,
            'a',
        )
        f(1, foo={})
        f(1, 'a', foo={})
        f(1, c=[25, 2])
        with self.assertRaises(TypeError):
            f(1, '2', a=2)

    def test_extra_args(self):
        @typecheck(x=int)
        def f(x):
            pass

        f(1)
        with self.assertRaises(TypeError):
            f(1, 2)
