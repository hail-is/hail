import unittest
from check import *


class ContextTests(unittest.TestCase):
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

        @typecheck(x=int, y=int, args=tupleof(int))
        def good_signature_1(x, y, *args):
            pass

        good_signature_1(1, 2)
        good_signature_1(1, 2, 3)
        good_signature_1(1, 2, 3, 4, 5)

        self.assertRaises(TypeError, lambda: good_signature_1(1, 2, 3, '4'))
        self.assertRaises(TypeError, lambda: good_signature_1(1, 2, '4'))

        @typecheck(x=int, y=int, kwargs=dictof(strlike, int))
        def good_signature_2(x, y, **kwargs):
            pass

        good_signature_2(1, 2, a=5, z=2)
        good_signature_2(1, 2)
        good_signature_2(1, 2, a=5)

        self.assertRaises(TypeError, lambda: good_signature_2(1, 2, a='2'))
        self.assertRaises(TypeError, lambda: good_signature_2(1, 2, a='2', b=5, c=10))

        @typecheck(x=int, y=int, args=tupleof(int), kwargs=dictof(strlike, int))
        def good_signature_3(x, y, *args, **kwargs):
            pass

        good_signature_3(1, 2)
        good_signature_3(1, 2, 3)
        good_signature_3(1, 2, a=3)
        good_signature_3(1, 2, 3, a=4)

        self.assertRaises(TypeError, lambda: good_signature_3(1, 2, a='2'))
        self.assertRaises(TypeError, lambda: good_signature_3(1, 2, '3', b=5, c=10))
        self.assertRaises(TypeError, lambda: good_signature_3(1, 2, '3', b='5', c=10))

        @typecheck(x=int, y=int, args=tupleof(int), kwargs=dictof(strlike, oneof(listof(int), strlike)))
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



    def test_helpers(self):
        # check nullable
        @typecheck(x=nullable(int))
        def f(x):
            pass

        f(5)
        f(None)
        self.assertRaises(TypeError, lambda: f('2'))

        # check integral
        @typecheck(x=integral)
        def f(x):
            pass

        f(1)
        f(1L)
        self.assertRaises(TypeError, lambda: f(1.1))

        # check numeric
        @typecheck(x=numeric)
        def f(x):
            pass

        f(1)
        f(1.0)
        f(1L)
        self.assertRaises(TypeError, lambda: f('1.1'))

        # check strlike
        @typecheck(x=strlike)
        def f(x):
            pass

        f('str')
        f(u'unicode')
        self.assertRaises(TypeError, lambda: f(['abc']))

    def test_nested(self):
        @typecheck(
            x=int,
            y=oneof(nullable(strlike), listof(listof(dictof(oneof(strlike, int), anytype))))
        )
        def f(x, y):
            pass

        f(5, None)
        f(5, u'7')
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
                self.a = a
                self.b = b

            @typecheck_method(x=int, y=int)
            def bar(self, x, y):
                pass

            @staticmethod
            @typecheck(x=int, y=int)
            def baz(x, y):
                pass

            @typecheck(x=int, y=int)
            def qux(self, x, y):
                pass

        Foo(2, '2')

        self.assertRaises(TypeError, lambda: Foo('2', '2'))

        f = Foo(2, '2')

        f.bar(2, 2)
        f.baz(2, 2)
        Foo.baz(2, 2)

        self.assertRaises(TypeError, lambda: f.bar('2', '2'))
        self.assertRaises(TypeError, lambda: f.baz('2', '2'))
        self.assertRaises(TypeError, lambda: Foo.baz('2', '2'))
        self.assertRaises(RuntimeError, lambda: f.qux(2, 2))
