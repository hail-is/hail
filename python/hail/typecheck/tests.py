import unittest
from check import *


class ContextTests(unittest.TestCase):
    def test_errors(self):
        @typecheck(x=int, y=int)
        def f1(x, y):
            return x + y

        # ensure that f1 and f2 both run with correct arguments
        f1(1, 2)

        self.assertRaises(TypeError, lambda: f1('2', 3))

        # ensure that functions with args and kwargs raise exceptions
        @typecheck()
        def bad_signature_1(x, y, *args):
            pass

        @typecheck()
        def bad_signature_2(x, y, **kwargs):
            pass

        @typecheck()
        def bad_signature_3(*args, **kwargs):
            pass

        for f in [bad_signature_1, bad_signature_2, bad_signature_3]:
            self.assertRaises(RuntimeError, lambda: f(1, 2))

        @typecheck()
        def f():
            pass

        f()

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
        class Foo(object):
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

