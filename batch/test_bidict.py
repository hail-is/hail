import unittest
from bidict import Bidict

class Test(unittest.TestCase):
    def test_bidict(self):
        d = Bidict()
        
        self.assertEqual(len(d), 0)
        
        d['a'] = 5
        d['c'] = 3
        self.assertEqual(len(d), 2)
        self.assertTrue('a' in d)
        self.assertEqual(d['c'], 3)
        self.assertTrue(d.revcontains(5))
        self.assertEqual(d.revgetitem(3), 'c')
        self.assertFalse('b' in d)

        del d['c']
        self.assertEqual(len(d), 1)
        self.assertFalse(d.revcontains(3))
